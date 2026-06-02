import os
import csv
import xml.etree.ElementTree as ET
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from language_helper import build_multilingual_prompt

load_dotenv()

# Source: https://github.com/abachaa/MedQuAD
VALID_FOLDERS = [
    "1_CancerGov_QA", "2_GARD_QA", "3_GHR_QA",
    "4_MPlus_Health_Topics_QA", "5_NIDDK_QA", "6_NINDS_QA",
    "7_SeniorHealth_QA", "8_NHLBI_QA_XML", "9_CDC_QA"
]

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.environ["GROQ_API_KEY"],
    temperature=0.1
)

instructor_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb_file_path = "faiss_index_medical"


def parse_medquad_xml(xml_path):
    qa_pairs = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for qapair in root.iter("QAPair"):
            question_el = qapair.find("Question")
            answer_el = qapair.find("Answer")
            if question_el is not None and answer_el is not None:
                question = (question_el.text or "").strip()
                answer = (answer_el.text or "").strip()
                if question and answer:
                    qa_pairs.append((question, answer))
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
    return qa_pairs


def build_medquad_csv(medquad_path, output_csv_path):
    all_pairs = []
    for folder in VALID_FOLDERS:
        folder_path = os.path.join(medquad_path, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue
        xml_files = [f for f in os.listdir(folder_path) if f.endswith(".xml")]
        print(f"Parsing {len(xml_files)} files from {folder}...")
        for xml_file in xml_files:
            pairs = parse_medquad_xml(os.path.join(folder_path, xml_file))
            all_pairs.extend(pairs)

    print(f"Total QA pairs extracted: {len(all_pairs)}")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "response"])
        for question, answer in all_pairs:
            writer.writerow([question, answer])
    return output_csv_path


def create_medical_vector_db():
    medquad_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dataset/MedQuAD")
    output_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dataset/medquad.csv")

    if not os.path.exists(medquad_path):
        raise FileNotFoundError(
            f"MedQuAD folder not found at: {medquad_path}\n"
            "Please place the MedQuAD dataset folders inside dataset/MedQuAD/"
        )

    if not os.path.exists(output_csv_path):
        build_medquad_csv(medquad_path, output_csv_path)
    else:
        print("medquad.csv already exists, skipping XML parsing.")

    from langchain_community.document_loaders.csv_loader import CSVLoader
    loader = CSVLoader(file_path=output_csv_path, source_column="prompt", encoding="utf-8")
    data = loader.load()
    data = data[:5000]
    print(f"Building FAISS index from {len(data)} QA pairs...")
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)
    print("Medical knowledgebase created successfully.")


def get_medical_qa_chain(lang_code="en"):
    vectordb = FAISS.load_local(
        vectordb_file_path,
        instructor_embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(score_threshold=0.7)

    base_prompt = """You are a medical information assistant using the MedQuAD dataset from NIH sources.
Use the following context to answer the medical question accurately.
If the answer is not found in the context, say "I don't have specific information about that. Please consult a healthcare professional."
Always remind users to consult a doctor for personal medical advice.

CONTEXT: {context}

QUESTION: {question}

MEDICAL ANSWER:"""

    final_prompt = build_multilingual_prompt(base_prompt, lang_code)

    PROMPT = PromptTemplate(
        template=final_prompt, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    return chain
