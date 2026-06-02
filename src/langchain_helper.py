import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from language_helper import build_multilingual_prompt

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.environ["GROQ_API_KEY"],
    temperature=0.1
)

instructor_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectordb_file_path = "faiss_index"


def create_vector_db():
    loader = CSVLoader(
        file_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "dataset.csv"),
        source_column="prompt",
        encoding="cp1252"
    )
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)


def get_qa_chain(lang_code="en"):
    vectordb = FAISS.load_local(
        vectordb_file_path,
        instructor_embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(score_threshold=0.7)

    base_prompt = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

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
