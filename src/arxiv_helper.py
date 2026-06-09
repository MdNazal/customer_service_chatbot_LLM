import os
import csv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from dotenv import load_dotenv
from language_helper import build_multilingual_prompt
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Source: https://www.kaggle.com/datasets/Cornell-University/arxiv
# Filtered to computer science papers only

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.environ["GROQ_API_KEY"],
    temperature=0.2
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb_file_path = "faiss_index_arxiv"
ARXIV_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "arxiv_cs.csv")


def load_arxiv_documents(max_rows=5000):
    documents = []
    with open(ARXIV_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            title = row.get("title", "").strip()
            abstract = row.get("abstract", "").strip()
            authors = row.get("authors", "").strip()
            paper_id = row.get("id", "").strip()
            categories = row.get("categories", "").strip()

            if title and abstract:
                content = f"Title: {title}\nAuthors: {authors}\nCategories: {categories}\nAbstract: {abstract}"
                doc = Document(
                    page_content=content,
                    metadata={"id": paper_id, "title": title, "authors": authors, "categories": categories}
                )
                documents.append(doc)

    print(f"Loaded {len(documents)} arXiv CS papers.")
    return documents


def create_arxiv_vector_db():
    documents = load_arxiv_documents(max_rows=5000)
    print("Building FAISS index...")
    vectordb = FAISS.from_documents(documents=documents, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)
    print("arXiv knowledgebase created successfully.")


def get_arxiv_qa_chain(lang_code="en"):
    vectordb = FAISS.load_local(
        vectordb_file_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    base_prompt = """You are an expert research assistant specializing in computer science.
You have access to abstracts from arXiv research papers.
Use the following research paper context to answer the question thoroughly.
When relevant, mention specific paper titles and explain concepts clearly.
If asked to summarize, provide a concise but comprehensive summary.
If the answer is not in the context, say so honestly and provide general knowledge if possible.

RESEARCH CONTEXT:
{context}

QUESTION: {question}

EXPERT ANSWER:"""

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


def search_papers(query, k=5):
    vectordb = FAISS.load_local(
        vectordb_file_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    results = vectordb.similarity_search(query, k=k)
    papers = []
    for doc in results:
        papers.append({
            "title": doc.metadata.get("title", "Unknown"),
            "authors": doc.metadata.get("authors", "Unknown"),
            "categories": doc.metadata.get("categories", ""),
            "id": doc.metadata.get("id", ""),
            "abstract": doc.page_content.split("Abstract:")[-1].strip()
        })
    return papers

def summarize_paper(title_or_query):
    """
    Find a paper by title/query and return a structured summary
    using LangChain's map-reduce summarization chain.
    """
    vectordb = FAISS.load_local(
        vectordb_file_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Find most relevant paper
    results = vectordb.similarity_search(title_or_query, k=3)

    if not results:
        return "No relevant papers found for summarization."

    # Split text for summarization
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(results)

    # Use map-reduce summarization chain
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce"
    )

    summary = summary_chain.run(split_docs)
    return summary

def extract_paper_info(query):
    """
    Extract structured information from retrieved papers:
    - Main topic/problem
    - Methodology used
    - Key findings/results
    - Contributions
    """
    vectordb = FAISS.load_local(
        vectordb_file_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    results = vectordb.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""Given these research paper abstracts, extract and structure the following information:

PAPERS:
{context}

Extract:
1. Main Problem/Topic being addressed
2. Methodology/Approach used
3. Key Results/Findings
4. Main Contributions to the field

Format your response clearly with these four sections."""

    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content