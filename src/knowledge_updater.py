import os
import csv
import time
import json
import shutil
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")
NULLCLASS_CSV = os.path.join(DATASET_DIR, "dataset.csv")
MEDQUAD_CSV = os.path.join(DATASET_DIR, "medquad.csv")
NULLCLASS_INDEX = "faiss_index"
MEDICAL_INDEX = "faiss_index_medical"
METADATA_FILE = os.path.join(BASE_DIR, "../dataset/update_metadata.json")


def load_metadata():
    """Load metadata about last update times and row counts."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {
        "nullclass": {"last_updated": None, "row_count": 0},
        "medical": {"last_updated": None, "row_count": 0}
    }


def save_metadata(metadata):
    """Save metadata to disk."""
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)


def get_row_count(csv_path):
    """Count rows in a CSV file."""
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, "r", encoding="cp1252") as f:
        return sum(1 for _ in f) - 1  # subtract header


def merge_new_csv(existing_csv, new_csv_content, mode="nullclass"):
    """
    Merge new CSV content into the existing dataset.
    new_csv_content is a list of dicts with 'prompt' and 'response' keys.
    Returns number of new rows added.
    """
    # Read existing prompts to avoid duplicates
    existing_prompts = set()
    if os.path.exists(existing_csv):
        with open(existing_csv, "r", encoding="cp1252", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_prompts.add(row.get("prompt", "").strip().lower())

    # Filter out duplicates
    new_rows = []
    for row in new_csv_content:
        prompt = row.get("prompt", "").strip()
        response = row.get("response", "").strip()
        if prompt and response and prompt.lower() not in existing_prompts:
            new_rows.append((prompt, response))
            existing_prompts.add(prompt.lower())

    if not new_rows:
        return 0

    # Append to existing CSV
    with open(existing_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for prompt, response in new_rows:
            writer.writerow([prompt, response])

    return len(new_rows)


def rebuild_index(csv_path, index_path, max_rows=5000):
    """Rebuild FAISS index from CSV."""
    loader = CSVLoader(file_path=csv_path, source_column="prompt", encoding="cp1252")
    data = loader.load()
    data = data[:max_rows]

    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(index_path)
    return len(data)


def update_nullclass_knowledge(new_rows):
    """
    Update Nullclass knowledgebase with new QA pairs.
    new_rows: list of dicts [{"prompt": "...", "response": "..."}]
    Returns (rows_added, total_rows)
    """
    rows_added = merge_new_csv(NULLCLASS_CSV, new_rows, mode="nullclass")

    if rows_added > 0:
        total = rebuild_index(NULLCLASS_CSV, NULLCLASS_INDEX)
        metadata = load_metadata()
        metadata["nullclass"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata["nullclass"]["row_count"] = total
        save_metadata(metadata)
        return rows_added, total

    return 0, get_row_count(NULLCLASS_CSV)


def update_medical_knowledge(new_rows):
    """
    Update Medical knowledgebase with new QA pairs.
    new_rows: list of dicts [{"prompt": "...", "response": "..."}]
    Returns (rows_added, total_rows)
    """
    rows_added = merge_new_csv(MEDQUAD_CSV, new_rows, mode="medical")

    if rows_added > 0:
        total = rebuild_index(MEDQUAD_CSV, MEDICAL_INDEX)
        metadata = load_metadata()
        metadata["medical"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata["medical"]["row_count"] = total
        save_metadata(metadata)
        return rows_added, total

    return 0, get_row_count(MEDQUAD_CSV)


def check_for_updates(mode="nullclass"):
    """
    Check if the dataset has been updated since last index build.
    Returns True if update is needed.
    """
    metadata = load_metadata()
    key = "nullclass" if mode == "nullclass" else "medical"
    csv_path = NULLCLASS_CSV if mode == "nullclass" else MEDQUAD_CSV
    index_path = NULLCLASS_INDEX if mode == "nullclass" else MEDICAL_INDEX

    # If no index exists, update needed
    if not os.path.exists(index_path):
        return True

    current_count = get_row_count(csv_path)
    last_count = metadata[key].get("row_count", 0)

    return current_count > last_count


def auto_update_if_needed(mode="nullclass"):
    """
    Automatically rebuild index if new data detected.
    Returns (updated: bool, message: str)
    """
    if check_for_updates(mode):
        csv_path = NULLCLASS_CSV if mode == "nullclass" else MEDQUAD_CSV
        index_path = NULLCLASS_INDEX if mode == "nullclass" else MEDICAL_INDEX

        total = rebuild_index(csv_path, index_path)

        metadata = load_metadata()
        key = "nullclass" if mode == "nullclass" else "medical"
        metadata[key]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata[key]["row_count"] = total
        save_metadata(metadata)

        return True, f"Auto-updated: {total} documents indexed."

    return False, "Knowledge base is up to date."


def get_last_updated(mode="nullclass"):
    """Get last updated timestamp for display."""
    metadata = load_metadata()
    key = "nullclass" if mode == "nullclass" else "medical"
    last = metadata[key].get("last_updated")
    count = metadata[key].get("row_count", 0)
    return last, count
