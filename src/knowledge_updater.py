import os
import csv
import json
import requests
import threading
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
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
METADATA_FILE = os.path.join(DATASET_DIR, "update_metadata.json")

# External source URLs for periodic updates
# These are public GitHub raw CSV URLs that serve as external data sources
EXTERNAL_SOURCES = {
    "nullclass": "https://raw.githubusercontent.com/MdNazal/customer_service_chatbot_LLM/main/dataset/dataset.csv",
    "medical": "https://raw.githubusercontent.com/MdNazal/customer_service_chatbot_LLM/main/dataset/medquad.csv"
}

# Global scheduler instance
_scheduler = None
_scheduler_lock = threading.Lock()


def load_metadata():
    """Load metadata about last update times and row counts."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {
        "nullclass": {"last_updated": None, "row_count": 0, "last_external_fetch": None},
        "medical": {"last_updated": None, "row_count": 0, "last_external_fetch": None}
    }


def save_metadata(metadata):
    """Save metadata to disk."""
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)


def get_row_count(csv_path):
    """Count rows in a CSV file."""
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, "r", encoding="cp1252", errors="ignore") as f:
        return sum(1 for _ in f) - 1


def fetch_from_external_source(mode="nullclass"):
    """
    Fetch new QA data from external URL source.
    Returns list of new rows as dicts with prompt and response keys.
    """
    url = EXTERNAL_SOURCES.get(mode)
    if not url:
        return []

    try:
        print(f"Fetching from external source: {url}")
        response = requests.get(url, timeout=15)

        if response.status_code == 200:
            lines = response.text.splitlines()
            reader = csv.DictReader(lines)
            rows = [row for row in reader if row.get("prompt") and row.get("response")]
            print(f"Fetched {len(rows)} rows from external source.")
            return rows
        else:
            print(f"External fetch failed with status: {response.status_code}")
            return []

    except requests.exceptions.RequestException as e:
        print(f"Error fetching from external source: {e}")
        return []


def merge_new_csv(existing_csv, new_rows):
    """
    Merge new rows into existing CSV, deduplicating by prompt.
    Returns number of new rows added.
    """
    existing_prompts = set()
    if os.path.exists(existing_csv):
        try:
            with open(existing_csv, "r", encoding="cp1252", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_prompts.add(row.get("prompt", "").strip().lower())
        except Exception:
            pass

    new_unique_rows = []
    for row in new_rows:
        prompt = row.get("prompt", "").strip()
        response = row.get("response", "").strip()
        if prompt and response and prompt.lower() not in existing_prompts:
            new_unique_rows.append((prompt, response))
            existing_prompts.add(prompt.lower())

    if not new_unique_rows:
        return 0

    with open(existing_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for prompt, response in new_unique_rows:
            writer.writerow([prompt, response])

    return len(new_unique_rows)


def rebuild_index(csv_path, index_path, max_rows=5000):
    """Rebuild FAISS index from CSV."""
    loader = CSVLoader(file_path=csv_path, source_column="prompt", encoding="cp1252")
    data = loader.load()
    data = data[:max_rows]
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(index_path)
    return len(data)


def update_nullclass_knowledge(new_rows):
    """Update Nullclass knowledgebase with new QA pairs."""
    rows_added = merge_new_csv(NULLCLASS_CSV, new_rows)
    if rows_added > 0:
        total = rebuild_index(NULLCLASS_CSV, NULLCLASS_INDEX)
        metadata = load_metadata()
        metadata["nullclass"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata["nullclass"]["row_count"] = total
        save_metadata(metadata)
        return rows_added, total
    return 0, get_row_count(NULLCLASS_CSV)


def update_medical_knowledge(new_rows):
    """Update Medical knowledgebase with new QA pairs."""
    rows_added = merge_new_csv(MEDQUAD_CSV, new_rows)
    if rows_added > 0:
        total = rebuild_index(MEDQUAD_CSV, MEDICAL_INDEX)
        metadata = load_metadata()
        metadata["medical"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata["medical"]["row_count"] = total
        save_metadata(metadata)
        return rows_added, total
    return 0, get_row_count(MEDQUAD_CSV)


def periodic_external_update(mode="nullclass"):
    """
    Called by APScheduler periodically.
    Fetches data from external source and updates knowledge base if new data found.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running periodic update for {mode}...")

    new_rows = fetch_from_external_source(mode)
    if not new_rows:
        print(f"No new data fetched for {mode}.")
        return

    csv_path = NULLCLASS_CSV if mode == "nullclass" else MEDQUAD_CSV
    index_path = NULLCLASS_INDEX if mode == "nullclass" else MEDICAL_INDEX

    rows_added = merge_new_csv(csv_path, new_rows)

    if rows_added > 0:
        rebuild_index(csv_path, index_path)
        metadata = load_metadata()
        metadata[mode]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata[mode]["row_count"] = get_row_count(csv_path)
        metadata[mode]["last_external_fetch"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_metadata(metadata)
        print(f"Added {rows_added} new rows for {mode} from external source.")
    else:
        # Update fetch timestamp even if no new rows
        metadata = load_metadata()
        metadata[mode]["last_external_fetch"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_metadata(metadata)
        print(f"No new unique rows found for {mode}.")


def start_scheduler(interval_minutes=30):
    """
    Start APScheduler background scheduler for periodic external updates.
    Runs periodic_external_update for both modes at the given interval.
    """
    global _scheduler

    with _scheduler_lock:
        if _scheduler is not None and _scheduler.running:
            print("Scheduler already running.")
            return _scheduler

        _scheduler = BackgroundScheduler()
        _scheduler.add_job(
            periodic_external_update,
            trigger="interval",
            minutes=interval_minutes,
            args=["nullclass"],
            id="nullclass_update",
            replace_existing=True
        )
        _scheduler.add_job(
            periodic_external_update,
            trigger="interval",
            minutes=interval_minutes,
            args=["medical"],
            id="medical_update",
            replace_existing=True
        )
        _scheduler.start()
        print(f"Scheduler started. Updating every {interval_minutes} minutes from external sources.")
        return _scheduler


def stop_scheduler():
    """Stop the background scheduler."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown()
        _scheduler = None
        print("Scheduler stopped.")


def check_for_updates(mode="nullclass"):
    """Check if dataset has grown since last index build."""
    metadata = load_metadata()
    key = "nullclass" if mode == "nullclass" else "medical"
    csv_path = NULLCLASS_CSV if mode == "nullclass" else MEDQUAD_CSV
    index_path = NULLCLASS_INDEX if mode == "nullclass" else MEDICAL_INDEX

    if not os.path.exists(index_path):
        return True

    current_count = get_row_count(csv_path)
    last_count = metadata[key].get("row_count", 0)
    return current_count > last_count


def auto_update_if_needed(mode="nullclass"):
    """Automatically rebuild index if new data detected locally."""
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
    """Get last updated timestamp and document count."""
    metadata = load_metadata()
    key = "nullclass" if mode == "nullclass" else "medical"
    last = metadata[key].get("last_updated")
    count = metadata[key].get("row_count", 0)
    last_fetch = metadata[key].get("last_external_fetch")
    return last, count, last_fetch
