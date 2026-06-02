# Customer Service, Medical, Research & Multi-Modal Chatbot

A multi-mode AI chatbot built on LangChain and Streamlit, featuring sentiment analysis, medical entity recognition, research paper search, dynamic knowledge base management, multi-modal image understanding, and multilingual support.

## Modes
- **Customer Service (Nullclass)** — FAQ-based chatbot for Nullclass e-learning platform queries
- **Medical Q&A (MedQuAD)** — Medical question answering using the MedQuAD dataset from NIH
- **Research Expert (arXiv CS)** — Computer science research paper search and expert Q&A
- **Multi-Modal Chat** — Image understanding and conversational AI using BLIP + Groq

## Features

### Task 1 - Sentiment Analysis
- Detects positive, negative, or neutral sentiment in every user message using VADER
- Adapts response tone accordingly:
  - Positive: warm acknowledgment before the answer
  - Negative: empathetic opener before the answer
  - Neutral: direct answer
- Sentiment prefixes available in all supported languages

### Task 2 - Medical Q&A Chatbot
- Answers medical questions using the MedQuAD dataset (https://github.com/abachaa/MedQuAD)
- Detects medical entities in user queries: diseases, symptoms, and treatments
- Displays detected entities before the answer
- Always reminds users to consult a healthcare professional

### Task 3 - Dynamic Knowledge Base
- Upload new CSV files to expand the knowledge base at any time
- Automatically deduplicates entries before merging
- Rebuilds the FAISS index after every update
- Auto-update detection: checks if new data exists and rebuilds index automatically
- Configurable auto-refresh intervals: 5 minutes, 30 minutes, 1 hour
- Separate knowledge bases for each mode
- Tracks last updated time and document count

### Task 4 - Research Expert Chatbot (arXiv CS)
- Answers complex computer science research questions using arXiv dataset
- Paper search: retrieves top 5 most relevant papers for any topic
- Shows paper title, authors, categories, abstract and direct arXiv link
- Concept visualization: bar charts of top keywords and research areas from search results
- Multi-turn chat with follow-up question support using session history
- Dataset: arXiv CS subset filtered from https://www.kaggle.com/datasets/Cornell-University/arxiv

### Task 5 - Multi-Modal Chat
- Upload images and get detailed analysis using BLIP Large (Salesforce/blip-image-captioning-large)
- Multi-turn conversation about uploaded images with context memory
- Ask follow-up questions about the same image across multiple turns
- Text-only mode when no image is uploaded — Groq answers general questions
- Fully local image understanding — no external API needed after model download

### Task 6 - Multilingual Support
- Automatically detects user language using langdetect
- Supports 5 languages: English, Malayalam, Arabic, Spanish, Hindi
- Responds in the same language as the user's question
- Language-specific sentiment prefixes for culturally appropriate responses
- Works seamlessly across all modes — Customer Service, Medical, Research, and Multi-Modal
- Displays detected language label in the UI

## Tech Stack
- **LLM**: Groq (llama-3.1-8b-instant)
- **Embeddings**: HuggingFace all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **Sentiment Analysis**: VADER (vaderSentiment)
- **Medical Entity Recognition**: Keyword-based extraction
- **Image Understanding**: BLIP Large (Salesforce/blip-image-captioning-large)
- **Language Detection**: langdetect
- **Framework**: LangChain + Streamlit
- **Datasets**: Nullclass FAQ CSV, MedQuAD, arXiv CS subset

## Supported Languages
| Language | Code | Script |
|----------|------|--------|
| English | en | Latin |
| Malayalam | ml | Malayalam |
| Arabic | ar | Arabic |
| Spanish | es | Latin |
| Hindi | hi | Devanagari |

## Project Structure
```
customer_service_chatbot_LLM/
├── dataset/
│   ├── dataset.csv               <- Nullclass FAQ dataset (download via Google Drive)
│   ├── medquad.csv               <- Large file — download via Google Drive (see below)
│   ├── arxiv_cs.csv              <- Large file — download via Google Drive (see below)
│   ├── update_metadata.json      <- Tracks last update times and row counts
│   └── MedQuAD/                  <- Large folder — download via Google Drive (see below)
│       ├── 1_CancerGov_QA/
│       ├── 2_GARD_QA/
│       ├── 3_GHR_QA/
│       ├── 4_MPlus_Health_Topics_QA/
│       ├── 5_NIDDK_QA/
│       ├── 6_NINDS_QA/
│       ├── 7_SeniorHealth_QA/
│       ├── 8_NHLBI_QA_XML/
│       └── 9_CDC_QA/
├── src/
│   ├── main.py                   <- Streamlit UI with all modes
│   ├── langchain_helper.py       <- Nullclass RAG chain
│   ├── medquad_helper.py         <- MedQuAD XML parser + RAG chain
│   ├── arxiv_helper.py           <- arXiv paper search + RAG chain
│   ├── multimodal_helper.py      <- BLIP image captioning + Groq conversation
│   ├── sentiment_helper.py       <- VADER sentiment detection
│   ├── entity_recognition.py     <- Medical entity extraction
│   ├── knowledge_updater.py      <- Dynamic knowledge base management
│   ├── visualizer.py             <- Keyword and category visualization
│   └── language_helper.py        <- Language detection and multilingual support
├── faiss_index/                  <- Saved Nullclass FAISS index (regenerated locally)
├── faiss_index_medical/          <- Saved Medical FAISS index (regenerated locally)
├── faiss_index_arxiv/            <- Saved arXiv FAISS index (regenerated locally)
├── .env                          <- API keys — NOT committed (see .env.example)
├── .env.example                  <- Template for required API keys
├── requirements.txt
└── README.md
```

## 📁 Datasets (Google Drive)

Due to file size limits, large dataset files are hosted on Google Drive.

👉 **[Download Datasets from Google Drive](https://drive.google.com/drive/folders/1fLdxb_nJn3nTyA1tqZm0u9UoOKl9YkZ9?usp=drive_link)**

After downloading, place the files as follows:
- `medquad.csv` → `dataset/medquad.csv`
- `dataset.csv` → `dataset/dataset.csv`
- `arxiv_cs.csv` → `dataset/arxiv_cs.csv`
- `MedQuAD/` folder → `dataset/MedQuAD/`

> The FAISS index folders (`faiss_index/`, `faiss_index_medical/`, `faiss_index_arxiv/`) are also excluded from the repo. They will be **auto-generated** when you click "Create Knowledgebase" in the app for each mode.

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/customer_service_chatbot_LLM.git
cd customer_service_chatbot_LLM
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Copy `.env.example` to `.env` and fill in your API keys:
```bash
copy .env.example .env
```
Then edit `.env`:
```
GROQ_API_KEY="your_groq_api_key_here"
HF_API_TOKEN="your_huggingface_token_here"
```
- Groq API key: https://console.groq.com
- HuggingFace token: https://huggingface.co/settings/tokens

### 5. Download datasets
Download the large dataset files from the [Google Drive link above](#-datasets-google-drive) and place them in the `dataset/` folder.

### 6. Run the app
```bash
streamlit run src/main.py
```

### 7. First time setup per mode
- **Customer Service** — Open Knowledge Base Management → Create Knowledgebase
- **Medical Q&A** — Place MedQuAD folders in `dataset/MedQuAD/` → Create Knowledgebase
- **Research Expert** — Place `arxiv_cs.csv` in `dataset/` → Create Knowledgebase
- **Multi-Modal** — No setup needed, BLIP model downloads automatically on first use

## Usage

### Customer Service & Medical Modes
- Type your question in any supported language
- Language is detected automatically and shown in the UI
- Sentiment is detected and response tone is adjusted in the detected language
- Medical mode shows detected entities above the answer

### Research Expert Mode
- Use the search bar to find relevant papers by topic
- View paper details, abstracts and arXiv links
- See concept visualization charts for search results
- Use the chat to ask research questions in any supported language

### Multi-Modal Mode
- Upload an image to get automatic analysis
- Ask follow-up questions about the uploaded image
- Ask text-only questions without uploading an image

### Updating the Knowledge Base
- Open Knowledge Base Management panel
- Upload a CSV with columns: prompt, response
- Click Add to Knowledge Base
- Use auto-update to detect and apply changes automatically

## Multilingual Examples
| Language | Sample Question |
|----------|----------------|
| Malayalam | ഡയബറ്റിസിന്റെ ലക്ഷണങ്ങൾ എന്തൊക്കെയാണ്? |
| Arabic | ما هي أعراض مرض السكري؟ |
| Spanish | ¿Cuáles son los síntomas del asma? |
| Hindi | मशीन लर्निंग क्या है? |

## Notes
- MedQuAD XML files parsed once and cached as `dataset/medquad.csv` on first run
- Folders 10, 11, 12 of MedQuAD excluded due to MedlinePlus copyright restrictions
- arXiv dataset filtered to CS papers only using `filter_arxiv.py`
- BLIP model (~900MB) downloads automatically on first use of Multi-Modal mode
- Medical information is for educational purposes only
- Always consult a qualified healthcare professional for personal medical advice
