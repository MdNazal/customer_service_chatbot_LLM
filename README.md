# Customer Service, Medical, Research & Multi-Modal Chatbot

A multi-mode AI chatbot built on LangChain and Streamlit, featuring sentiment analysis, medical entity recognition, research paper search, dynamic knowledge base management, multi-modal image understanding with Google Gemini AI, and multilingual support.

## Modes
- **Customer Service (Nullclass)** — FAQ-based chatbot for Nullclass e-learning platform queries
- **Medical Q&A (MedQuAD)** — Medical question answering using the MedQuAD dataset from NIH
- **Research Expert (arXiv CS)** — Computer science research paper search, summarization, and expert Q&A
- **Multi-Modal Chat (Gemini AI)** — Image understanding via Google Gemini Vision and image generation via Pollinations.ai

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
- Detects medical entities using scispaCy biomedical NLP model (en_core_sci_sm)
- Identifies diseases, symptoms, and treatments in context
- Displays detected entities before the answer
- Always reminds users to consult a healthcare professional

### Task 3 - Dynamic Knowledge Base
- APScheduler background scheduler fetches new data from external GitHub source every 30 minutes
- Upload new CSV files to expand the knowledge base at any time
- Automatically deduplicates entries before merging
- Rebuilds the FAISS index after every update
- Configurable auto-refresh intervals: 5 minutes, 30 minutes, 1 hour
- Separate knowledge bases for each mode
- Tracks last updated time, document count, and last external fetch timestamp

### Task 4 - Research Expert Chatbot (arXiv CS)
- Answers complex computer science research questions using arXiv dataset
- Paper search: retrieves top 5 most relevant papers for any topic
- Shows paper title, authors, categories, abstract and direct arXiv link
- Concept visualization: bar charts of top keywords and research areas from search results
- Paper summarization using LangChain map-reduce summarization chain
- Information extraction: extracts problem, methodology, results, and contributions from papers
- Multi-turn chat with follow-up question support using session history
- Dataset: arXiv CS subset filtered from https://www.kaggle.com/datasets/Cornell-University/arxiv

### Task 5 - Multi-Modal Chat (Google Gemini AI)
- Image understanding using Google Gemini Vision (gemini-2.0-flash)
- Image generation from text prompts using Pollinations.ai
- Multi-turn conversation about uploaded images with context memory
- Text-only mode when no image is uploaded — Gemini answers general questions
- Note: Gemini Vision requires a valid API key from a supported region. Regional free-tier restrictions may apply.

### Task 6 - Multilingual Support
- Automatically detects user language using langdetect
- Supports 5 languages: English, Malayalam, Arabic, Spanish, Hindi
- Responds in the same language as the user's question
- Language-specific sentiment prefixes for culturally appropriate responses
- Works seamlessly across all modes
- Displays detected language label in the UI

## Tech Stack
- **LLM**: Groq (llama-3.1-8b-instant)
- **Image Understanding**: Google Gemini Vision (gemini-2.0-flash)
- **Image Generation**: Pollinations.ai (free, no API key required)
- **Embeddings**: HuggingFace all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **Sentiment Analysis**: VADER (vaderSentiment)
- **Medical Entity Recognition**: scispaCy (en_core_sci_sm biomedical NLP model)
- **Summarization**: LangChain map-reduce summarization chain
- **Scheduling**: APScheduler (background periodic updates)
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
│   ├── dataset.csv               <- Nullclass FAQ dataset
│   ├── medquad.csv               <- Parsed MedQuAD QA pairs
│   ├── arxiv_cs.csv              <- Filtered arXiv CS papers
│   ├── update_metadata.json      <- Tracks last update times and row counts
│   └── MedQuAD/                  <- MedQuAD XML folders (download via Google Drive)
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
│   ├── arxiv_helper.py           <- arXiv paper search, summarization, info extraction
│   ├── multimodal_helper.py      <- Gemini Vision + Pollinations image generation
│   ├── sentiment_helper.py       <- VADER sentiment detection
│   ├── entity_recognition.py     <- scispaCy biomedical NER
│   ├── knowledge_updater.py      <- APScheduler + dynamic knowledge base management
│   ├── visualizer.py             <- Keyword and category visualization
│   └── language_helper.py        <- Language detection and multilingual support
├── faiss_index/                  <- Saved Nullclass FAISS index (auto-generated)
├── faiss_index_medical/          <- Saved Medical FAISS index (auto-generated)
├── faiss_index_arxiv/            <- Saved arXiv FAISS index (auto-generated)
├── .env                          <- API keys — NOT committed (see .env.example)
├── .env.example                  <- Template for required API keys
├── requirements.txt
└── README.md
```

## 📁 Datasets (Google Drive)

Due to file size limits, the MedQuAD XML folders are hosted on Google Drive.

👉 **[Download Datasets from Google Drive](https://drive.google.com/drive/folders/1fLdxb_nJn3nTyA1tqZm0u9UoOKl9YkZ9?usp=drive_link)**

After downloading, place the MedQuAD folder as follows:
- `MedQuAD/` folder → `dataset/MedQuAD/`

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/MdNazal/customer_service_chatbot_LLM.git
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
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

### 4. Set up environment variables
Copy `.env.example` to `.env` and fill in your API keys:
```bash
copy .env.example .env
```
Then edit `.env`:
```
GROQ_API_KEY="your_groq_api_key_here"
GOOGLE_API_KEY="your_gemini_api_key_here"
HF_API_TOKEN="your_huggingface_token_here"
```
- Groq API key: https://console.groq.com
- Gemini API key: https://aistudio.google.com
- HuggingFace token: https://huggingface.co/settings/tokens

### 5. Run the app
```bash
streamlit run src/main.py
```

### 6. First time setup per mode
- **Customer Service** — Open Knowledge Base Management → Create Knowledgebase
- **Medical Q&A** — Place MedQuAD folders in `dataset/MedQuAD/` → Create Knowledgebase
- **Research Expert** — `arxiv_cs.csv` is included in repo → Create Knowledgebase
- **Multi-Modal** — No setup needed, requires valid Gemini API key in `.env`

## Usage

### Customer Service & Medical Modes
- Type your question in any supported language
- Language is detected automatically and shown in the UI
- Sentiment is detected and response tone is adjusted in the detected language
- Medical mode shows detected entities (diseases, symptoms, treatments) above the answer

### Research Expert Mode
- Use the search bar to find relevant papers by topic
- View paper details, abstracts and arXiv links
- See concept visualization charts for search results
- Use the summarize section to get a structured summary of any topic
- Use the information extraction section to get problem, methodology, results and contributions
- Use the chat to ask research questions in any supported language

### Multi-Modal Mode
- Type a description in the Generate Image section to create an image using AI
- Upload an image to get automatic analysis using Gemini Vision
- Ask follow-up questions about the uploaded image
- Ask text-only questions without uploading an image

### Updating the Knowledge Base
- Open Knowledge Base Management panel
- APScheduler automatically fetches updates from external source every 30 minutes
- Upload a CSV with columns: prompt, response to manually add data
- Click Add to Knowledge Base to merge and rebuild
- Use Check & Auto Update Now to trigger an immediate check

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
- Gemini Vision requires a valid API key — regional free-tier restrictions may prevent usage in some areas
- Medical information is for educational purposes only
- Always consult a qualified healthcare professional for personal medical advice
