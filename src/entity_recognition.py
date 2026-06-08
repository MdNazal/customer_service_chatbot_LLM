import spacy

# Load biomedical NLP model from scispaCy
# Model: en_core_sci_sm - trained on biomedical text (PubMed abstracts)
try:
    nlp = spacy.load("en_core_sci_sm")
    SCISPACY_LOADED = True
except OSError:
    SCISPACY_LOADED = False
    print("scispaCy model not found. Install with:")
    print("pip install scispacy")
    print("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz")

# Fallback keyword lists if scispaCy not available
SYMPTOM_KEYWORDS = [
    "pain", "fever", "fatigue", "nausea", "vomiting", "headache", "dizziness",
    "cough", "shortness of breath", "chest pain", "swelling", "rash", "itching",
    "bleeding", "numbness", "weakness", "confusion", "insomnia", "diarrhea",
    "constipation", "weight loss", "weight gain", "loss of appetite", "blurred vision",
    "hearing loss", "sore throat", "runny nose", "chills", "sweating", "tremors",
    "seizures", "fainting", "palpitations", "bruising", "hair loss", "frequent urination"
]

DISEASE_KEYWORDS = [
    "diabetes", "cancer", "hypertension", "asthma", "arthritis", "alzheimer",
    "parkinson", "epilepsy", "depression", "anxiety", "schizophrenia", "autism",
    "hiv", "aids", "tuberculosis", "malaria", "hepatitis", "pneumonia",
    "bronchitis", "gastritis", "colitis", "lupus", "fibromyalgia", "migraine",
    "obesity", "anemia", "leukemia", "lymphoma", "stroke", "heart disease",
    "kidney disease", "liver disease", "thyroid", "psoriasis", "eczema",
    "multiple sclerosis", "celiac", "crohn", "copd", "osteoporosis", "gout"
]

TREATMENT_KEYWORDS = [
    "surgery", "chemotherapy", "radiation", "medication", "therapy", "vaccine",
    "antibiotic", "insulin", "dialysis", "transplant", "immunotherapy",
    "physiotherapy", "psychotherapy", "antidepressant", "antiviral", "painkiller",
    "steroid", "inhaler", "injection", "biopsy", "screening", "rehabilitation",
    "diet", "exercise", "supplement", "vitamin", "aspirin", "ibuprofen",
    "metformin", "statin", "beta blocker", "diuretic"
]


def extract_medical_entities(text):
    """
    Extract medical entities from text using scispaCy biomedical NLP model.
    Falls back to keyword matching if model is not available.
    """
    if SCISPACY_LOADED:
        return _extract_with_scispacy(text)
    else:
        return _extract_with_keywords(text)


def _extract_with_scispacy(text):
    """
    Use scispaCy en_core_sci_sm model to extract biomedical entities.
    The model identifies entities like diseases, chemicals, genes, and anatomy.
    """
    doc = nlp(text)

    entities = {
        "diseases": [],
        "symptoms": [],
        "treatments": []
    }

    symptom_terms = {
        "pain", "fever", "fatigue", "nausea", "vomiting", "headache",
        "dizziness", "cough", "swelling", "rash", "bleeding", "numbness",
        "weakness", "confusion", "insomnia", "diarrhea", "chills", "tremors"
    }

    treatment_terms = {
        "surgery", "therapy", "treatment", "medication", "vaccine", "antibiotic",
        "insulin", "dialysis", "transplant", "chemotherapy", "radiation",
        "immunotherapy", "physiotherapy", "psychotherapy", "screening", "injection"
    }

    seen = set()

    for ent in doc.ents:
        text_lower = ent.text.lower().strip()

        # Skip very short or duplicate entities
        if len(text_lower) < 3 or text_lower in seen:
            continue
        seen.add(text_lower)

        # Classify entity based on context
        if any(term in text_lower for term in treatment_terms):
            entities["treatments"].append(ent.text)
        elif any(term in text_lower for term in symptom_terms):
            entities["symptoms"].append(ent.text)
        else:
            # Default to disease/condition for other biomedical entities
            entities["diseases"].append(ent.text)

    return entities


def _extract_with_keywords(text):
    """Fallback keyword-based extraction."""
    text_lower = text.lower()
    return {
        "symptoms": [s for s in SYMPTOM_KEYWORDS if s in text_lower],
        "diseases": [d for d in DISEASE_KEYWORDS if d in text_lower],
        "treatments": [t for t in TREATMENT_KEYWORDS if t in text_lower]
    }


def has_medical_entities(entities):
    """Check if any medical entities were found."""
    return any([entities["symptoms"], entities["diseases"], entities["treatments"]])


def format_entities_for_display(entities):
    """Format entities into readable strings for Streamlit display."""
    lines = []
    if entities["diseases"]:
        lines.append(f"🦠 **Disease/Condition:** {', '.join(entities['diseases'])}")
    if entities["symptoms"]:
        lines.append(f"🩺 **Symptoms detected:** {', '.join(entities['symptoms'])}")
    if entities["treatments"]:
        lines.append(f"💊 **Treatment-related:** {', '.join(entities['treatments'])}")
    return lines
