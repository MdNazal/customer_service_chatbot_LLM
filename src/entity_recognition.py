SYMPTOM_KEYWORDS = [
    "pain", "fever", "fatigue", "nausea", "vomiting", "headache", "dizziness",
    "cough", "shortness of breath", "chest pain", "swelling", "rash", "itching",
    "bleeding", "numbness", "weakness", "confusion", "anxiety", "depression",
    "insomnia", "diarrhea", "constipation", "weight loss", "weight gain",
    "loss of appetite", "blurred vision", "hearing loss", "sore throat",
    "runny nose", "chills", "sweating", "tremors", "seizures", "fainting",
    "palpitations", "bruising", "hair loss", "dry mouth", "frequent urination"
]

DISEASE_KEYWORDS = [
    "diabetes", "cancer", "hypertension", "asthma", "arthritis", "alzheimer",
    "parkinson", "epilepsy", "depression", "anxiety", "schizophrenia", "autism",
    "hiv", "aids", "tuberculosis", "malaria", "hepatitis", "pneumonia",
    "bronchitis", "gastritis", "colitis", "lupus", "fibromyalgia", "migraine",
    "obesity", "anemia", "leukemia", "lymphoma", "stroke", "heart disease",
    "kidney disease", "liver disease", "thyroid", "psoriasis", "eczema",
    "multiple sclerosis", "celiac", "crohn", "copd", "osteoporosis", "gout",
    "sickle cell", "meningitis", "sepsis", "covid", "influenza", "cholesterol"
]

TREATMENT_KEYWORDS = [
    "surgery", "chemotherapy", "radiation", "medication", "therapy", "vaccine",
    "antibiotic", "insulin", "dialysis", "transplant", "immunotherapy",
    "physiotherapy", "psychotherapy", "antidepressant", "antiviral", "painkiller",
    "steroid", "inhaler", "injection", "biopsy", "screening", "rehabilitation",
    "diet", "exercise", "lifestyle", "supplement", "vitamin", "mineral",
    "aspirin", "ibuprofen", "metformin", "statin", "beta blocker", "diuretic"
]


def extract_medical_entities(text):
    text_lower = text.lower()

    found_symptoms = [s for s in SYMPTOM_KEYWORDS if s in text_lower]
    found_diseases = [d for d in DISEASE_KEYWORDS if d in text_lower]
    found_treatments = [t for t in TREATMENT_KEYWORDS if t in text_lower]

    return {
        "symptoms": found_symptoms,
        "diseases": found_diseases,
        "treatments": found_treatments
    }


def has_medical_entities(entities):
    return any([entities["symptoms"], entities["diseases"], entities["treatments"]])


def format_entities_for_display(entities):
    lines = []
    if entities["diseases"]:
        lines.append(f"🦠 **Disease/Condition:** {', '.join(entities['diseases'])}")
    if entities["symptoms"]:
        lines.append(f"🩺 **Symptoms detected:** {', '.join(entities['symptoms'])}")
    if entities["treatments"]:
        lines.append(f"💊 **Treatment-related:** {', '.join(entities['treatments'])}")
    return lines
