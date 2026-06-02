from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Makes detection consistent across runs
DetectorFactory.seed = 0

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ml": "Malayalam",
    "ar": "Arabic",
    "es": "Spanish",
    "hi": "Hindi"
}

# Cultural context instructions for each language
CULTURAL_CONTEXT = {
    "en": "",
    "ml": (
        "Respond in Malayalam script. Use formal Malayalam appropriate for an educational or medical context. "
        "Be respectful and use culturally appropriate greetings if needed."
    ),
    "ar": (
        "Respond in Arabic script. Use Modern Standard Arabic (Fusha). "
        "Be respectful and culturally sensitive. Use appropriate Islamic greetings if contextually relevant."
    ),
    "es": (
        "Respond in Spanish. Use formal Spanish (usted form) appropriate for professional contexts. "
        "Be culturally sensitive and warm in tone."
    ),
    "hi": (
        "Respond in Hindi using Devanagari script. Use formal Hindi appropriate for educational or medical contexts. "
        "Be respectful and culturally appropriate."
    )
}

# Language-specific sentiment prefixes
SENTIMENT_PREFIXES = {
    "en": {
        "positive": "😊 We're glad you're having a great experience! ",
        "negative": "😔 We're sorry to hear you're having trouble. We're here to help. ",
        "neutral": ""
    },
    "ml": {
        "positive": "😊 നിങ്ങള്‍ക്ക് നല്ല അനുഭവം ഉണ്ടെന്ന് കേട്ടതില്‍ സന്തോഷം! ",
        "negative": "😔 നിങ്ങള്‍ക്ക് ബുദ്ധിമുട്ട് ഉണ്ടെന്ന് കേട്ടതില്‍ ഖേദമുണ്ട്. ഞങ്ങള്‍ സഹായിക്കാന്‍ ഇവിടെ ഉണ്ട്. ",
        "neutral": ""
    },
    "ar": {
        "positive": "😊 يسعدنا أنك تمر بتجربة رائعة! ",
        "negative": "😔 نأسف لسماع أنك تواجه صعوبة. نحن هنا للمساعدة. ",
        "neutral": ""
    },
    "es": {
        "positive": "😊 ¡Nos alegra que tenga una gran experiencia! ",
        "negative": "😔 Lamentamos escuchar que tiene dificultades. Estamos aquí para ayudar. ",
        "neutral": ""
    },
    "hi": {
        "positive": "😊 यह जानकर खुशी हुई कि आपका अनुभव अच्छा है! ",
        "negative": "😔 यह सुनकर दुख हुआ कि आपको परेशानी हो रही है। हम यहाँ मदद के लिए हैं। ",
        "neutral": ""
    }
}


def detect_language(text):
    """
    Detect the language of the input text.
    Returns language code (en, ml, ar, es, hi).
    Falls back to English if detection fails or language not supported.
    """
    try:
        detected = detect(text)
        if detected in SUPPORTED_LANGUAGES:
            return detected
        return "en"
    except LangDetectException:
        return "en"


def get_language_name(lang_code):
    """Get full language name from code."""
    return SUPPORTED_LANGUAGES.get(lang_code, "English")


def get_cultural_context(lang_code):
    """Get cultural context instructions for the language."""
    return CULTURAL_CONTEXT.get(lang_code, "")


def get_language_sentiment_prefix(sentiment, lang_code):
    """Get sentiment prefix in the detected language."""
    lang_prefixes = SENTIMENT_PREFIXES.get(lang_code, SENTIMENT_PREFIXES["en"])
    return lang_prefixes.get(sentiment, "")


def build_multilingual_prompt(base_prompt, lang_code):
    """
    Append language instructions to any prompt template.
    """
    cultural_context = get_cultural_context(lang_code)
    lang_name = get_language_name(lang_code)

    if lang_code == "en":
        return base_prompt

    language_instruction = f"\n\nIMPORTANT: You must respond in {lang_name}. {cultural_context}"
    return base_prompt + language_instruction
