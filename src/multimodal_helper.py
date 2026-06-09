import os
import requests
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
# Source: https://ai.google.dev/gemini-api/docs
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
genai.configure(api_key=GOOGLE_API_KEY)

# Gemini Vision model for image understanding
GEMINI_MODEL = "gemini-2.0-flash"

# Pollinations.ai for free image generation (no API key required)
# Source: https://pollinations.ai
POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}?width=512&height=512&nologo=true"


def analyze_image(image_file, user_question=None):
    """
    Analyze an uploaded image using Google Gemini Vision.
    Returns detailed analysis of the image content.
    """
    try:
        image = Image.open(image_file)

        model = genai.GenerativeModel(GEMINI_MODEL)

        if user_question:
            prompt = user_question
        else:
            prompt = (
                "Please analyze this image in detail. Describe what you see, "
                "including any text, objects, diagrams, charts, or concepts present. "
                "If this is a technical or scientific image, explain what it represents."
            )

        response = model.generate_content([prompt, image])
        return response.text

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return (
                "⚠️ Gemini API quota exceeded. This is due to regional free-tier restrictions. "
                "Please use a valid Gemini API key from a supported region to enable image analysis. "
                "Get your key at: https://aistudio.google.com"
            )
        return f"Error analyzing image: {error_msg}"


def chat_with_image(image_file, conversation_history):
    """
    Multi-turn conversation about an uploaded image using Gemini Vision.
    """
    try:
        image_file.seek(0)
        image = Image.open(image_file)

        model = genai.GenerativeModel(GEMINI_MODEL)

        # Build conversation context
        history_text = ""
        for msg in conversation_history[:-1]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

        last_question = conversation_history[-1]["content"]

        if history_text:
            prompt = (
                f"I am showing you an image. Here is our conversation so far:\n"
                f"{history_text}\n"
                f"Now answer this follow-up question about the image: {last_question}"
            )
        else:
            prompt = last_question

        response = model.generate_content([prompt, image])
        return response.text

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return (
                "⚠️ Gemini API quota exceeded due to regional restrictions. "
                "Please use a valid API key from a supported region."
            )
        return f"Error: {error_msg}"


def generate_image(text_prompt):
    """
    Generate an image from a text prompt using Pollinations.ai.
    Free, no API key required.
    """
    try:
        clean_prompt = requests.utils.quote(text_prompt)
        image_url = f"https://image.pollinations.ai/prompt/{clean_prompt}?width=512&height=512&nologo=true&seed={hash(text_prompt) % 1000}"
        return image_url
    except Exception as e:
        print(f"Image generation error: {e}")
        return None


def generate_text_explanation(text_prompt):
    """
    Generate a text explanation using Gemini for text-only questions.
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            f"Provide a detailed and clear explanation of: {text_prompt}"
        )
        return response.text

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return (
                "⚠️ Gemini API quota exceeded due to regional restrictions. "
                "Please use a valid API key from a supported region."
            )
        return f"Error: {error_msg}"
