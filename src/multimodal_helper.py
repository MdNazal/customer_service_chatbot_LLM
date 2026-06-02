import os
from PIL import Image
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Groq for conversation
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2
)

# Load BLIP model once globally
print("Loading BLIP image captioning model...")
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model.eval()
    BLIP_LOADED = True
    print("BLIP model loaded successfully.")
except Exception as e:
    BLIP_LOADED = False
    print(f"Could not load BLIP model: {e}")


def get_image_caption(image_file):
    """Generate a caption for the image using local BLIP model."""
    if not BLIP_LOADED:
        return "Image captioning model not available."

    try:
        image_file.seek(0)
        img = Image.open(image_file).convert("RGB")

        inputs = processor(img, return_tensors="pt")
        with __import__('torch').no_grad():
            output = blip_model.generate(**inputs, max_new_tokens=100)

        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption

    except Exception as e:
        return f"Error generating caption: {str(e)}"


def analyze_image(image_file, user_question=None):
    """
    Analyze image using BLIP caption + Groq for detailed explanation.
    """
    caption = get_image_caption(image_file)

    if user_question:
        prompt = f"""An image has been analyzed and described as: "{caption}"

Based on this description, answer the following question: {user_question}

Provide a detailed and helpful response."""
    else:
        prompt = f"""An image has been analyzed and described as: "{caption}"

Based on this description, provide a detailed analysis of the image including:
- What is shown in the image
- Key elements and their significance  
- Any technical or conceptual aspects if relevant
- Overall context or purpose of the image"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return f"**Caption:** {caption}\n\n**Analysis:**\n{response.content}"


def chat_with_image(image_file, conversation_history):
    """
    Multi-turn conversation about an image using caption + Groq.
    """
    caption = get_image_caption(image_file)

    history_text = ""
    for msg in conversation_history[:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    last_question = conversation_history[-1]["content"]

    prompt = f"""You are analyzing an image described as: "{caption}"

Previous conversation:
{history_text}

Current question: {last_question}

Answer based on the image description and conversation context. Be detailed and helpful."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def generate_text_explanation(text_prompt):
    """Generate explanation for a concept using Groq (no image)."""
    response = llm.invoke([HumanMessage(content=f"Provide a detailed and clear explanation of: {text_prompt}")])
    return response.content
