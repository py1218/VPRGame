import os
import runpod
from transformers import pipeline

# Fetch Hugging Face token from environment
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

# Initialize the zero-shot classification pipeline with Bart-Large-MNLI
print("🔄 Loading facebook/bart-large-mnli zero-shot classifier...")
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    use_auth_token=HF_TOKEN
)

def handler(event):
    """
    Serverless handler that performs zero-shot classification on input text.

    Expects `event['input']` to be a dict with:
      - "prompt": the text to classify
      - "labels" (optional): a list of candidate labels

    Returns the full Hugging Face pipeline output.
    """
    input_data = event.get("input", {})
    text = input_data.get("prompt", "")
    labels = input_data.get("labels", ["negative", "positive"])

    print(f"🔍 Classifying: \"{text}\" with labels {labels}")
    result = classifier(text, candidate_labels=labels)
    return result

# Start the Serverless function
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
