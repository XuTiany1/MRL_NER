from transformers import AutoModel, AutoTokenizer
import os

# List of Hugging Face models to download
models = [
    "Davlan/afro-xlmr-large-76L",
    "Davlan/afro-xlmr-large",
    "FacebookAI/xlm-roberta-large",
    "urchade/gliner_multi-v2.1"
]

# Base directory for saving models
base_dir = "./models"

# Function to download and save models
def download_model(model_name):
    print(f"Downloading model: {model_name}...")
    model_dir = os.path.join(base_dir, model_name.split("/")[-1])  # Use the model name for the folder
    os.makedirs(model_dir, exist_ok=True)

    # Download model and tokenizer
    try:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Save model and tokenizer locally
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"Model saved to: {model_dir}")
    except Exception as e:
        print(f"Failed to download model {model_name}: {e}")

# Create base directory if it doesn't exist
os.makedirs(base_dir, exist_ok=True)

# Download each model
for model_name in models:
    download_model(model_name)

print("All models downloaded and saved successfully.")