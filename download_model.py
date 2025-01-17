from transformers import AutoTokenizer, AutoModelForTokenClassification

# Download and save the model
checkpoint = "Davlan/afro-xlmr-large"  # Hugging Face model name
save_dir = "models/afro-xlmr-large/"  # Local folder to store the model

# Ensure the model folder exists
if not os.path.exists(save_dir):
    print(f"Downloading model {checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.save_pretrained(save_dir)

    model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    model.save_pretrained(save_dir)
    print(f"Model and tokenizer saved to {save_dir}")
else:
    print(f"Model already exists in {save_dir}")