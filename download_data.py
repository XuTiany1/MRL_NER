from datasets import load_dataset
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Define language-specific directories
base_dirs = {
    "conll2003_de_noMISC": "./data/german",
    "conll2003_noMISC": "./data/english",
    "turkish-wikiNER": "./data/turkey"
}

# List of additional languages for masakhaner_2.0
masakhaner_languages = ["sna", "hau", "ibo", "yor"]

# Dataset configurations
datasets = [
    {
        "name": "Davlan/conll2003_de_noMISC",
        "dir": base_dirs["conll2003_de_noMISC"]
    },
    {
        "name": "Davlan/conll2003_noMISC",
        "dir": base_dirs["conll2003_noMISC"]
    },
    {
        "name": "turkish-nlp-suite/turkish-wikiNER",
        "dir": base_dirs["turkish-wikiNER"]
    }
]

# Function to save dataset splits
def save_splits(ds, base_dir, split_key="ner_tags", tag_mapping=None):
    os.makedirs(base_dir, exist_ok=True)
    for split in ["train", "validation", "test"]:
        if split not in ds:
            print(f"No {split} split found. Skipping...")
            continue
        file_path = os.path.join(base_dir, f"{split}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            for example in ds[split]:
                tokens = example["tokens"]
                tags = example.get(split_key, example.get("tags"))  # Dynamically handle tag key
                if tags is None:
                    raise KeyError(f"Could not find tags in dataset for split '{split}'.")
                for token, tag in zip(tokens, tags):
                    tag_name = tag_mapping[tag] if tag_mapping else tag
                    f.write(f"{token} {tag_name}\n")
                f.write("\n")  # Add an empty line between sentences
        print(f"Saved {split} split to {file_path}")

# Process pre-defined datasets (German, English, Turkish)
for dataset in datasets:
    print(f"Downloading dataset: {dataset['name']}...")
    try:
        ds = load_dataset(dataset["name"])
        save_splits(ds, dataset["dir"])
    except Exception as e:
        print(f"Failed to download {dataset['name']}: {e}")

# Process masakhaner_2.0 datasets (African languages)
dataset_cache_dir = "./datasets_cache"
os.makedirs(dataset_cache_dir, exist_ok=True)
for lang in masakhaner_languages:
    lang_dir = f"./data/{lang}"  # Each language gets its own folder
    print(f"Downloading dataset for {lang}...")
    try:
        ds = load_dataset("masakhane/masakhaner2", lang, cache_dir=dataset_cache_dir)
        tag_mapping = ds["train"].features["ner_tags"].feature.names
        save_splits(ds, lang_dir, tag_mapping=tag_mapping)
    except Exception as e:
        print(f"Failed to download dataset for {lang}: {e}")

print("All datasets downloaded and saved successfully.")