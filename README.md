# MRL NER

This task is for NER prediction done by the McGill Team.

---

## **Setting Up the Environment**

1. Build the virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Downloading Models and Data**

### **Models**

This project utilizes an ensemble of the following pre-trained models from Hugging Face:

- **afro-xlmr-large**
- **afro-xlmr-large-76L**
- **xlm-roberta-large**

To download the models, run the following command:
```bash
python3 download_model.py
```

### **Datasets**

The following datasets are used for training and evaluation:

| **Dataset**            | **Language/Region** | **Train** | **Validation** | **Test** |
|-------------------------|---------------------|-----------|----------------|----------|
| **MasakhaNER 2.0**      | Yoruba              | 6.8k      | 1k             | 1.9k     |
|                         | Igbo                | 7.6k      | 1k             | 2k       |
|                         | Hausa               | 5.7k      | 0.8k           | 1.6k     |
|                         | chiShona            | 6.2k      | 0.8k           | 1.7k     |
|                         | Naijia              | 5.6k      | 0.8k           | 1.3k     |
| **CoNLL03**             | German              | 12k       | 2.8k           | 3k       |
|                         | English             | 14k       | 3k             | 3k       |
| **Turkish WikiNER**     | Turkey              | 18k       | 1k             | 1k       |
| **UZNER**               | Uzbek               | 7k        | 2k             | 2k       |

To download the datasets, run:
```bash
python3 download_data.py
```

---

## **Training and Evaluation**

1. To train the models, use the following command:
   ```bash
   python3 train_model.py
   ```

2. For evaluation on test datasets, run:
   ```bash
   python3 evaluate_model.py
   ```

---

## **Acknowledgments**

This project uses the following resources:

- [MasakhaNER 2.0 Dataset](https://arxiv.org/abs/2106.13807)
- [CoNLL03 Dataset](https://www.clips.uantwerpen.be/conll2003/ner/)
- [Turkish WikiNER Dataset](https://github.com/stefan-it/turkish-nlp-suite)
- [UZNER Dataset](https://github.com/layik/uzner)
