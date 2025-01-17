
# ------------------------------
# IMPORT NECESSARY LIBRARY 
# ------------------------------
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import os
from seqeval.metrics import f1_score, precision_score, recall_score
from random import shuffle
import random
from datasets import Dataset
from transformers import DataCollatorWithPadding
import MRL_NER.evaluate as evaluate
import numpy as np
from tqdm import tqdm
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import DataCollatorForTokenClassification
import MRL_NER.evaluate as evaluate
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer


# ------------------------------
# Setting Up Environment and Parameters
# ------------------------------
random.seed(42)
seqeval = evaluate.load("seqeval")

# This is where the trained model and intermediate results will be saved
output_dir = 'model_saving/'
dataset_folder = "data/"

all_lang = os.listdir(dataset_folder)
print(all_lang)

# Set a limit of 10,000 examples per language for training or processing
each_lang_limit = 10000

# Initialize "map numerical indices (IDs) to NER tag names" and Initialize "map NER tag names to numerical indices "
id2label = dict()
label2id = dict()


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# CHANGE TO DIFFERENT MODEL FOLDERS TO TRAIN DIFFERENT MODELS

# This is where I will start my training
checkpoint =  "models/afro-xlmr-large"   
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Loads the tokenizer associated with model at checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint)       
tokenizer.model_max_length = 514  

# NER label from all languages, Sentences from all languages
labels_from_all_lang = []   
sentences_from_all_lang = []

# NER label, Sentences for validation set
val_labels = []
val_sentences = []
count = 0

print("CHECKPOINT 1: Setting Up Environment and Parameters Complete!") 

# ------------------------------
# Defining Token Labels
# ------------------------------
# B-PER (beginning of a person), I-PER (inside a person), ...
NER_tags = ["O","B-PER","I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
for i in range(len(NER_tags)):
    id2label[i] = NER_tags[i]
    label2id[NER_tags[i]] = i


# ------------------------------
# Model Configuration
# ------------------------------
config = AutoConfig.from_pretrained(checkpoint)
config.max_position_embeddings = 514  # Increase the max sequence length as needed
config.num_labels=len(id2label.keys())
config.id2label = id2label
config.label2id=label2id
# model = AutoModelForTokenClassification.from_pretrained(
#     checkpoint, num_labels=len(id2label.keys()), id2label=id2label, label2id=label2id, config=config 
# )

# Initialize the model
model = AutoModelForTokenClassification.from_pretrained(
    checkpoint, config=config, ignore_mismatched_sizes=True
)
lang_now = "hh"

# ------------------------------
# Read NER Data
# ------------------------------

# Reads and processes NER data from a CoNLL-formatted file.
# Converts tokens and NER tags into a structured format suitable for model training.
def read_conll_file(file):

    count = 0
    
    tem_storing_input = []
    tem_storing_output = []
    
    storing_all_input = []
    storing_all_output = []
    lines_record = []

    for line in file.readlines():
        if count > 0:
            if line.strip() == "":
                if len(tem_storing_input) == len(tem_storing_output):
                    storing_all_input.append(tem_storing_input)
                    storing_all_output.append(tem_storing_output)
                tem_storing_input = []
                tem_storing_output = []
            else:
                try:
                    lines_record.append(line)
                    line = line.split()
                    tem_storing_input.append(line[0])
                    
                    # DATE is excluded from the test, masked to O
                    if "DATE" in line[1].strip():
                        tem_storing_output.append(label2id["O"])
                    else:   
                        tem_storing_output.append(label2id[line[1].strip()])
                except:
                    print(count)
                    print(line)
                    print(lang_now)
                    pass
            
        count += 1
    
    return storing_all_input, storing_all_output


# ------------------------------
# Data Sampling for Training
# ------------------------------

# Merges lists of sentences or labels from multiple languages.
def zip_list_of_lists(lists):
    longest_length = 0
    result_list = []
    for list_item in lists:
        if len(list_item) > longest_length:
            longest_length = len(list_item)
    for j in tqdm(range(longest_length)):
        for i in range(len(lists)): 
            if len(lists[i]) > j:
                result_list.append(lists[i][j])
    return result_list


# Duplicates data for balancing when a certain entity type has fewer samples.
def custom_sampling(source,times):
    result_list = []
    
    integer_part = int(times)
    for i in range(integer_part):
        result_list += source
    remaining_sample_number = int(len(source) *(times - integer_part))
    
    result_list += random.sample(source, remaining_sample_number)

    return result_list


# Ensures balanced sampling across entities (e.g., PER, ORG, LOC).
def balanced_tag_sampling(zipped_sentences_labels):
    sampled_zipped_data = []
    # per org loc
    sentence_label_lists = [[],[],[]]
    id2index = {1:0, 2:0, 3:1,4:1, 5:2,6:2}
    
    no_tag_sentencce_label_list = []
    for i in range(len(zipped_sentences_labels)):
        counts = [0,0,0]
        label =  zipped_sentences_labels[i][1]
        for j in range(len(label)):
            if label[j] > 0:
                counts[id2index[label[j]]] += 1
        max_index = counts.index(max(counts))
        if counts[max_index] == 0:
            no_tag_sentencce_label_list.append(zipped_sentences_labels[i])
        else:    
        # print(max_index)
        # print(label)
        # if(i>100):
        #     break
            sentence_label_lists[max_index].append(zipped_sentences_labels[i])
    
    length_list = [len(sentence_label_list) for sentence_label_list in sentence_label_lists]
    max_length = length_list[length_list.index(max(length_list))]
    print(max_length)
    print(length_list)
    print(len(no_tag_sentencce_label_list))
    if len(zipped_sentences_labels) < each_lang_limit:
        return zipped_sentences_labels
    for h in range(len(sentence_label_lists)):
            times = min(each_lang_limit/len(sentence_label_lists),max_length)/len(sentence_label_lists[h])
            # print(len(custom_sampling(sentence_label_lists[h], times)))
            sampled_zipped_data += custom_sampling(sentence_label_lists[h], times)
    print(len(sampled_zipped_data))
    if len(no_tag_sentencce_label_list) + len(sampled_zipped_data) < each_lang_limit:
        sampled_zipped_data += no_tag_sentencce_label_list
    else:
        sampled_zipped_data += no_tag_sentencce_label_list[:int(each_lang_limit-len(sampled_zipped_data))]
    
    return sampled_zipped_data 


#  Further refines the sampled data to limit the size (each_lang_limit).
def select_data_sample(zipped_sentences_labels):
    sampled_zipped_data = []
    # per org loc
    sentence_label_lists = [[],[],[]]
    id2index = {1:0, 2:0, 3:1,4:1, 5:2,6:2}
    
    no_tag_sentencce_label_list = []
    for i in range(len(zipped_sentences_labels)):
        counts = [0,0,0]
        label =  zipped_sentences_labels[i][1]
        for j in range(len(label)):
            if label[j] > 0:
                counts[id2index[label[j]]] += 1
        max_index = counts.index(max(counts))
        if counts[max_index] == 0:
            no_tag_sentencce_label_list.append(zipped_sentences_labels[i])
        else:    
        # print(max_index)
        # print(label)
        # if(i>100):
        #     break
            sentence_label_lists[max_index].append(zipped_sentences_labels[i])
    
    length_list = [len(sentence_label_list) for sentence_label_list in sentence_label_lists]
    max_length = length_list[length_list.index(max(length_list))]
    min_length = length_list[length_list.index(min(length_list))]
    print(length_list)
    if len(zipped_sentences_labels) < each_lang_limit:
        return zipped_sentences_labels
    
    if min_length > each_lang_limit/3:
        for h in range(len(length_list)):
            times = min(each_lang_limit/len(sentence_label_lists),max_length)/len(sentence_label_lists[h])
            # print(len(custom_sampling(sentence_label_lists[h], times)))
            sampled_zipped_data += custom_sampling(sentence_label_lists[h], times)
    elif sum(length_list) < each_lang_limit:
        sampled_zipped_data += sentence_label_lists[0]
        sampled_zipped_data += sentence_label_lists[1]
        sampled_zipped_data += sentence_label_lists[2]
        sampled_zipped_data += no_tag_sentencce_label_list[:each_lang_limit-len(sampled_zipped_data)]
    else:
        sampled_zipped_data += sentence_label_lists[length_list.index(min(length_list))]
        left_length = each_lang_limit - len(sampled_zipped_data)
        print(len(sampled_zipped_data))
        if length_list[1] > left_length/2:
            for h in range(len(length_list) - 1):
                times = min( left_length/2,max_length)/len(sentence_label_lists[h])
                sampled_zipped_data += custom_sampling(sentence_label_lists[h], times)
                print(len(sampled_zipped_data))
        else:
            sampled_zipped_data += sentence_label_lists[1]
            sampled_zipped_data += sentence_label_lists[0]
            sampled_zipped_data = sampled_zipped_data[:each_lang_limit]
    print(len(sampled_zipped_data))
    return sampled_zipped_data


# Data processing for each language
for lang in all_lang:
    if len(lang.split(".")) == 1:
        lang_now = lang

        # Datafile folders are here!
        train_data_file =  open(dataset_folder +lang + "/train.txt","r",encoding="utf-8")
        dev_data_file =  open(dataset_folder +lang + "/dev.txt","r",encoding="utf-8")
        test_data_file =  open(dataset_folder +lang + "/test.txt","r",encoding="utf-8")
        
        train_input_sentences, train_output_sentences = read_conll_file(train_data_file)
        test_input_sentences, test_output_sentences = read_conll_file(test_data_file)
        dev_input_sentences, dev_output_sentences = read_conll_file(dev_data_file)
        
        # Combine Training and Test data 
        train_input_sentences += test_input_sentences
        train_output_sentences += test_output_sentences
        print(lang)
        zipped = list(zip(train_input_sentences,train_output_sentences))
        

        # If combined data exceed the limit, further refine size
        if len(zipped) > each_lang_limit:
            print(lang)
            zipped = select_data_sample(zipped)
        
        # if lang == "aze":
        #     zipped = select_data_sample(zipped)
        # print(zipped[0])
        # if lang == "eng" or lang == "deu":
        #     pass
        # else:
        # zipped = balanced_tag_sampling(zipped)
            # shuffle(zipped) # randomly shuffle the list and do the sample the first 10000 samples
            # zipped = zipped[:15000]
        
        if lang == "aze":
            dev_input_sentences = dev_input_sentences[:2000]
            dev_output_sentences = dev_output_sentences[:2000]
        
        # Sort sentences by length
        zipped.sort(key=lambda s: len(s[0]),reverse=True)
        
        # Unzip the sorted data into separate lists for sentences and labels
        train_input_sentences, train_output_sentences = zip(*zipped)
        
        # Aggregate training sentences and labels for all languages
        sentences_from_all_lang.append(train_input_sentences)
        val_sentences += dev_input_sentences
        # Aggregate validation sentences and labels for all languages
        labels_from_all_lang.append(train_output_sentences)
        val_labels += dev_output_sentences
        


# Interleaves sentences and labels from all languages using zip_list_of_lists 
sentences = zip_list_of_lists(sentences_from_all_lang)
sentences.reverse()
labels = zip_list_of_lists(labels_from_all_lang)
labels.reverse()

print(sentences[5000:5010])


# ------------------------------
# Preparing Dataset
# ------------------------------

# Create a Dataset object from python library
train_ds = Dataset.from_dict({"ner_tags": labels, "tokens": sentences })
dev_ds = Dataset.from_dict({"ner_tags": val_labels, "tokens": val_sentences})



# ------------------------------
# Tokenize Data
# ------------------------------
# Tokenizes input sentences while aligning NER tags to token IDs.
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    # Loops through each sentence’s NER tags
    for i, label in enumerate(examples[f"ner_tags"]):
        # Maps each token in the tokenized sentence back to its original word index.
        word_ids = tokenized_inputs.word_ids(batch_index=i) 
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    # Returns the tokenized inputs with aligned labels.
    return tokenized_inputs


# ------------------------------
# Dataset Creation
# ------------------------------

# Creates a DataCollatorForTokenClassification object, which is used during model training and evaluation to:
    # Batch Input Data: Combines multiple input examples into a batch.
    # Pad Inputs Dynamically: Ensures all sequences in a batch have the same length by adding padding where necessary.
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)



# ------------------------------
# Metrics Computation
# ------------------------------
import numpy as np

# computes evaluation metrics (precision, recall, F1 score, and accuracy) for a token classification task like Named Entity Recognition (NER).
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [NER_tags[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [NER_tags[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
           }


# Tokenized and aligned training dataset and validataion dataset
tokenized_train_ds = train_ds.map(tokenize_and_align_labels, batched=True)
tokenized_dev_ds = dev_ds.map(tokenize_and_align_labels, batched=True)


batch_size = 2
batch_size = 32


# ------------------------------
# TrainingArguments Setup
# ------------------------------
args = TrainingArguments(
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=15,
    report_to="none",
    save_total_limit=6,
    optim="adamw_torch",
    weight_decay=0.001,
    gradient_accumulation_steps = 1,
    output_dir=output_dir,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    seed=42)
       

# ------------------------------
# Trainer Initialization
# ------------------------------
trainer = Trainer(
    model,
    args,
    # train_dataset=train_dataset,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_dev_ds,
    tokenizer=tokenizer,
    
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ------------------------------
# Train Model
# ------------------------------
trainer.train()