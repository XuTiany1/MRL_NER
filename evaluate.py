

###########################
# IMPORT LIBRARIES
###########################

from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.metrics import classification_report
from transformers import AutoModelForTokenClassification
import numpy as np
import os
from transformers import AutoTokenizer


###########################
# Spcify the location of the pre-trained models
###########################

# checkpoint = '../scratch/mrl-2024/model-saving/saving_NER/afro-xlmr-large-76L/test1/checkpoint-34020'
# checkpoint = '../scratch/mrl-2024/model-saving/saving_NER/xlmr-large/test1/checkpoint-36450'
# checkpoint = '../scratch/mrl-2024/model-saving/saving_NER/afro-xlmr-large-76L/itegrateVal/checkpoint-37005'
# checkpoint = '../scratch/mrl-2024/model-saving/saving_NER/xlm-roberta-large/itegrateVal/checkpoint-37005'
checkpoint = '../scratch/mrl-2024/model-saving/saving_NER/encoder-only-token-pred-10000limit-select-plustest-itegrateVal/checkpoint-37005'
model_name = "afro-xlmr-large-76L"

###########################
# Load Models and Tokenize
###########################
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForTokenClassification.from_pretrained(checkpoint).to(device)

id2label = dict()
label2id = dict()

###########################
# Define NER Tags
###########################
NER_tags = ["O","B-PER","I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
for i in range(len(NER_tags)):
    id2label[i] = NER_tags[i]
    label2id[NER_tags[i]] = i


###########################
# Read and Process Data
###########################
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
                    line = line.split(" -X- _ ")
                    tem_storing_input.append(line[0])
                    if "DATE" in line[1].strip():
                        tem_storing_output.append(label2id["O"])
                    else:   
                        tem_storing_output.append(label2id[line[1].strip()])
                except:
                    print(count)
                    print(line)
            
        count += 1
    
    return storing_all_input, storing_all_output


###########################
# Tokenize and align labels
###########################
# Tokenizes sentences while aligning word-level labels to token-level labels.
def tokenize_and_align_labels(tokens, tags):
    tokenized_inputs = tokenizer([tokens], truncation=True, is_split_into_words=True)
    # print(tokens)
    # print(tags)
    labels = []
    
    word_ids = tokenized_inputs.word_ids(batch_index=0)  # Map tokens to their respective word.
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:  # Set the special tokens to -100.
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:  # Only label the first token of a given word.
            label_ids.append(tags[word_idx])
        else:
            label_ids.append(-100)
        previous_word_idx = word_idx
    
    return np.array(label_ids)





# text = "The Golden State Warriors are an American professional basketball team based in San Francisco."
# inputs = tokenizer(text, return_tensors="pt")
# with torch.no_grad():
#     logits = model(**inputs).logits
# predictions = torch.argmax(logits, dim=2)
# predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]




###########################
# Test the NER model
###########################
# Loops through test samples, tokenizes them, and makes predictions using the model.
def test_Ner(inputs, labels):

    output_tags = []
    golden_labels = []
    for i in tqdm(range(len(inputs))):
        #text = "The Golden State Warriors are an American professional basketball team based in San Francisco."
        input_text = inputs[i]
        # print(" ".join(text))
        tokenized_input = tokenizer(" ".join(input_text), return_tensors="pt").to(device)
        algined_labels = tokenize_and_align_labels(input_text, labels[i])

        # print(len(algined_labels))
        # print(tokenized_input)
        
        with torch.no_grad():
            logits = model(**tokenized_input).logits
        predictions = torch.argmax(logits, dim=2)
        predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
        # print(len(predicted_token_class))
        # break
        out_tag = []
        for j in range(len(predicted_token_class)):
            if algined_labels[j] != -100:
                out_tag.append(predicted_token_class[j])
        output_tags.append(out_tag)
        # golden_labels.append(algined_labels)
        
        golden_labels.append([id2label[id] for id in labels[i]])
        
    for h in range(len(golden_labels)):
        if len(golden_labels[h]) != len(output_tags[h]):
            # print(golden_labels[h])
            # print(output_tags[h])
            # print(inputs[h])
            # print(len(golden_labels[h]))
            # print(len(output_tags[h]))
            # print(predicted_token_class)
            # print(algined_labels)
            # print(tokenizer([" ".join(input_text)], truncation=True))
            output_tags[h].append("O")
    results = {
        "precision": precision_score(golden_labels, output_tags),
        "recall": recall_score(golden_labels, output_tags),
        "f1": f1_score(golden_labels, output_tags),
    }    
    
    print(classification_report(golden_labels, output_tags))
    return results


# Similar to test_Ner but writes the predicted tags and input tokens to a .conll file for evaluation.
def test_Ner_and_write(inputs, labels, lang_name):
    lang_name = "./NER-output/%s/"%model_name + lang_name.split(".")[0] + "_output.conll"
    with open(lang_name, "w", encoding="utf-8") as f_out:
        output_tags = []
        golden_labels = []
        for i in tqdm(range(len(inputs))):
            #text = "The Golden State Warriors are an American professional basketball team based in San Francisco."
            input_text = inputs[i]
            # print(" ".join(text))
            tokenized_input = tokenizer(" ".join(input_text), return_tensors="pt").to(device)
            algined_labels = tokenize_and_align_labels(input_text, labels[i])

            # print(len(algined_labels))
            # print(tokenized_input)
            
            with torch.no_grad():
                logits = model(**tokenized_input).logits
            predictions = torch.argmax(logits, dim=2)
            predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
            # print(len(predicted_token_class))
            # break
            out_tag = []
            for j in range(len(predicted_token_class)):
                if algined_labels[j] != -100:
                    out_tag.append(predicted_token_class[j])
            output_tags.append(out_tag)
            # golden_labels.append(algined_labels)
            
            golden_labels.append([id2label[id] for id in labels[i]])
            
        for h in range(len(golden_labels)):
            if len(golden_labels[h]) != len(output_tags[h]):
                output_tags[h].append("O")
        
        
        for x in range(len(output_tags)):
            input_tokens = inputs[x]
            output_tag = output_tags[x]
            for y in range(len(input_tokens)):
                f_out.write(input_tokens[y] + "   " + output_tag[y]+"\n")
            f_out.write("\n")
        # print(classification_report(golden_labels, output_tags))
        print( {
        "precision": precision_score(golden_labels, output_tags),
        "recall": recall_score(golden_labels, output_tags),
        "f1": f1_score(golden_labels, output_tags),
    }    )


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# CHANGE THIS 
# A boolean flag that determines whether to generate prediction files (True) or evaluate the model’s performance (False).
generate_file = False
all_lang = ["bam"]

dataset_folder = "../scratch/mrl-2024/data/NER_VAL_organizer/"
all_lang = os.listdir(dataset_folder)

for lang in all_lang:
    if lang != "NER_IG_Val.conll":
        test_data_file =  open(dataset_folder+lang,"r",encoding="utf-8")
        test_input_sentences, test_output_sentences = read_conll_file(test_data_file)
        print(lang)
        if generate_file:
            test_Ner_and_write(test_input_sentences, test_output_sentences, lang)
        else:
            print(test_Ner(test_input_sentences, test_output_sentences))


# The above after !!!! does
# Loops through multiple test datasets in the CoNLL format for various languages or domains.
# Either evaluates the NER model on these datasets or generates output predictions for further use, depending on the generate_file flag.
# Automates the evaluation or prediction generation across all test files in the specified folder.