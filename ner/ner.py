import json
import csv
from transformers import AutoTokenizer, AutoModelForTokenClassification
import nltk
import torch
import numpy as np
nltk.download('punkt')
from nltk.tokenize import word_tokenize

custom_labels = ["O", "B-job", "I-job", "B-nationality", "B-person", "I-person", "B-location","B-time", "I-time", "B-event", "I-event", "B-organization", "I-organization", "I-location", "I-nationality", "B-product", "I-product", "B-artwork", "I-artwork"]

def _extract_ner(text: str, model: AutoModelForTokenClassification,
                 tokenizer: AutoTokenizer, start_token: str="▁"):
    print("Input Text:", text)  # Debug print
    tokenized_sentence = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    print("Tokenized Sentence:", tokenized_sentence)  # Debug print
    tokenized_sentences = tokenized_sentence['input_ids'].numpy()

    with torch.no_grad():
        output = model(**tokenized_sentence)

    print("Model Output:", output)  # Debug print

    last_hidden_states = output[0].numpy()
    label_indices = np.argmax(last_hidden_states[0], axis=1)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_sentences[0])
    special_tags = set(tokenizer.special_tokens_map.values())

    grouped_tokens = []
    for token, label_idx in zip(tokens, label_indices):
        if token not in special_tags:
            if not token.startswith(start_token) and len(token.replace(start_token,"").strip()) > 0:
                grouped_tokens[-1]["token"] += token
            else:
                grouped_tokens.append({"token": token, "label": custom_labels[label_idx]})

    print("Grouped Tokens:", grouped_tokens)  # Debug print

    # extract entities
    ents = []
    prev_label = "O"
    for token in grouped_tokens:
        label = token["label"].replace("I-","").replace("B-","")
        if token["label"] != "O":
            
            if label != prev_label:
                ents.append({"token": [token["token"]], "label": label})
            else:
                ents[-1]["token"].append(token["token"])
            
        prev_label = label
    
    print("Extracted Entities:", ents)  # Debug print

    # group tokens
    ents = [{"token": "".join(rec["token"]).replace(start_token," ").strip(), "label": rec["label"]}  for rec in ents ]

    return ents



# Function to extract names using NER model
def extract_arabic_full_names(json_file_path, model, tokenizer, start_token="▁"):
    with open(json_file_path, 'r', encoding="utf-8") as json_file:
        data = json.load(json_file)

    arabic_full_names = []  # Use a list to store names

    for entry in data:
        if "Arabic Text" in entry:
            text = entry.get("Arabic Text", "")
            if text.strip():
                print("Processing text:", text)  # Debug print
                ents = _extract_ner(text, model, tokenizer, start_token)
                print("Entities:", ents)  # Debug print

                current_name = ""
                current_label = None
                for ent in ents:
                    if ent["label"] == "person":
                        if current_label == "person":
                            current_name += " " + ent["token"]
                        else:
                            current_name = ent["token"]
                            current_label = "person"
                    else:
                        if current_label == "person" and current_name.count(" ") >= 2:
                            arabic_full_names.append(current_name.strip())
                            if len(arabic_full_names) >= 2:  # Stop after extracting the first 2 names
                                return arabic_full_names
                        current_name = ""
                        current_label = None

                if current_label == "person" and current_name.count(" ") >= 2:
                    arabic_full_names.append(current_name.strip())
                    if len(arabic_full_names) >= 2:  # Stop after extracting the first 2 names
                        return arabic_full_names

    return arabic_full_names

# Load the NER model and tokenizer
model_cp = "marefa-nlp/marefa-ner"
tokenizer = AutoTokenizer.from_pretrained(model_cp)
model = AutoModelForTokenClassification.from_pretrained(model_cp, num_labels=len(custom_labels))

# Specify the path to your basic_info_frame_2.json file
json_file_path = "basic_info_frame_2.json"

# Extract the unique Arabic full names from basic_info_frame_2.json
arabic_full_names = extract_arabic_full_names(json_file_path, model, tokenizer, start_token="▁")
print("Extracted Arabic Full Names:", arabic_full_names)
