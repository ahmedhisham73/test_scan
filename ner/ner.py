import json
import csv
from transformers import AutoTokenizer, AutoModelForTokenClassification
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Function to extract names using NER model
def extract_arabic_full_names(json_file_path, model, tokenizer, start_token="▁"):
    with open(json_file_path, 'r', encoding="utf-8") as json_file:
        data = json.load(json_file)

    arabic_full_names = {"first accountant": None, "second accountant": None}

    for entry in data:
        if "Arabic Text" in entry:
            text = entry.get("Arabic Text", "")
            if text.strip():
                ents = _extract_ner(text, model, tokenizer, start_token)

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
                            if arabic_full_names["first accountant"] is None:
                                arabic_full_names["first accountant"] = current_name.strip()
                            elif arabic_full_names["second accountant"] is None:
                                arabic_full_names["second accountant"] = current_name.strip()
                            if arabic_full_names["first accountant"] is not None and arabic_full_names["second accountant"] is not None:
                                return arabic_full_names
                        current_name = ""
                        current_label = None

                if current_label == "person" and current_name.count(" ") >= 2:
                    if arabic_full_names["first accountant"] is None:
                        arabic_full_names["first accountant"] = current_name.strip()
                    elif arabic_full_names["second accountant"] is None:
                        arabic_full_names["second accountant"] = current_name.strip()
                    if arabic_full_names["first accountant"] is not None and arabic_full_names["second accountant"] is not None:
                        return arabic_full_names

    return arabic_full_names

# Load the NER model and tokenizer
model_cp = "marefa-nlp/marefa-ner"
tokenizer = AutoTokenizer.from_pretrained(model_cp)
model = AutoModelForTokenClassification.from_pretrained(model_cp, num_labels=len(custom_labels))

# Specify the path to your basic_info_frame_2.json file
json_file_path = "basic_info_frame_2.json"

# Extract the first 2 unique Arabic full names from basic_info_frame_2.json
arabic_full_names = extract_arabic_full_names(json_file_path, model, tokenizer, start_token="▁")

# Load existing data from extracted_info.csv
existing_data = {}
with open("extracted_info.csv", 'r', encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        existing_data[row[0]] = row[1]

# Update the data with extracted names
existing_data["first accountant"] = arabic_full_names["first accountant"]
existing_data["second accountant"] = arabic_full_names["second accountant"]

# Write the updated data to the CSV file
with open("extracted_info.csv", mode="w", encoding="utf-8", newline='') as csv_output_file:
    fieldnames = ["Pattern Name", "Extracted Data"]
    writer = csv.DictWriter(csv_output_file, fieldnames=fieldnames)
    writer.writeheader()

    for pattern_name, data in existing_data.items():
        if data:
            writer.writerow({"Pattern Name": pattern_name, "Extracted Data": data})

print("Updated data has been saved to extracted_info.csv.")

