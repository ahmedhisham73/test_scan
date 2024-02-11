import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Function to extract names using MAREFA NER model
def extract_arabic_names(json_data, model, tokenizer):
    arabic_names = set()

    for entry in json_data:
        if "Arabic Text" in entry:
            text = entry["Arabic Text"]
            tokenized_text = tokenizer.tokenize(text)
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)
            predicted_labels = [model.config.id2label[label_id] for label_id in predictions[0]]
            
            current_name = ""
            for token, label in zip(tokenized_text, predicted_labels):
                if label == "B-person":
                    current_name = token
                elif label == "I-person":
                    current_name += " " + token
                elif label != "O" and current_name:
                    arabic_names.add(current_name)
                    current_name = ""

            if current_name:
                arabic_names.add(current_name)

    return arabic_names

# Load the MAREFA NER model and tokenizer
model_name = "marefa-nlp/marefa-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Load JSON data from the file
with open("basic_info_frame_2.json", "r", encoding="utf-8") as file:
    json_data = json.load(file)

# Extract names from the JSON data using MAREFA model
arabic_names = extract_arabic_names(json_data, model, tokenizer)

# Print the extracted names
if arabic_names:
    print("Arabic names extracted:")
    for name in arabic_names:
        print("Name:", name)
else:
    print("No Arabic names found.")
