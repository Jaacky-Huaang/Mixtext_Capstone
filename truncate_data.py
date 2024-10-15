import json
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split
import pdb

def load_json(input_file):
    with open(input_file, 'r') as file:
        return json.load(file)

def save_json(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

# Load tokenizer for GPT-2 Medium
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

def truncate_data(data, max_length=1023): # set to 1023 to avoid dimension error
    for dataset in ['train', 'test']:
        for i in range(len(data[dataset]['text'])):
            # Encode and truncate the text
            tokens = tokenizer(data[dataset]['text'][i], max_length=max_length, truncation=True, return_tensors="pt")
            # Decode tokens to string and update
            data[dataset]['text'][i] = tokenizer.decode(tokens['input_ids'][0])
    return data


def convert_format(input_file, output_file):
    data = load_json(input_file)
    data = truncate_data(data)
    save_json(data, output_file)

if __name__ == "__main__":
    convert_format("/scratch/jh7956/mixset_data/split_data.json", "/scratch/jh7956/mixset_data/truncated_data.json")
    print("Data conversion completed!")
