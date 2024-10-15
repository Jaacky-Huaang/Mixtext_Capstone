# Current json format
# {
#     "0": {
#         "abstract": "We present the design and implementation of a custom discrete optimization technique for building rule lists over a categorical feature space. Our algorithm provides the optimal solution, with a certificate of optimality. By leveraging algorithmic bounds, efficient data structures, and computational reuse, we achieve several orders of magnitude speedup in time and a massive reduction of memory consumption. We demonstrate that our approach produces optimal rule lists on practical problems in seconds. This framework is a novel alternative to CART and other decision tree methods.",
#         "title": "Learning Certifiably Optimal Rule Lists",
#         "id": "/doi/10.1145/3097983.3098047",
#         "type": "human_written"
#     },
#     "1": {
#         "abstract": "Consider a random preferential attachment model G(p) for network evolution that allows both node and edge arrivals. Starting with an arbitrary nonempty graph G0, at each time step, there are two possible events: with probability p > 0 a new node arrives and a new edge is added between the new node and an existing node, and with probability 1 - p a new edge is added between two existing nodes. In both cases, the involved existing nodes are chosen at random according to preferential attachment, i.e., with probability proportional to their degree. G(p) is known to generate power law networks, i.e., the fraction of nodes with degree k is proportional to k-β. Here β=(4-p)/(2-p) is in the range (2,3].",
#         "title": "Improved Degree Bounds and Full Spectrum Power Laws in Preferential Attachment Networks",
#         "id": "/doi/10.1145/3097983.3098012",
#         "type": "human_written"
#     },

# Target json format
# {
    # "train": 
    #     {
    #         "text": ["Sentence 1", "Sentence 2", "Sentence 3", ...],
    #         "label": [0, 1, 0, ...]
    #     }
    # "test": 
    #     {
    #         "text": ["Sentence 1", "Sentence 2", "Sentence 3", ...],
    #         "label": [0, 1, 0, ...]
    #     }



import json
from sklearn.model_selection import train_test_split

def load_json(input_file):
    with open(input_file, 'r') as file:
        return json.load(file)

def save_json(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

def process_data(data):
    texts = []
    labels = []
    for key, value in data.items():
        texts.append(value['abstract'])
        if value['type'] == 'human_written':
            labels.append(0)
        elif value['type'] == 'machine_written':
            labels.append(1)
        else:
            labels.append(2)
    return texts, labels

def split_data(texts, labels):
    # Split the data into training and testing sets (80/20 split)
    text_train, text_test, label_train, label_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    print("Training set size: ", len(text_train))
    print("Testing set size: ", len(text_test))
    return {
        'train': {'text': text_train, 'label': label_train},
        'test': {'text': text_test, 'label': label_test}
    }

def convert_format(input_file, output_file):
    raw_data = load_json(input_file)
    texts, labels = process_data(raw_data)
    final_data = split_data(texts, labels)
    save_json(final_data, output_file)

if __name__ == "__main__":
    convert_format("/scratch/jh7956/original_data/labeled_data.json", "/scratch/jh7956/original_data/split_data.json")
    print("Data conversion completed!")



