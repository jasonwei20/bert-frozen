import utils_common
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

model_class = BertModel
tokenizer_class = BertTokenizer
pretrained_weights = 'bert-base-uncased'
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

def get_embedding_dict(input_file, output_dict_path):

    string_to_embedding = {}
    
    lines = open(input_file, 'r').readlines()
    for line in tqdm(lines):
        parts = line[:-1].split('\t')
        string = parts[1]
        embedding = get_embedding(string, tokenizer, model)
        string_to_embedding[string] = embedding

    utils_common.save_pickle(output_dict_path, string_to_embedding)
    print(f"string_to_embedding of size {len(string_to_embedding)} saved to {output_dict_path}")

# Encode text
def get_embedding(input_string, tokenizer, model):
    input_ids = torch.tensor([tokenizer.encode(input_string, add_special_tokens = True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0].numpy()  # Models outputs are now tuples
        # last_hidden_states = last_hidden_states[:, 0, :]
        last_hidden_states = np.mean(last_hidden_states, axis = 1)
        last_hidden_states = last_hidden_states.flatten()
        return last_hidden_states