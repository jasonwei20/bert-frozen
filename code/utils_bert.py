import utils_common
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import gc
from sklearn.metrics.pairwise import cosine_similarity

model_class = BertModel
tokenizer_class = BertTokenizer
pretrained_weights = 'bert-base-uncased'
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

def get_embedding_dict(input_file, output_dict_path):

    string_to_embedding = {}
    
    lines = open(input_file, 'r').readlines()
    print(f"string_to_embedding of size {len(lines)} saving to {output_dict_path}")

    for line in tqdm(lines):
        parts = line[:-1].split('\t')
        string = parts[1]
        embedding = get_embedding(string, tokenizer, model)
        string_to_embedding[string] = embedding

    utils_common.save_pickle(output_dict_path, string_to_embedding)
    gc.collect()

# Encode text
def get_embedding(input_string, tokenizer, model):
    input_ids = torch.tensor([tokenizer.encode(input_string, add_special_tokens = True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0].numpy()  # Models outputs are now tuples
        # last_hidden_states = last_hidden_states[:, 0, :]
        last_hidden_states = np.mean(last_hidden_states, axis = 1)
        last_hidden_states = last_hidden_states.flatten()
        return last_hidden_states

def compute_sent_similarities(original_sentence, augmented_sentences):
    
    original_sentence_embedding = get_embedding(original_sentence, tokenizer, model)
    
    cosine_sim_list = []
    for augmented_sentence in augmented_sentences:
        augmented_sentence_embedding = get_embedding(augmented_sentence, tokenizer, model)
        cosine_sim = float(cosine_similarity(X=[original_sentence_embedding], Y=[augmented_sentence_embedding])[0][0])
        cosine_sim_list.append(cosine_sim)
    
    return cosine_sim_list
    