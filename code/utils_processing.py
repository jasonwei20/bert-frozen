import utils_common
import numpy as np
from sklearn.utils import shuffle
import pathlib
import eda

def get_x_y(txt_path, embedding_path):
    lines = open(txt_path).readlines()
    string_to_embedding = utils_common.load_pickle(embedding_path)

    x = np.zeros((len(lines), 768))
    y = np.zeros((len(lines), ))

    for i, line in enumerate(lines):
        parts = line[:-1].split('\t')
        label = int(parts[0])
        string = parts[1]
        assert string in string_to_embedding
        embedding = string_to_embedding[string]
        x[i, :] = embedding
        y[i] = label
    
    x, y = shuffle(x, y, random_state = 0)
    return x, y

def augment_swap(source_txt_path, target_txt_path, n_aug):
    
    writer = open(target_txt_path, 'w')
    lines = open(source_txt_path, 'r').readlines()
    for line in lines:
        parts = line[:-1].split('\t')
        label = int(parts[0])
        string = parts[1]
        augmented_sentences = eda.get_swap_sentences(string, n_aug=n_aug)

        for augmented_sentence in augmented_sentences:
            output_line = '\t'.join([str(label), augmented_sentence])
            writer.write(output_line + '\n')
    
    print(f"output file at {target_txt_path}")

def augment_insert(source_txt_path, target_txt_path, n_aug):
    
    writer = open(target_txt_path, 'w')
    lines = open(source_txt_path, 'r').readlines()
    for line in lines:
        parts = line[:-1].split('\t')
        label = int(parts[0])
        string = parts[1]
        augmented_sentences = eda.get_insert_sentences(string, n_aug=n_aug)

        for augmented_sentence in augmented_sentences:
            output_line = '\t'.join([str(label), augmented_sentence])
            writer.write(output_line + '\n')
    
    print(f"output file at {target_txt_path}")

def augment_delete(source_txt_path, target_txt_path, n_aug):
    
    writer = open(target_txt_path, 'w')
    lines = open(source_txt_path, 'r').readlines()
    for line in lines:
        parts = line[:-1].split('\t')
        label = int(parts[0])
        string = parts[1]
        augmented_sentences = eda.get_delete_sentences(string, n_aug=n_aug)

        for augmented_sentence in augmented_sentences:
            output_line = '\t'.join([str(label), augmented_sentence])
            writer.write(output_line + '\n')
    
    print(f"output file at {target_txt_path}")