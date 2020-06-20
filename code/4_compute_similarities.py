import config
import eda
import utils_config
import utils_bert
from statistics import mean, stdev
from tqdm import tqdm


def compute_insertswap_similarity(source_txt_path, n_aug=10, alpha=0.3):
    
    lines = open(source_txt_path, 'r').readlines()

    dataset_cosine_sim_list = []

    for line in tqdm(lines[:1000]):
        parts = line[:-1].split('\t')
        label = int(parts[0])
        string = parts[1]
        augmented_sentences = eda.get_swap_sentences(string, n_aug=n_aug, alpha=alpha) + eda.get_insert_sentences(string, n_aug=n_aug, alpha=alpha)

        cosine_sim_list = utils_bert.compute_sent_similarities(string, augmented_sentences)
        dataset_cosine_sim_list += cosine_sim_list
    
    return dataset_cosine_sim_list 

def compute_swap_similarity(source_txt_path, n_aug=10, alpha=0.3):
    
    lines = open(source_txt_path, 'r').readlines()

    dataset_cosine_sim_list = []

    for line in tqdm(lines[:1000]):
        parts = line[:-1].split('\t')
        label = int(parts[0])
        string = parts[1]
        augmented_sentences = eda.get_swap_sentences(string, n_aug=n_aug, alpha=alpha)

        cosine_sim_list = utils_bert.compute_sent_similarities(string, augmented_sentences)
        dataset_cosine_sim_list += cosine_sim_list
    
    return dataset_cosine_sim_list

def compute_insert_similarity(source_txt_path, n_aug=10, alpha=0.3):
    
    lines = open(source_txt_path, 'r').readlines()

    dataset_cosine_sim_list = []

    for line in tqdm(lines[:1000]):
        parts = line[:-1].split('\t')
        label = int(parts[0])
        string = parts[1]
        augmented_sentences = eda.get_insert_sentences(string, n_aug=n_aug, alpha=alpha)

        cosine_sim_list = utils_bert.compute_sent_similarities(string, augmented_sentences)
        dataset_cosine_sim_list += cosine_sim_list
    
    return dataset_cosine_sim_list

# def compute_eda_similarity(source_txt_path, n_aug=10, alpha=0.1):
    
#     lines = open(source_txt_path, 'r').readlines()

#     dataset_cosine_sim_list = []

#     for line in tqdm(lines[:10]):
#         parts = line[:-1].split('\t')
#         label = int(parts[0])
#         string = parts[1]
#         augmented_sentences = eda.eda(string, num_aug=n_aug)

#         cosine_sim_list = utils_bert.compute_sent_similarities(string, augmented_sentences)
#         dataset_cosine_sim_list += cosine_sim_list
    
#     return dataset_cosine_sim_list

# def compute_delete_similarity(source_txt_path, n_aug=10, alpha=0.1):
    
#     lines = open(source_txt_path, 'r').readlines()

#     dataset_cosine_sim_list = []

#     for line in tqdm(lines[:50]):
#         parts = line[:-1].split('\t')
#         label = int(parts[0])
#         string = parts[1]
#         augmented_sentences = eda.get_delete_sentences(string, n_aug=n_aug, alpha_rd=alpha)

#         cosine_sim_list = utils_bert.compute_sent_similarities(string, augmented_sentences)
#         dataset_cosine_sim_list += cosine_sim_list
    
#     return dataset_cosine_sim_list

if __name__ == "__main__":



    for method, name in [   (compute_swap_similarity, 'swap'),
                            (compute_insert_similarity, 'insert'),
                            (compute_insertswap_similarity, 'insertswap'),
                            ]:

        all_cosine_sim_list = []
        for dataset_name in ['sst2', 'subj', 'trec']:

            data_folder = config.data_folders[dataset_name]
            delete_folder = utils_config.make_exp_folder(data_folder, 'delete')
            delete_train_txt_path, _, _, _ = utils_config.get_txt_paths(delete_folder)
            dataset_cosine_sim_list = method(delete_train_txt_path)
            all_cosine_sim_list += dataset_cosine_sim_list
        
        print(f"{name} has sim {mean(all_cosine_sim_list):.3f} stdev {stdev(all_cosine_sim_list):.3f}")