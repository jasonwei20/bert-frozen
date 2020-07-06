import config, utils_config, utils_mlp_9_analyze_gradients
from pathlib import Path

dataset_name = 'sst2'
data_folder = config.data_folders[dataset_name]
output_folder = Path("outputs")
exp_id = '9b'
num_classes = 2
train_subset = 200
resume_checkpoint_path = Path("checkpoints/sst2_small/e0.pt")

if __name__ == "__main__":

    train_txt_path, train_embedding_path, test_txt_path, test_embedding_path = utils_config.get_txt_paths(data_folder)

    utils_mlp_9_analyze_gradients.train_mlp_multiple(   
        train_txt_path,
        train_embedding_path,
        test_txt_path,
        test_embedding_path,
        num_classes = num_classes,
        dataset_name = dataset_name,
        exp_id = exp_id,
        train_subset = train_subset,
        resume_checkpoint_path = resume_checkpoint_path,
        num_seeds = 1,
        )
                