import config, utils_config, utils_mlp_a1_grad_sim_classification
from pathlib import Path

dataset_name = 'sst2'
data_folder = config.data_folders[dataset_name]
output_folder = Path("outputs")
exp_id = '9b'
num_classes = 2
resume_checkpoint_path = Path("checkpoints/sst2_vanilla_mlp_50_seednum0/e10_va0.7460.pt")

if __name__ == "__main__":

    train_txt_path, train_embedding_path, test_txt_path, test_embedding_path = utils_config.get_txt_paths(data_folder)

    for train_subset in [500]:#, 20, 50]:

        mean_val_acc, stdev_acc, mean_conf_acc, stdev_conf_acc, mlp_val_acc, mlp_stdev_acc, mlp_conf_acc, mlp_stdev_conf_acc = utils_mlp_a1_grad_sim_classification.train_mlp_multiple(   
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
        
        print(f"{train_subset},{mean_val_acc:.3f},{stdev_acc:.3f},{mean_conf_acc:.3f},{stdev_conf_acc:.3f}, {mlp_val_acc:.3f}, {mlp_stdev_acc:.3f}, {mlp_conf_acc:.3f}, {mlp_stdev_conf_acc:.3f}")