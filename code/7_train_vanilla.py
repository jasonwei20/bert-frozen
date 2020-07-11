import config, utils_config, utils_mlp_7_vanilla
from pathlib import Path

dataset_name = 'sst2'
data_folder = config.data_folders[dataset_name]
output_folder = Path("outputs")
exp_id = 'vanilla_mlp'
num_classes = 2
minibatch_size = 5

if __name__ == "__main__":

    train_txt_path, train_embedding_path, test_txt_path, test_embedding_path = utils_config.get_txt_paths(data_folder)

    for train_subset in [10, 20, 50]:

        mean_val_acc, stdev_acc, mean_conf_acc, stdev_conf_acc = utils_mlp_7_vanilla.train_mlp_multiple(   
            train_txt_path,
            train_embedding_path,
            test_txt_path,
            test_embedding_path,
            num_classes,
            dataset_name,
            exp_id,
            train_subset,
            minibatch_size=minibatch_size,
            num_seeds=1,
            )
                    
        print(f"{train_subset},{mean_val_acc:.3f},{stdev_acc:.3f},{mean_conf_acc:.3f},{stdev_conf_acc:.3f}")   