import config, utils_config, utils_mlp_8_knn
from pathlib import Path

dataset_name = 'sst2'
data_folder = config.data_folders[dataset_name]
output_folder = Path("outputs")
exp_id = 'vanilla'
num_classes = 2
train_size = 50

if __name__ == "__main__":

    train_txt_path, train_embedding_path, test_txt_path, test_embedding_path = utils_config.get_txt_paths(data_folder)

    mean_val_acc, stdev_acc = utils_mlp_8_knn.train_mlp_multiple( train_txt_path,
                                                                train_embedding_path,
                                                                test_txt_path,
                                                                test_embedding_path,
                                                                num_classes,
                                                                output_folder,
                                                                exp_id,
                                                                num_seeds=10,
                                                                train_size=train_size,
                                                                )
                
    print(f"{mean_val_acc:.3f},{stdev_acc:.3f}")