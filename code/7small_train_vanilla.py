import config, utils_config, utils_mlp_7_vanilla
from pathlib import Path

dataset_name = 'sst2'
data_folder = config.data_folders[dataset_name]
output_folder = Path("outputs")
exp_id = 'small'
num_classes = 2
train_subset = 20
minibatch_size = 5

if __name__ == "__main__":

    train_txt_path, train_embedding_path, test_txt_path, test_embedding_path = utils_config.get_txt_paths(data_folder)

    mean_val_acc, stdev_acc = utils_mlp_7_vanilla.train_mlp_multiple(   train_txt_path,
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
                
    print(f"{mean_val_acc:.3f},{stdev_acc:.3f}")