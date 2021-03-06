import config, utils_config, utils_mlp
from pathlib import Path

dataset_name = 'sst2'
data_folder = config.data_folders[dataset_name]
num_classes = config.num_classes_dict[dataset_name]
output_folder = Path("outputs")
exp_id = 'sst_1'
annealling = 4
top_k = 32

if __name__ == "__main__":

    train_txt_path, train_embedding_path, test_txt_path, test_embedding_path = utils_config.get_txt_paths(data_folder)

    for flip_ratio in [0.2]: #[0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0]:

        for top_k in [10]: #[4, 16, 32, 64]:

            mean_val_acc, stdev_acc = utils_mlp.train_mlp_multiple( train_txt_path,
                                                                    train_embedding_path,
                                                                    test_txt_path,
                                                                    test_embedding_path,
                                                                    flip_ratio,
                                                                    num_classes,
                                                                    output_folder,
                                                                    exp_id,
                                                                    top_k=top_k,
                                                                    annealling=annealling,
                                                                    num_seeds=10,
                                                                    )
            
            print(f"{flip_ratio},{mean_val_acc:.3f},{stdev_acc:.3f}")
