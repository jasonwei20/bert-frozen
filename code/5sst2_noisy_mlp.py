import config, utils_config, utils_mlp

data_folder = config.data_folders['sst2']

if __name__ == "__main__":

    train_txt_path, train_embedding_path, test_txt_path, test_embedding_path = utils_config.get_txt_paths(data_folder)
    num_classes = 2

    for flip_ratio in [0.5, 0.4, 0.3, 0.2, 0.1, 0]:

        mean_val_acc, stdev_acc = utils_mlp.train_mlp_multiple( train_txt_path,
                                                                train_embedding_path,
                                                                test_txt_path,
                                                                test_embedding_path,
                                                                flip_ratio,
                                                                num_classes,
                                                                num_seeds=5
                                                                )
        
        print(f"flip ratio {flip_ratio} has acc {mean_val_acc:.3f} with stdev {stdev_acc:.3f}")
