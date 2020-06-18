import config, utils_bert, utils_config, utils_svm, utils_processing

data_folder = config.data_folders['trec']

if __name__ == "__main__":

    train_txt_path, train_embedding_path, test_txt_path, test_embedding_path = utils_config.get_txt_paths(data_folder)  
    # utils_bert.get_embedding_dict(train_txt_path, train_embedding_path)
    # utils_bert.get_embedding_dict(test_txt_path, test_embedding_path)

    insert_folder = utils_config.make_exp_folder(data_folder, f"insert-eval")
    _, _, insert_test_txt_path, insert_test_embedding_path = utils_config.get_txt_paths(insert_folder)
    # utils_processing.augment_insert(test_txt_path, insert_test_txt_path, n_aug=2, alpha=0.3)
    # utils_bert.get_embedding_dict(insert_test_txt_path, insert_test_embedding_path)

    swap_folder = utils_config.make_exp_folder(data_folder, f"swap-eval")
    _, _, swap_test_txt_path, swap_test_embedding_path = utils_config.get_txt_paths(swap_folder)
    # utils_processing.augment_swap(test_txt_path, swap_test_txt_path, n_aug=2, alpha=0.2)
    # utils_bert.get_embedding_dict(swap_test_txt_path, swap_test_embedding_path)

    # delete_folder = utils_config.make_exp_folder(data_folder, 'delete')
    # delete_train_txt_path, delete_train_embedding_path, _, _ = utils_config.get_txt_paths(delete_folder)
    # utils_processing.augment_delete(train_txt_path, delete_train_txt_path)
    # utils_bert.get_embedding_dict(delete_train_txt_path, delete_train_embedding_path)

    # eda_folder = utils_config.make_exp_folder(data_folder, 'eda')
    # eda_train_txt_path, eda_train_embedding_path, _, _ = utils_config.get_txt_paths(eda_folder)
    # utils_processing.augment_eda(train_txt_path, eda_train_txt_path)
    # utils_bert.get_embedding_dict(eda_train_txt_path, eda_train_embedding_path)

    # utils_svm.evaluate_svm_baselines(   train_txt_path,
    #                                     test_txt_path,
    #                                     train_embedding_path,
    #                                     test_embedding_path,
    #                                     insert_test_txt_path,
    #                                     insert_test_embedding_path,
    #                                     swap_test_txt_path,
    #                                     swap_test_embedding_path,
    #                                     delete_train_txt_path,
    #                                     delete_train_embedding_path,
    #                                     eda_train_txt_path,
    #                                     eda_train_embedding_path,
    #                                     )

    for alpha in [0.5, 0.4, 0.3, 0.2, 0.1]: #0.05, #0.1, #, 0.15, 0.25, 0.35, 0.45

        insert_folder = utils_config.make_exp_folder(data_folder, f"insert{alpha}")
        insert_train_txt_path, insert_train_embedding_path, _, _ = utils_config.get_txt_paths(insert_folder)
        # utils_processing.augment_insert(train_txt_path, insert_train_txt_path, n_aug=2, alpha=alpha)
        # utils_bert.get_embedding_dict(insert_train_txt_path, insert_train_embedding_path)

        swap_folder = utils_config.make_exp_folder(data_folder, f"swap{alpha}")
        swap_train_txt_path, swap_train_embedding_path, _, _ = utils_config.get_txt_paths(swap_folder)
        # utils_processing.augment_swap(train_txt_path, swap_train_txt_path, n_aug=2, alpha=alpha)
        # utils_bert.get_embedding_dict(swap_train_txt_path, swap_train_embedding_path)

        utils_svm.evaluate_svm_big_ablation(train_txt_path,
                                            test_txt_path,
                                            train_embedding_path,
                                            test_embedding_path,
                                            insert_train_txt_path,
                                            insert_test_txt_path,
                                            insert_train_embedding_path,
                                            insert_test_embedding_path,
                                            swap_train_txt_path,
                                            swap_test_txt_path,
                                            swap_train_embedding_path,
                                            swap_test_embedding_path,
                                            )