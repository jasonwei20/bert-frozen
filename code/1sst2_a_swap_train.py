import config, utils_bert, utils_config, utils_svm, utils_processing

data_folder = config.data_folders['sst2']
train_txt_path, train_embedding_path, test_txt_path, test_embedding_path = utils_config.get_txt_paths(data_folder)

insert_folder = utils_config.make_exp_folder(data_folder, 'insert')
insert_train_txt_path, insert_train_embedding_path, insert_test_txt_path, insert_test_embedding_path = utils_config.get_txt_paths(insert_folder)

delete_folder = utils_config.make_exp_folder(data_folder, 'delete')
delete_train_txt_path, delete_train_embedding_path, delete_test_txt_path, delete_test_embedding_path = utils_config.get_txt_paths(delete_folder)

swap_folder = utils_config.make_exp_folder(data_folder, 'swap')
swap_train_txt_path, swap_train_embedding_path, swap_test_txt_path, swap_test_embedding_path = utils_config.get_txt_paths(swap_folder)

if __name__ == "__main__":

    # utils_bert.get_embedding_dict(train_txt_path, train_embedding_path)
    # utils_bert.get_embedding_dict(test_txt_path, test_embedding_path)

    # utils_processing.augment_insert(train_txt_path, insert_train_txt_path, n_aug=2)
    # utils_processing.augment_insert(test_txt_path, insert_test_txt_path, n_aug=2)
    # utils_bert.get_embedding_dict(insert_train_txt_path, insert_train_embedding_path)
    # utils_bert.get_embedding_dict(insert_test_txt_path, insert_test_embedding_path)

    # utils_processing.augment_delete(train_txt_path, delete_train_txt_path, n_aug=2)
    # utils_processing.augment_delete(test_txt_path, delete_test_txt_path, n_aug=2)
    # utils_bert.get_embedding_dict(delete_train_txt_path, delete_train_embedding_path)
    # utils_bert.get_embedding_dict(delete_test_txt_path, delete_test_embedding_path)

    # utils_processing.augment_swap(train_txt_path, swap_train_txt_path, n_aug=2)
    # utils_processing.augment_swap(test_txt_path, swap_test_txt_path, n_aug=2)
    # utils_bert.get_embedding_dict(swap_train_txt_path, swap_train_embedding_path)
    # utils_bert.get_embedding_dict(swap_test_txt_path, swap_test_embedding_path)

    utils_svm.evaluate_svm_experiments( train_txt_path,
                                        test_txt_path,
                                        train_embedding_path,
                                        test_embedding_path,
                                        insert_train_txt_path,
                                        insert_test_txt_path,
                                        insert_train_embedding_path,
                                        insert_test_embedding_path,
                                        delete_train_txt_path,
                                        delete_test_txt_path,
                                        delete_train_embedding_path,
                                        delete_test_embedding_path,
                                        swap_train_txt_path,
                                        swap_test_txt_path,
                                        swap_train_embedding_path,
                                        swap_test_embedding_path,
                                        )