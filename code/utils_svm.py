import utils_processing
import copy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import math
import numpy as np

def train_svm(  train_x,
                train_y,
                ):
    
    print(f"train_x = {train_x.shape}, train_y={train_y.shape}")
    clf = make_pipeline(StandardScaler(), svm.LinearSVC())
    clf.fit(train_x, train_y)

    return clf

def evaluate_svm(   clf,
                    test_x,
                    test_y,
                    exp_str,
                    ):
    
    test_y_pred = clf.predict(test_x)
    acc = accuracy_score(test_y, test_y_pred)
    print(f"{exp_str} (n_test = {test_y_pred.shape[0]}): {acc:.4f}")

def combine_training_sets(  train_x,
                            train_y,
                            aug_train_x,
                            aug_train_y,
                            reg_ratio = 0.5
                            ):
    
    num_target_reg = int(reg_ratio / (1-reg_ratio) * len(aug_train_x))
    num_copies = math.ceil(num_target_reg / len(train_x)) - 1
    train_x_copy = np.copy(train_x)
    train_y_copy = np.copy(train_y)
    for _ in range(num_copies):
        train_x_copy = np.concatenate((train_x_copy, train_x), axis=0)
        train_y_copy = np.concatenate((train_y_copy, train_y), axis=0)
    train_x_copy = train_x_copy[:num_target_reg, :]
    train_y_copy = train_y_copy[:num_target_reg,]

    combined_train_x = np.concatenate((train_x_copy, aug_train_x), axis=0)
    combined_train_y = np.concatenate((train_y_copy, aug_train_y), axis=0)
    combined_train_x, combined_train_y = shuffle(combined_train_x, combined_train_y, random_state = 0)

    return combined_train_x, combined_train_y

def evaluate_svm_experiments(   train_txt_path,
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
                                ):

    train_x, train_y = utils_processing.get_x_y(train_txt_path, train_embedding_path)
    test_x, test_y = utils_processing.get_x_y(test_txt_path, test_embedding_path)
    insert_train_x, insert_train_y = utils_processing.get_x_y(insert_train_txt_path, insert_train_embedding_path)
    insert_test_x, insert_test_y = utils_processing.get_x_y(insert_test_txt_path, insert_test_embedding_path)
    delete_train_x, delete_train_y = utils_processing.get_x_y(delete_train_txt_path, delete_train_embedding_path)
    delete_test_x, delete_test_y = utils_processing.get_x_y(delete_test_txt_path, delete_test_embedding_path)
    swap_train_x, swap_train_y = utils_processing.get_x_y(swap_train_txt_path, swap_train_embedding_path)
    swap_test_x, swap_test_y = utils_processing.get_x_y(swap_test_txt_path, swap_test_embedding_path)
    
    clf_reg = train_svm(train_x, train_y)
    evaluate_svm(clf_reg, test_x, test_y, "reg on reg")
    evaluate_svm(clf_reg, insert_test_x, insert_test_y, "reg on insert")
    evaluate_svm(clf_reg, delete_test_x, delete_test_y, "reg on delete")
    evaluate_svm(clf_reg, swap_test_x, swap_test_y, "reg on swap")
    print()

    for aug_train_x, aug_train_y, aug_type in [
                                        (insert_train_x, insert_train_y, 'insert'),
                                        (delete_train_x, delete_train_y, 'delete'),
                                        (swap_train_x, swap_train_y, 'swap'),
                                        ]:

        clf_aug = train_svm(aug_train_x, aug_train_y)
        evaluate_svm(clf_aug, test_x, test_y, f"{aug_type} on reg")
        evaluate_svm(clf_aug, insert_test_x, insert_test_y, f"{aug_type} on insert")
        evaluate_svm(clf_aug, delete_test_x, delete_test_y, f"{aug_type} on delete")
        evaluate_svm(clf_aug, swap_test_x, swap_test_y, f"{aug_type} on swap")

        for reg_ratio in [0.5, 0.75, 0.9]:
            combined_train_x, combined_train_y = combine_training_sets(train_x, train_y, aug_train_x, aug_train_y, reg_ratio = reg_ratio)
            clf_combined = train_svm(combined_train_x, combined_train_y)
            evaluate_svm(clf_combined, test_x, test_y, f"{aug_type} combined {reg_ratio} on reg")
            evaluate_svm(clf_combined, insert_test_x, insert_test_y, f"{aug_type} combined {reg_ratio} on insert")
            evaluate_svm(clf_combined, delete_test_x, delete_test_y, f"{aug_type} combined {reg_ratio} on delete")
            evaluate_svm(clf_combined, swap_test_x, swap_test_y, f"{aug_type} combined {reg_ratio} on swap")

            clf_reg_copy = copy.deepcopy(clf_reg)
            clf_reg_copy.fit(combined_train_x, combined_train_y)
            evaluate_svm(clf_reg_copy, test_x, test_y, f"{aug_type} CL {reg_ratio} on reg")
            evaluate_svm(clf_reg_copy, insert_test_x, insert_test_y, f"{aug_type} CL {reg_ratio} on insert")
            evaluate_svm(clf_reg_copy, delete_test_x, delete_test_y, f"{aug_type} CL {reg_ratio} on delete")
            evaluate_svm(clf_reg_copy, swap_test_x, swap_test_y, f"{aug_type} CL {reg_ratio} on swap")