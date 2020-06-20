import utils_processing
import copy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from statistics import mean, stdev
import math
import numpy as np
import gc

@ignore_warnings(category=ConvergenceWarning)
def train_svm(  train_x,
                train_y,
                random_state,
                max_iter,
                ):

    clf = make_pipeline(StandardScaler(), svm.LinearSVC(random_state = random_state, max_iter=max_iter))
    clf.fit(train_x, train_y)

    return clf

def evaluate_svm(   clf,
                    test_x,
                    test_y,
                    ):
    
    test_y_pred = clf.predict(test_x)
    acc = accuracy_score(test_y, test_y_pred)
    return acc

def combine_training_sets(  train_x,
                            train_y,
                            aug_train_x,
                            aug_train_y,
                            reg_ratio = 0.5
                            ):
    
    if reg_ratio >= 0.5:
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
    
    else:
        num_target_aug = int((1-reg_ratio) / reg_ratio * len(train_x))
        num_copies = math.ceil(num_target_aug / len(aug_train_x)) - 1
        aug_train_x_copy = np.copy(aug_train_x)
        aug_train_y_copy = np.copy(aug_train_y)
        for _ in range(num_copies):
            aug_train_x_copy = np.concatenate((aug_train_x_copy, aug_train_x), axis=0)
            aug_train_y_copy = np.concatenate((aug_train_y_copy, aug_train_y), axis=0)
        aug_train_x_copy = aug_train_x_copy[:num_target_aug, :]
        aug_train_y_copy = aug_train_y_copy[:num_target_aug,]

        combined_train_x = np.concatenate((train_x, aug_train_x_copy), axis=0)
        combined_train_y = np.concatenate((train_y, aug_train_y_copy), axis=0)
        combined_train_x, combined_train_y = shuffle(combined_train_x, combined_train_y, random_state = 0)

        return combined_train_x, combined_train_y 

def train_eval_svm( train_x, train_y,
                    test_x, test_y,
                    insert_test_x, insert_test_y,
                    swap_test_x, swap_test_y,
                    train_name,
                    n_reg_train_x
                    ):

    max_iter = 10000 * n_reg_train_x / len(train_x)    
    print(f"(n={len(train_x)}) for {max_iter:.1f} iterations : {train_name}")
    reg_acc_list, insert_acc_list, swap_acc_list = [], [], []

    for random_state in [1, 2, 3, 4, 5]:
        clf_reg = train_svm(train_x, train_y, random_state, max_iter = max_iter)
        reg_acc_list.append(evaluate_svm(clf_reg, test_x, test_y))
        insert_acc_list.append(evaluate_svm(clf_reg, insert_test_x, insert_test_y))
        swap_acc_list.append(evaluate_svm(clf_reg, swap_test_x, swap_test_y))
    
    print(f"{mean(reg_acc_list):.4f},{mean(insert_acc_list):.4f},{mean(swap_acc_list):.4f},{stdev(reg_acc_list):.4f},{stdev(insert_acc_list):.4f},{stdev(swap_acc_list):.4f}")

def evaluate_svm_baselines(     train_txt_path,
                                test_txt_path,
                                train_embedding_path,
                                test_embedding_path,
                                insert_test_txt_path,
                                insert_test_embedding_path,
                                swap_test_txt_path,
                                swap_test_embedding_path,
                                delete_train_txt_path,
                                delete_train_embedding_path,
                                eda_train_txt_path,
                                eda_train_embedding_path,
                                ):

    train_x, train_y = utils_processing.get_x_y(train_txt_path, train_embedding_path)
    test_x, test_y = utils_processing.get_x_y(test_txt_path, test_embedding_path)
    insert_test_x, insert_test_y = utils_processing.get_x_y(insert_test_txt_path, insert_test_embedding_path)
    swap_test_x, swap_test_y = utils_processing.get_x_y(swap_test_txt_path, swap_test_embedding_path)
    delete_train_x, delete_train_y = utils_processing.get_x_y(delete_train_txt_path, delete_train_embedding_path)
    eda_train_x, eda_train_y = utils_processing.get_x_y(eda_train_txt_path, eda_train_embedding_path)
    
    train_eval_svm( train_x, train_y,
                    test_x, test_y,
                    insert_test_x, insert_test_y,
                    swap_test_x, swap_test_y, 
                    "regular training",
                    n_reg_train_x = len(train_x)
                    )

    train_eval_svm( delete_train_x, delete_train_y,
                    test_x, test_y,
                    insert_test_x, insert_test_y,
                    swap_test_x, swap_test_y, 
                    "delete",
                    n_reg_train_x = len(train_x)
                    )

    train_eval_svm( eda_train_x, eda_train_y,
                    test_x, test_y,
                    insert_test_x, insert_test_y,
                    swap_test_x, swap_test_y, 
                    "eda",
                    n_reg_train_x = len(train_x)
                    )

def evaluate_svm_big_ablation(  train_txt_path,
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
                                ):

    train_x, train_y = utils_processing.get_x_y(train_txt_path, train_embedding_path)
    test_x, test_y = utils_processing.get_x_y(test_txt_path, test_embedding_path)
    insert_train_x, insert_train_y = utils_processing.get_x_y(insert_train_txt_path, insert_train_embedding_path)
    insert_test_x, insert_test_y = utils_processing.get_x_y(insert_test_txt_path, insert_test_embedding_path)
    swap_train_x, swap_train_y = utils_processing.get_x_y(swap_train_txt_path, swap_train_embedding_path)
    swap_test_x, swap_test_y = utils_processing.get_x_y(swap_test_txt_path, swap_test_embedding_path)
    insertswap_aug_train_x = np.concatenate((insert_train_x, swap_train_x), axis=0)
    insertswap_aug_train_y = np.concatenate((insert_train_y, swap_train_y), axis=0)

    for aug_train_x, aug_train_y, aug_type in [
                                                (insert_train_x, insert_train_y, 'insert'),
                                                (swap_train_x, swap_train_y, 'swap'),
                                                (insertswap_aug_train_x, insertswap_aug_train_y, 'insertswap'),
                                                ]:

        # train_eval_svm( aug_train_x, aug_train_y,
        #                 test_x, test_y,
        #                 insert_test_x, insert_test_y,
        #                 swap_test_x, swap_test_y, 
        #                 f"only {aug_type}",
        #                 n_reg_train_x = len(train_x)
        #                 )

        for reg_ratio in [0.1]: #[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

            combined_train_x, combined_train_y = combine_training_sets(train_x, train_y, aug_train_x, aug_train_y, reg_ratio = reg_ratio)
            train_eval_svm( combined_train_x, combined_train_y,
                            test_x, test_y,
                            insert_test_x, insert_test_y,
                            swap_test_x, swap_test_y, 
                            f"reg {reg_ratio} + {aug_type}",
                            n_reg_train_x = len(train_x)
                            )
            gc.collect()
