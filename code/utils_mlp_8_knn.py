import numpy as np
import operator
from sklearn.utils import shuffle
from statistics import mean, stdev
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import gc

import utils_autograd_hacks as autograd_hacks
import utils_grad
import utils_processing
import utils_mlp_helper

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class Net(nn.Module):

    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(768, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

def train_mlp_checkpoint(  
                        train_txt_path,
                        train_embedding_path,
                        test_txt_path,
                        test_embedding_path,
                        num_classes,
                        seed_num,
                        minibatch_size,
                        num_epochs,
                        criterion,
                        checkpoint_folder,
                        ):

    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    
    train_x, train_y = utils_processing.get_x_y(train_txt_path, train_embedding_path)
    train_x, ul_x, train_y, ul_y = train_test_split(train_x, train_y, train_size=20, random_state=seed_num, stratify=train_y)
    # test_x, test_y = utils_processing.get_x_y(test_txt_path, test_embedding_path)

    print(train_x.shape, train_y.shape, ul_x.shape, ul_y.shape)

    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(train_x, train_y)
    ul_y_predict = neigh.predict(ul_x)
    ul_acc = accuracy_score(ul_y, ul_y_predict)
        # print(f"{train_loss:.3f},{train_acc:.3f},{val_loss:.3f},{val_acc:.3f}\n")

    gc.collect()
    return ul_acc

def train_mlp_multiple(  
                        train_txt_path,
                        train_embedding_path,
                        test_txt_path,
                        test_embedding_path,
                        num_classes,
                        output_folder,
                        exp_id,
                        num_seeds,
                        minibatch_size = 5,
                        num_epochs = 20,
                        criterion = nn.CrossEntropyLoss(),
                        ):

    val_acc_list = []

    for seed_num in range(num_seeds):

        val_acc = train_mlp_checkpoint(  
                            train_txt_path,
                            train_embedding_path,
                            test_txt_path,
                            test_embedding_path,
                            num_classes,
                            seed_num,
                            minibatch_size,
                            num_epochs,
                            criterion,
                            checkpoint_folder = None
                            )

        val_acc_list.append(val_acc)

    val_acc_stdev = stdev(val_acc_list) if len(val_acc_list) >= 2 else -1 
    return mean(val_acc_list), val_acc_stdev