import numpy as np
from sklearn.utils import shuffle
from statistics import mean, stdev
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils_processing
import utils_mlp_helper

class Net(nn.Module):

    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(768, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

def train_mlp(  
                train_txt_path,
                train_embedding_path,
                test_txt_path,
                test_embedding_path,
                flip_ratio,
                num_classes,
                seed_num,
                minibatch_size,
                num_epochs,
                criterion,
                ):

    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    
    train_x, train_y = utils_processing.get_x_y(train_txt_path, train_embedding_path)
    test_x, test_y = utils_processing.get_x_y(test_txt_path, test_embedding_path)

    train_y, train_y_orig = utils_mlp_helper.get_labels_uniform_flip(train_y, flip_ratio, num_classes)

    # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    model = Net(num_classes=num_classes)
    optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.05) #wow, works for even large learning rates
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    num_minibatches_train = int(train_x.shape[0] / minibatch_size)
    num_minibatches_val = int(test_x.shape[0] / minibatch_size)
    val_acc_list = []

    for epoch in range(1, num_epochs + 1):

        ######## training ########
        model.train(mode=True)
        train_running_loss, train_running_corrects = 0.0, 0

        train_x, train_y = shuffle(train_x, train_y, random_state = seed_num)

        for minibatch_num in range(num_minibatches_train):
            
            start_idx = minibatch_num * minibatch_size
            end_idx = start_idx + minibatch_size
            train_inputs = torch.from_numpy(train_x[start_idx:end_idx].astype(np.float32))
            train_labels = torch.from_numpy(train_y[start_idx:end_idx].astype(np.long))
            optimizer.zero_grad()

            # Forward and backpropagation.
            with torch.set_grad_enabled(mode=True):
                train_outputs = model(train_inputs)
                __, train_preds = torch.max(train_outputs, dim=1)
                train_loss = criterion(input=train_outputs, target=train_labels)
                train_loss.backward(retain_graph=True)
                # autograd_hacks.compute_grad1(model)

                optimizer.step()

            train_running_loss += train_loss.item() * train_inputs.size(0)
            train_running_corrects += int(torch.sum(train_preds == train_labels.data, dtype=torch.double))

        train_loss = train_running_loss / (num_minibatches_train * minibatch_size)
        train_acc = train_running_corrects / (num_minibatches_train * minibatch_size)

        ######## validation ########
        model.train(mode=False)
        val_running_loss, val_running_corrects = 0.0, 0

        for minibatch_num in range(num_minibatches_val):
            
            start_idx = minibatch_num * minibatch_size
            end_idx = start_idx + minibatch_size
            val_inputs = torch.from_numpy(test_x[start_idx:end_idx].astype(np.float32))
            val_labels = torch.from_numpy(test_y[start_idx:end_idx].astype(np.long))

            # Feed forward.
            with torch.set_grad_enabled(mode=False):
                val_outputs = model(val_inputs)
                _, val_preds = torch.max(val_outputs, dim=1)
                val_loss = criterion(input=val_outputs, target=val_labels)
            val_running_loss += val_loss.item() * val_inputs.size(0)
            val_running_corrects += int(torch.sum(val_preds == val_labels.data, dtype=torch.double))

        val_loss = val_running_loss / (num_minibatches_val * minibatch_size)
        val_acc = val_running_corrects / (num_minibatches_val * minibatch_size)
        val_acc_list.append(val_acc)

        # print(f"train loss={train_loss:.3f} acc={train_acc:.3f}; val loss = {val_loss:.3f} acc={val_acc:.3f}")
    
    return mean(val_acc_list[-5:])

def train_mlp_multiple(  
                train_txt_path,
                train_embedding_path,
                test_txt_path,
                test_embedding_path,
                flip_ratio,
                num_classes,
                num_seeds,
                minibatch_size = 256,
                num_epochs = 10,
                criterion = nn.CrossEntropyLoss(),
                ):
    
    val_acc_list = []

    for seed_num in range(num_seeds):
        val_acc = train_mlp(  
                            train_txt_path,
                            train_embedding_path,
                            test_txt_path,
                            test_embedding_path,
                            flip_ratio,
                            num_classes,
                            seed_num,
                            minibatch_size,
                            num_epochs,
                            criterion,
                            )
        val_acc_list.append(val_acc)
    
    return mean(val_acc_list), stdev(val_acc_list)