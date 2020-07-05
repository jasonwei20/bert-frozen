import numpy as np
import operator
from pathlib import Path
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

class Net(nn.Module):

    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(768, num_classes)
        # self.fc1 = nn.Linear(768, 50)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        # x = self.relu1(x)
        # x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train_mlp_checkpoint(  
                        train_txt_path,
                        train_embedding_path,
                        test_txt_path,
                        test_embedding_path,
                        num_classes,
                        train_subset,
                        resume_checkpoint_path,
                        num_epochs,
                        train_minibatch_size,
                        ul_minibatch_size,
                        seed_num,
                        criterion,
                        checkpoint_folder,
                        val_minibatch_size = 256,
                        ):

    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    
    train_x, train_y = utils_processing.get_x_y(train_txt_path, train_embedding_path)
    test_x, test_y = utils_processing.get_x_y(test_txt_path, test_embedding_path)
    if train_subset:
        train_x, ul_x, train_y, ul_y = train_test_split(train_x, train_y, train_size=train_subset, random_state=42, stratify=train_y)

    print(train_x.shape, train_y.shape, ul_x.shape, ul_y.shape, test_x.shape, test_y.shape)

    model = Net(num_classes=num_classes)
    optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.05) #wow, works for even large learning rates
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    
    # if resume_checkpoint_path:
    #     ckpt = torch.load(f=resume_checkpoint_path)
    #     model.load_state_dict(state_dict=ckpt["model_state_dict"])
    #     optimizer.load_state_dict(state_dict=ckpt["optimizer_state_dict"])
    #     scheduler.load_state_dict(state_dict=ckpt["scheduler_state_dict"])
    #     print(f"loaded from {resume_checkpoint_path}")

    autograd_hacks.add_hooks(model)
    model.train(mode=True)

    ######## training ########
    train_inputs = torch.from_numpy(train_x.astype(np.float32))
    train_labels = torch.from_numpy(train_y.astype(np.long))
    optimizer.zero_grad()
    train_gradients = {k: {k:[] for k in range(num_classes)} for k in range(num_classes)}

    with torch.set_grad_enabled(mode=True):
        train_outputs = model(train_inputs)
        __, train_preds = torch.max(train_outputs, dim=1)
        train_loss = criterion(input=train_outputs, target=train_labels)
        train_loss.backward(retain_graph=True)
        autograd_hacks.compute_grad1(model)

        # optimizer.step()

        idx_to_grad = utils_grad.get_idx_to_grad(model, global_normalize=True)
        for idx, gradient in idx_to_grad.items():
            gt_label = int(train_labels[idx])
            given_label = gt_label
            train_gradients[gt_label][given_label].append(gradient)
        
        autograd_hacks.clear_backprops(model)
        # print('\n', idx_to_grad[1][-5:])

    ######## ul ########
    ul_inputs = torch.from_numpy(ul_x.astype(np.float32))
    ul_labels = torch.from_numpy(ul_y.astype(np.long))
    optimizer.zero_grad()
    ul_gradients = {k: {k:[] for k in range(num_classes)} for k in range(num_classes)}

    for given_label in range(num_classes):

        # Forward and backpropagation.
        with torch.set_grad_enabled(mode=True):
            ul_outputs = model(ul_inputs)
            __, ul_preds = torch.max(ul_outputs, dim=1)
            given_ul_labels = torch.from_numpy((np.zeros(ul_labels.shape) + given_label).astype(np.long))
            ul_loss = criterion(input=ul_outputs, target=given_ul_labels)
            ul_loss.backward(retain_graph=True)
            autograd_hacks.compute_grad1(model)

            idx_to_grad = utils_grad.get_idx_to_grad(model)
            for idx, gradient in idx_to_grad.items():
                gt_label = int(ul_labels[idx])
                ul_gradients[gt_label][given_label].append(gradient)

            # optimizer.step()
            autograd_hacks.clear_backprops(model)

    for gt_label in range(num_classes):
        gt_grad_list = train_gradients[gt_label][gt_label]
        for given_label in range(num_classes):
            candidate_grad_list = ul_gradients[gt_label][given_label]
            gt_agreement_list = utils_grad.get_agreement_list_all(gt_grad_list, gt_grad_list)
            agreement_list = utils_grad.get_agreement_list_all(gt_grad_list, candidate_grad_list)
            print(gt_label, given_label, mean(gt_agreement_list), mean(agreement_list))

        # ######## validation ########
        
        # minibatch_size = 128
        # num_minibatches_val = int(test_x.shape[0] / minibatch_size)
        # model.train(mode=False)
        # val_running_loss, val_running_corrects = 0.0, 0

        # for minibatch_num in range(num_minibatches_val):
            
        #     start_idx = minibatch_num * minibatch_size
        #     end_idx = start_idx + minibatch_size
        #     val_inputs = torch.from_numpy(test_x[start_idx:end_idx].astype(np.float32))
        #     val_labels = torch.from_numpy(test_y[start_idx:end_idx].astype(np.long))

        #     # Feed forward.
        #     with torch.set_grad_enabled(mode=False):
        #         val_outputs = model(val_inputs)
        #         _, val_preds = torch.max(val_outputs, dim=1)
        #         val_loss = criterion(input=val_outputs, target=val_labels)
        #     val_running_loss += val_loss.item() * val_inputs.size(0)
        #     val_running_corrects += int(torch.sum(val_preds == val_labels.data, dtype=torch.double))

        # val_loss = val_running_loss / (num_minibatches_val * minibatch_size)
        # val_acc = val_running_corrects / (num_minibatches_val * minibatch_size)

        # print(f"{val_loss:.3f},{val_acc:.3f}\n")

    gc.collect()

def train_mlp_multiple(  
                        train_txt_path,
                        train_embedding_path,
                        test_txt_path,
                        test_embedding_path,
                        num_classes,
                        dataset_name,
                        exp_id,
                        train_subset,
                        resume_checkpoint_path,
                        train_minibatch_size,
                        ul_minibatch_size,
                        num_seeds = 10,
                        num_epochs = 10,
                        criterion = nn.CrossEntropyLoss(),
                        ):

    for seed_num in range(num_seeds):

        train_mlp_checkpoint(  
            train_txt_path,
            train_embedding_path,
            test_txt_path,
            test_embedding_path,
            num_classes,
            train_subset,
            resume_checkpoint_path,
            num_epochs,
            train_minibatch_size,
            ul_minibatch_size,
            seed_num,
            criterion,
            checkpoint_folder = Path(f"checkpoints/{dataset_name}_{exp_id}")
            )
