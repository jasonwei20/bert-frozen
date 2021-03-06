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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import gc

import utils_autograd_hacks as autograd_hacks
import utils_common
import utils_grad
import utils_processing
import utils_mlp_helper

class LR(nn.Module):

    def __init__(self, num_classes):
        super(LR, self).__init__()
        self.fc1 = nn.Linear(768, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        output = torch.sigmoid(x)
        # output = torch.softmax(x, dim=1)
        return output

class MLP(nn.Module):

    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(768, 50)
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        output = torch.sigmoid(x)
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
                        seed_num,
                        criterion,
                        checkpoint_folder,
                        train_minibatch_size = 5,
                        val_minibatch_size = 256,
                        ):

    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    
    train_x, train_y = utils_processing.get_x_y(train_txt_path, train_embedding_path)
    # test_x, test_y = utils_processing.get_x_y(test_txt_path, test_embedding_path)
    if train_subset:
        train_x, ul_x, train_y, ul_y = train_test_split(train_x, train_y, train_size=train_subset, random_state=seed_num, stratify=train_y)
        test_x = ul_x; test_y = ul_y

    # print(train_x.shape, train_y.shape, ul_x.shape, ul_y.shape, test_x.shape, test_y.shape)

    model = MLP(num_classes=num_classes)
    optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.05) #wow, works for even large learning rates
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

    autograd_hacks.add_hooks(model)

    train_inputs = torch.from_numpy(train_x.astype(np.float32))
    train_labels = torch.from_numpy(train_y.astype(np.long))
    optimizer.zero_grad()

    ######## first train the model ########
    num_minibatches_train = int(train_x.shape[0] / train_minibatch_size)

    for epoch in range(1, num_epochs + 1):

        ######## training ########
        model.train(mode=True)

        train_x, train_y = shuffle(train_x, train_y, random_state = seed_num)

        for minibatch_num in range(num_minibatches_train):
            
            start_idx = minibatch_num * train_minibatch_size
            end_idx = start_idx + train_minibatch_size
            train_inputs_mb = torch.from_numpy(train_x[start_idx:end_idx].astype(np.float32))
            train_labels_mb = torch.from_numpy(train_y[start_idx:end_idx].astype(np.long))
            optimizer.zero_grad()

            # Forward and backpropagation.
            with torch.set_grad_enabled(mode=True):
                train_outputs = model(train_inputs_mb)
                __, train_preds = torch.max(train_outputs, dim=1)
                train_loss = criterion(input=train_outputs, target=train_labels_mb)
                train_loss.backward(retain_graph=True)
                autograd_hacks.compute_grad1(model)

                optimizer.step()
                autograd_hacks.clear_backprops(model)

    ######## validation ########
    model.train(mode=False)

    val_inputs = torch.from_numpy(test_x.astype(np.float32))
    val_labels = torch.from_numpy(test_y.astype(np.long))
    val_acc_list = []; conf_acc_list = []

    # Feed forward.
    with torch.set_grad_enabled(mode=False):
        val_outputs = model(val_inputs)
        val_confs, val_preds = torch.max(val_outputs, dim=1)
        val_loss = criterion(input=val_outputs, target=val_labels)
        val_loss_print = val_loss / val_inputs.shape[0]
        val_acc = accuracy_score(test_y, val_preds)
        val_acc_list.append(val_acc)

        val_preds = val_preds.tolist()
        val_confs = val_confs.numpy().tolist()
        val_threshold_idx = int(len(val_confs) / 10)
        val_conf_threshold = list(sorted(val_confs))[-val_threshold_idx]

        confident_predicted_labels = [val_preds[i] for i in range(len(val_confs)) if val_confs[i] >= val_conf_threshold]
        confident_ul_y = [ul_y[i] for i in range(len(val_confs)) if val_confs[i] >= val_conf_threshold]
        conf_acc = accuracy_score(confident_ul_y, confident_predicted_labels)
        conf_acc_list.append(conf_acc)

        # print(f"trained model has val acc {val_acc:.4f}, {conf_acc:.4f}")

    print(val_outputs.shape)

    ######## get train gradients ########
    model.train(mode=True)
    train_label_to_grads = {label: [] for label in range(num_classes)}

    with torch.set_grad_enabled(mode=True):
        train_outputs = model(train_inputs)
        __, train_preds = torch.max(train_outputs, dim=1)
        train_loss = criterion(input=train_outputs, target=train_labels)
        train_loss.backward(retain_graph=True)
        autograd_hacks.compute_grad1(model)

        train_grad_np = utils_grad.get_grad_np(model, global_normalize=True)
        for i in range(train_grad_np.shape[0]):
            train_grad = train_grad_np[i]
            label = int(train_labels[i])
            train_label_to_grads[label].append(train_grad)

        # optimizer.step()
        autograd_hacks.clear_backprops(model)

    ######## get unlabeled gradients ########
    ul_inputs = torch.from_numpy(ul_x.astype(np.float32))
    ul_labels = torch.from_numpy(ul_y.astype(np.long))
    optimizer.zero_grad()
    ul_grad_np_dict = {}

    for given_label in range(num_classes):

        with torch.set_grad_enabled(mode=True):
            ul_outputs = model(ul_inputs)
            __, ul_preds = torch.max(ul_outputs, dim=1)
            given_ul_labels = torch.from_numpy((np.zeros(ul_labels.shape) + given_label).astype(np.long))
            ul_loss = criterion(input=ul_outputs, target=given_ul_labels)
            ul_loss.backward(retain_graph=True)
            autograd_hacks.compute_grad1(model)

            grad_np = utils_grad.get_grad_np(model, global_normalize=True)
            ul_grad_np_dict[given_label] = grad_np

            # optimizer.step()
            autograd_hacks.clear_backprops(model)
    
    ### for a given unlabeled example,
    ### try both labels and see which gradient produced is closer to an existing gradient
    def get_grad_comparison(given_label_to_grad, train_label_to_grads):
        label_to_max_sim = {}
        for label, train_grads in train_label_to_grads.items():
            grad_from_given_label = given_label_to_grad[label]
            sim_list = [np.dot(grad_from_given_label, train_grad) for train_grad in train_grads]
            sim_list_sorted = list(sorted(sim_list))
            max_sim = mean(sim_list_sorted)
            label_to_max_sim[label] = max_sim
        signed_sim_diff = label_to_max_sim[1] - label_to_max_sim[0]
        sorted_label_to_max_sim = list(sorted(label_to_max_sim.items(), key=lambda x: x[1]))
        label, max_sim = sorted_label_to_max_sim[-1]
        sim_diff = sorted_label_to_max_sim[-1][-1] - sorted_label_to_max_sim[0][-1]
        return label, max_sim, sim_diff, signed_sim_diff

    predicted_labels = []
    sim_diff_list = []

    val_conf_i_list = []; signed_sim_diff_list = []

    for i in range(ul_inputs.shape[0]):
        given_label_to_grad = {k: v[i] for k, v in ul_grad_np_dict.items()}
        predicted_label, max_sim, sim_diff, signed_sim_diff = get_grad_comparison(given_label_to_grad, train_label_to_grads)
        predicted_labels.append(predicted_label)
        sim_diff_list.append(sim_diff)
    
        val_conf_i_list.append(float(val_outputs[i, 0]))
        signed_sim_diff_list.append(signed_sim_diff)

    utils_common.plot_jasons_scatterplot(val_conf_i_list, signed_sim_diff_list, 'plots/conf_and_grad_sim_500.png', 'Confidence', 'Grad Sim - Mean', 'Confidence and Grad Sim - Mean')

    predicted_labels = np.asarray(predicted_labels)
    acc = accuracy_score(ul_y, predicted_labels)
    sim_diff_threshold_idx = int(len(sim_diff_list) / 10)
    sim_diff_threshold = list(sorted(sim_diff_list))[-sim_diff_threshold_idx]

    confident_predicted_labels = [predicted_labels[i] for i in range(len(sim_diff_list)) if sim_diff_list[i] >= sim_diff_threshold]
    confident_ul_y = [ul_y[i] for i in range(len(sim_diff_list)) if sim_diff_list[i] >= sim_diff_threshold]
    conf_acc = accuracy_score(confident_ul_y, confident_predicted_labels)

    gc.collect()
    return acc, conf_acc, mean(val_acc_list[-5:]), mean(conf_acc_list[-5:])

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
                        num_seeds = 10,
                        num_epochs = 10,
                        criterion = nn.CrossEntropyLoss(),
                        ):

    acc_list = []
    conf_acc_list = []
    mlp_acc_list = []
    mlp_conf_acc_list = []

    for seed_num in range(num_seeds):

        acc, conf_acc, mlp_acc, mlp_conf_acc = train_mlp_checkpoint(  
            train_txt_path,
            train_embedding_path,
            test_txt_path,
            test_embedding_path,
            num_classes,
            train_subset,
            resume_checkpoint_path,
            num_epochs,
            seed_num,
            criterion,
            checkpoint_folder = Path(f"checkpoints/{dataset_name}_{exp_id}")
            )
        acc_list.append(acc)
        conf_acc_list.append(conf_acc)
        mlp_acc_list.append(mlp_acc)
        mlp_conf_acc_list.append(mlp_conf_acc)
    
    return mean(acc_list), stdev(acc_list), mean(conf_acc_list), stdev(conf_acc_list), mean(mlp_acc_list), stdev(mlp_acc_list), mean(mlp_conf_acc_list), stdev(mlp_conf_acc_list)
