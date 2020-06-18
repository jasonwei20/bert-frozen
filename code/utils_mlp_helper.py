import numpy as np
import random

def get_flip_ratios(sorted_d, flipped_indexes, n=10):
    top_group = [tup[0] for tup in sorted_d[:n]]
    bottom_group = [tup[0] for tup in sorted_d[-n:]]
    top_group_noise_ratio = len(list(set(top_group) & set(flipped_indexes)))
    bottom_group_noise_ratio = len(list(set(bottom_group) & set(flipped_indexes)))
    return top_group_noise_ratio, bottom_group_noise_ratio

def get_idx_to_gt(train_y_orig, start_idx, minibatch_size):
    idx_to_gt = {}
    train_y_minibatch = train_y_orig[start_idx:start_idx+minibatch_size]
    for minibatch_idx in range(train_y_minibatch.shape[0]):
        idx_to_gt[minibatch_idx] = train_y_minibatch[minibatch_idx]
    return idx_to_gt

def get_flipped_indexes(train_y, train_y_orig, minibatch_num, minibatch_size):
    
    start_idx = minibatch_num * minibatch_size
    end_idx = start_idx + minibatch_size

    train_y_batch = train_y[start_idx:end_idx]
    train_y_orig_batch = train_y_orig[start_idx:end_idx]

    flipped_indexes = set()
    for i in range(train_y_batch.shape[0]):
        if train_y_batch[i] != train_y_orig_batch[i]:
            flipped_indexes.add(i)
    
    return flipped_indexes

def get_labels_uniform_flip(train_y, flip_ratio, num_classes):

    train_y_orig = train_y
    train_y_noisy = np.zeros(train_y.shape)

    for i in range(train_y_orig.shape[0]):
        random_num = random.random()
        if random_num >= flip_ratio:
            train_y_noisy[i] = train_y_orig[i]
        else:
            new_label = random.randint(0, num_classes-1)
            while new_label == train_y_orig[i]:
                new_label = random.randint(0, num_classes-1)
            train_y_noisy[i] = new_label
    
    return train_y_noisy, train_y_orig
        