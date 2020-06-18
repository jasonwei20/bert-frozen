import numpy as np
import random

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
        