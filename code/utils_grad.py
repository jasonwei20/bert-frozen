import utils_autograd_hacks as autograd_hacks
import numpy as np

def module_with_params(module):
    return module.__class__.__name__ in ['Conv2d', 'BatchNorm2d', 'Linear']

def get_per_example_grad_as_tuple(model):

    per_example_grad_list = []

    for i, layer in enumerate(model.modules()):
        if module_with_params(layer) and autograd_hacks.is_supported(layer):
                for param in layer.parameters():
                    per_example_grad = param.grad1
                    per_example_grad_list.append(per_example_grad)

    return tuple(per_example_grad_list)

def per_example_grad_tuple_to_idx_to_grad_flatten(per_example_grad_tuple):

    idx_to_grad_flatten = {}
    train_batch_size = per_example_grad_tuple[0].size()[0]
    
    for per_example_grad in per_example_grad_tuple:
        for idx in range(train_batch_size):
            flattened_per_example_grad_layer = per_example_grad[idx].detach().cpu().numpy().flatten()
            if idx not in idx_to_grad_flatten:
                idx_to_grad_flatten[idx] = flattened_per_example_grad_layer
            else:
                idx_to_grad_flatten[idx] = np.concatenate((idx_to_grad_flatten[idx], flattened_per_example_grad_layer), axis=None)

    return idx_to_grad_flatten

def get_idx_to_grad(model, global_normalize=True):

    per_example_grad_tuple = get_per_example_grad_as_tuple(model)
    idx_to_grad_flatten = per_example_grad_tuple_to_idx_to_grad_flatten(per_example_grad_tuple)

    if global_normalize:
        for idx, grad_flatten in idx_to_grad_flatten.items():
            idx_to_grad_flatten[idx] = grad_flatten / np.linalg.norm(grad_flatten)

    return idx_to_grad_flatten


######################################################
# Weighting and ranking

def get_agreement(arr_1, arr_2):
    return np.dot(arr_1, arr_2)

def get_agreements_dict(idx_to_grad_flatten):

    idx_ij_to_agreement = {}
    keys = list(sorted(idx_to_grad_flatten.keys()))
    
    for i in keys:
        for j in keys:
            if j > i:
                grad_flatten_i = idx_to_grad_flatten[i]
                grad_flatten_j = idx_to_grad_flatten[j]
                agreement_i_j = get_agreement(grad_flatten_i, grad_flatten_j)
                idx_ij_to_agreement[f"{i};{j}"] = agreement_i_j
    
    return idx_ij_to_agreement

def get_softmax_single_value(x, annealling_factor):
    return np.exp(x * annealling_factor)

def get_softmax_denominator(d, annealling_factor):
    agreements_np = np.array([v for v in d.values()])
    agreements_log = np.exp(agreements_np * annealling_factor)
    softmax_denominator = np.sum(agreements_log)
    return softmax_denominator

def apply_softmax_to_dict(d, annealling_factor):
    softmax_denominator = get_softmax_denominator(d, annealling_factor)
    d_softmax = {}
    for k, v in d.items():
        d_softmax[k] = get_softmax_single_value(v, annealling_factor) / softmax_denominator
    return d_softmax

def get_relevant_keys(d, idx, within_class, idx_to_gt):
    relevant_keys = []
    for k in d.keys():
        items = k.split(';')
        idx_1 = int(items[0])
        idx_2 = int(items[1])
        if idx_1 == idx or idx_2 == idx:
            if within_class:
                if idx_to_gt[idx_1] == idx_to_gt[idx_2]:
                    relevant_keys.append(k)
            else:
                relevant_keys.append(k)
    return relevant_keys

def get_idx_agreement_sum(d, idx, top_k, within_class, idx_to_gt):
    relevant_keys = get_relevant_keys(d, idx, within_class, idx_to_gt)
    relevant_values = [v for k, v in d.items() if k in relevant_keys]
    idx_agreement_sum = None
    if top_k:
        relevant_values_sorted = list(reversed(sorted(relevant_values)))
        top_k_relevant = relevant_values_sorted[:top_k]
        idx_agreement_sum = sum(top_k_relevant)
    else:    
        idx_agreement_sum = sum(relevant_values)
    return idx_agreement_sum

def get_idx_to_weight(idx_to_grad_flatten, annealling_factor, idx_to_gt, inverse = False, top_k = None, within_class = False):

    def normalize_weights(idx_to_weight):
        weight_sum = sum(idx_to_weight.values())
        idx_to_weight_normalized = {}
        for idx, weight in idx_to_weight.items():
            idx_to_weight_normalized[idx] = weight / (weight_sum + 1e-7)
        return idx_to_weight_normalized

    if len(idx_to_grad_flatten) == 1:
        return {k: 1.0 for k in idx_to_grad_flatten.keys()}
    idx_ij_to_agreement = get_agreements_dict(idx_to_grad_flatten)
    if inverse:
        idx_ij_to_agreement = {k: -v for k, v in idx_ij_to_agreement.items()}
    idx_ij_to_agreement_softmax = apply_softmax_to_dict(idx_ij_to_agreement, annealling_factor)
    pos_same_class_count, neg_same_class_count = 0, 0
    pos_different_class_count, neg_different_class_count = 0, 0
    for k, v in idx_ij_to_agreement.items():
        parts = k.split(';')
        idx_1 = int(parts[0])
        idx_2 = int(parts[1])
        if idx_to_gt[idx_1] == idx_to_gt[idx_2]:
            if v > 0:
                pos_same_class_count += 1
            else:
                neg_same_class_count += 1
        else:
            if v > 0:
                pos_different_class_count += 1
            else:
                neg_different_class_count += 1
    # print(pos_same_class_count, neg_same_class_count, pos_different_class_count, neg_different_class_count)
    idx_to_weight = {}
    for idx in idx_to_grad_flatten.keys():
        idx_agreement_sum = get_idx_agreement_sum(idx_ij_to_agreement_softmax, idx, top_k, within_class, idx_to_gt)
        idx_to_weight[idx] = idx_agreement_sum
    
    idx_to_weight_normalized = normalize_weights(idx_to_weight)

    return idx_to_weight_normalized

######################################################
# update model using the weighted gradient

def get_weighted_layer_grad(idx_to_weight_batch, per_example_grad):

    indices = list(idx_to_weight_batch.keys())
    initial_idx = indices[0]
    weighted_grad = per_example_grad[initial_idx] * idx_to_weight_batch[initial_idx]
    for idx in indices[1:]:
        weight = idx_to_weight_batch[idx]
        weighted_per_example_grad = per_example_grad[idx] * weight
        weighted_grad = weighted_grad.add(weighted_per_example_grad)
    return weighted_grad