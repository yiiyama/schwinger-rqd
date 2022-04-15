import numpy as np

def global_cost_function(prob_dist):
    return 1. - prob_dist[0]

def local_cost_function(prob_dist):
    prob_dist = np.asarray(prob_dist)
    num_qubits = int(np.log2(prob_dist.shape[0]))

    indices = np.arange(prob_dist.shape[0])
    bit_masks = np.array([1 << i for i in range(num_qubits)])
    bit_masks = np.tile(np.expand_dims(bit_masks, axis=-1), (1, prob_dist.shape[0]))
    num_ones = np.sum((indices & bit_masks).T >> np.arange(num_qubits), axis=1)
    num_zeros = num_qubits - num_ones
    
    return 1. - np.sum(prob_dist * num_zeros) / num_qubits

def make_general_cost_function(q):
    return lambda p: q * global_cost_function(p) + (1. - q) * local_cost_function(p)
