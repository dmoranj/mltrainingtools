import numpy as np


def generate_metaparameters(number):
    base_alpha = 1
    alpha_range = 5

    base_M = 5
    range_M = 50

    base_l1 = 0.1
    l1_range = 1.9

    base_l2 = 0.1
    l2_range = 1.9

    base_coders = 200
    range_coders = 800

    alpha = list(np.power(10, -(alpha_range*np.random.rand(number) + base_alpha)))
    l1 = list(np.power(10, -(l1_range*np.random.rand(number) + base_l1)))
    l2 = list(np.power(10, -(l2_range*np.random.rand(number) + base_l2)))
    M = list(np.floor(base_M + range_M*np.random.rand(number)).astype(int))
    h_encoder = list(np.floor(base_coders + range_coders*np.random.rand(number)).astype(int))
    h_decoder = list(np.floor(base_coders + range_coders*np.random.rand(number)).astype(int))

    return [{
            "alpha": alpha[i],
            "M": M[i],
            "L2": l2[i],
            "L1": l1[i],
            "H_encoder": h_encoder[i],
            "H_decoder": h_decoder[i]}
            for i in range(number)]

