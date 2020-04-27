import argparse
import numpy as np
def str2bool(v):
    """
    Convert string to boolean
    :param v:
    :return:
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_binary_mask(x):
    """
    Return binary mask from numpy array
    :param x: numpy array
    :return: binary mask from numpy array
    """
    x[x >= 0.5] = 1.
    x[x < 0.5] = 0.
    return x