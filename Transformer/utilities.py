import torch 
import torch.nn as nn 
import torch.nn.functional as F 

def mask_(matrices, maskval = 0.0, mask_diagonal = True):

    """
    Masks out all values in the given batch of matrices where i < = j
    """

    b, h, w = matrices.size()

    indices = torch.triu_indices(h, w, offset = 0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval 


def device_(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'