import torch
import numpy as np

def get_selection(indices, dim):
    """
    Give selection to assign values to specific indices at given dimension.
    Enables dimension to be dynamic:
        tensor[get_selection(indices, dim=2)] = values
    Alternatively the dimension is fixed in code syntax:
        tensor[:, :, indices] = values
    """
    assert dim >= 0, "Negative dimension not supported."
    # Behaviour with python lists is unfortunately not working the same.
    if isinstance(indices, list):
        indices = torch.tensor(indices)
    assert isinstance(indices, (torch.Tensor, np.ndarray))
    selection = [slice(None) for _ in range(dim + 1)]
    selection[dim] = indices
    return selection

def is_constant(value):
    return value.ndim == 0 or value.shape == torch.Size([1])

def assign_values_to_dim(tensor, values, indices, dim, inplace=True):
    """
    Inplace tensor operation that assigns values to corresponding indices
    at given dimension.
    """
    if dim < 0:
        dim = dim + len(tensor.shape)
    selection = get_selection(indices, dim)
    if not inplace:
        tensor = tensor.clone()
    tensor[selection] = values
    return tensor
