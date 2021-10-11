from . import utils
import torch
import logging
import numpy as np

def convert_tensor_list_to_numpy(tensor_list):
    for ii, tensor in enumerate(tensor_list):
        tensor_list[ii] = tensor.detach().numpy()
    return tensor_list

def compute_loss(x, target, model, lossfunction, privacy_settings):
    model_input = torch.tensor(x, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)

    network_output = model(model_input)

    if target is None:
        network_loss = lossfunction(network_output)
    else:
        network_loss = lossfunction(network_output, target)

    noisy_loss = apply_noise(network_loss, privacy_settings)
    return noisy_loss

def aggregate_losses(private_losses, privacy_settings):
    accumulated_losses = {}
    n_grads = len(private_losses)

    k_anonymity_threshold = privacy_settings['k-anonymity']['threshold']
    if n_grads > k_anonymity_threshold:
        accumulated_losses = sum(private_losses)
        return accumulated_losses
    else:
        return None

def apply_noise(loss, privacy_settings):
    noise_type = privacy_settings['method']
    if noise_type=='laplacian':
        epsilon=privacy_settings['epsilon']
        loss += np.random.laplace(scale=epsilon, size=loss.shape)
    return loss


def compute_private_losses(input_data, model, model_name, lossfunction, privacy_settings):
    if model_name in input_data:
        private_gradients = [ compute_loss(datapoint['x'], datapoint['target'], model, lossfunction, privacy_settings)
                              for datapoint
                              in input_data[model_name] ]
        aggregated_gradients = aggregate_losses(private_gradients, privacy_settings)

        return aggregated_gradients
    else:
        return None
