import utils
import torch
import logging
import numpy as np

def extract_gradients(network, loss_at_input, mask, gradient_bound):
    grad_dict = {}

    loss_at_input.backward(gradient=torch.tensor(mask))#[0,:])

    for name, param in network.named_parameters():
        gradient = param.grad.numpy()
        gradient_norm = np.linalg.norm(gradient[:], 2)
        gradient_clip = max(1.0, gradient_norm/gradient_bound)
        grad_dict[name] = gradient.copy() / gradient_clip # Copying necessary so that zeroing the grads doesn't remove all our work

    network.zero_grad()

    return grad_dict

def convert_tensor_list_to_numpy(tensor_list):
    for ii, tensor in enumerate(tensor_list):
        tensor_list[ii] = tensor.detach().numpy()
    return tensor_list

def compute_gradient(x, target, model, mask, lossfunction, privacy_settings):
    model_input = torch.tensor(x, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)

    network_output = model(model_input)
    gradient_bound = privacy_settings['norm_bound']

    # logging.info("HELLO")

    if target is None:
        network_loss = lossfunction(network_output)
    else:
        # logging.info(network_output.shape)
        # logging.info(target.shape)
        network_loss = lossfunction(network_output, target)#[:,0].long())

    gradient_dict = extract_gradients(model, network_loss, mask, gradient_bound)
    return gradient_dict

def aggregate_grads(private_grads, privacy_settings):
    accumulated_grads = {}
    n_grads = len(private_grads)

    k_anonymity_threshold = privacy_settings['K']
    if n_grads > k_anonymity_threshold:

        for param in private_grads[0]:
            accumulated_grads[param] = private_grads[0][param]# / n_grads
        
        if n_grads>1:
            for grad in private_grads[1:]:
                for param in accumulated_grads:
                    accumulated_grads[param] += grad[param]# / n_grads

        noisy_gradient = apply_noise(accumulated_grads, privacy_settings)


        return noisy_gradient
    else:
        return None

def apply_noise(gradient, privacy_settings):
    noise_type = privacy_settings['method']
    if noise_type=='laplacian':
        epsilon=privacy_settings['epsilon']
        for layer in gradient:
            gradient[layer] += np.random.laplace(scale=1/epsilon, size=gradient[layer].shape)
    return gradient


def compute_private_gradients(input_data, model, model_name, lossfunction, privacy_settings):
    if model_name in input_data:
        private_gradients = [ compute_gradient(datapoint['x'], datapoint['target'], model, datapoint['mask'], lossfunction, privacy_settings)
                              for datapoint
                              in input_data[model_name] ]
        aggregated_gradients = aggregate_grads(private_gradients, privacy_settings)

        return aggregated_gradients
    else:
        return None
