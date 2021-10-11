import json
import logging

import torch
import onnx
import base64
import numpy as np

from . import FeedForwardFromONNX


class PayloadEncoder(json.JSONEncoder):

    def default(self, obj):
        # Deal with numpy arrays
        if isinstance(obj, np.ndarray):
            if obj.flags['C_CONTIGUOUS']:
                obj_data = obj.data
            else:
                cont_obj = np.ascontiguousarray(obj)
                assert (cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data
            data_json = obj_data.tolist()
            return dict(__ndarray__=data_json,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Encoding for onnx models, which are by default bytestreams
        # JSON wants Unicode so we have to convert to latin1
        if isinstance(obj, bytes):
            return dict(encoded_str=obj.decode('latin1'),
                        encoding='latin1')
        # Let the base class default method raise the TypeError
        super(PayloadEncoder, self).default(obj)


def helper_payload_hook(dct):
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = np.array(dct['__ndarray__'], dct['dtype']).reshape(dct['shape'])
        return data
    if isinstance(dct, dict) and 'encoded_str' in dct:
        return base64.b64decode(dct['encoded_str'])
    return dct


def gradient_list_to_json(gradient_dict):
    # gradient_list_numpy = {name: grad.detach().cpu().numpy() for name, grad in gradient_dict.items()}
    gradient_list = [{name: grad.detach().cpu().numpy() for name, grad in grad_dict.items()} for grad_dict in
                     gradient_dict]
    j = json.dumps(gradient_list, cls=PayloadEncoder)
    return j


def create_return_payload(gradients, helpername):
    outputlist = [dict(model_tag=tag, model_noisy_gradients=grads) for tag, grads in gradients.items()]
    return_payload = {
        "origin": helpername,
        "model_gradient_set": outputlist
    }
    json_return_payload = json.dumps(return_payload, cls=PayloadEncoder)
    return json_return_payload

def create_eval_return_payload(losses, helpername):
    outputlist = [dict(model_tag=tag, model_loss=loss) for tag, loss in losses.items()]
    return_payload = {
        "origin": helpername,
        "model_evaluation_result": outputlist
    }
    json_return_payload = json.dumps(return_payload, cls=PayloadEncoder)
    return json_return_payload


def parse_singleton_loss(loss_dict):
    loss_name = loss_dict['loss_name']
    loss_kwargs = loss_dict['loss_kwargs']
    if 'reduce' in loss_kwargs:
        del loss_kwargs['reduce']
    loss_kwargs['reduction'] = 'none'
    if loss_name == "BCELoss":
        loss = torch.nn.BCELoss(**loss_kwargs)
    elif loss_name == "MSELoss":
        loss = torch.nn.MSELoss(**loss_kwargs)
    else:
        raise ValueError("Your loss, " + str(loss_name) + " is not supported. " +
                         "Enter one of BCELoss or " +
                         "MSELoss.")
    return loss

def get_loss(loss_dict):
    # return torch.nn.BCEWithLogitsLoss(reduction='none')
    if loss_dict['loss_type'] == 'singleton':
        loss = parse_singleton_loss(loss_dict)
        return loss
    elif loss_dict['loss_type'] == 'composite':
        component_losses = loss_dict['loss_elements']
        loss_list = []
        for loss in component_losses:
            loss_list.append(
                dict(
                    loss=parse_singleton_loss(loss), weight=loss['loss_weight'], target=loss['target_name'], index=loss['output_index']
                )
            )
        return loss_list
    else:
        raise ValueError("loss_type must be one of singleton or composite. Your loss is not valid.")


def get_onnx_model(model_string):
    try:
        return onnx.load_model_from_string(model_string)
    except Exception as e:
        raise AttributeError('The model string could not be loaded, with exception {}'.format(e))


def get_pytorch_model_from_onnxstring(model_string):
    onnx_model = get_onnx_model(model_string)
    pytorch_model = FeedForwardFromONNX.ConvertModel(onnx_model)
    return pytorch_model


def convert_vector_to_batchsize_format(input_vector):
    if np.ndim(input_vector) == 0:
        return torch.tensor(np.expand_dims(input_vector, [0, 1]))
    elif np.ndim(input_vector) == 1:
        return torch.tensor(np.expand_dims(input_vector, [1]))
    elif np.ndim(input_vector) == 2:
        if np.shape(input_vector)[0] == 1:
            return torch.tensor(np.transpose(input_vector))
        elif np.shape(input_vector[1] == 1):
            return torch.tensor(input_vector)
        else:
            raise ValueError('This function should only be used on vectors.')
    else:
        raise ValueError('This function should only be used on vectors.')


def convert_features_to_1d_vector(feature_vector):
    if np.ndim(feature_vector) == 0:
        return torch.tensor(np.expand_dims(feature_vector, [0, 1]))
    elif np.ndim(feature_vector) == 1:
        return torch.tensor(np.expand_dims(feature_vector, [0]))
    else:
        raise ValueError('This function should only be used on vectors.')


def match_n_features_to_n_targets(net_input, targets):
    n_targets = np.shape(targets)[0]
    if n_targets > 1:
        duplicated_features = np.repeat(net_input, n_targets, axis=0)
        return duplicated_features
    else:
        return net_input


def sanitize_netinputs(net_input, targets, masks):
    try:
        net_input = np.asarray(net_input).astype(np.float32)
        targets = np.asarray(targets).astype(np.float32)
        masks = np.asarray(masks).astype(np.float32)

        input_dtype = net_input.dtype
        input_shape = np.shape(np.squeeze(net_input))
        if len(input_shape) > 1:
            raise ValueError('Network inputs with more than one one spatial dimension are invalid.')
        target_shape = np.shape(np.squeeze(targets))
        if len(target_shape) > 1:
            raise ValueError('Targets should be given as a list or one-dimensional numpy array.')
        masks_shape = np.shape(np.squeeze(masks))
        if len(masks_shape) > 1:
            raise ValueError('Masks should be given as a list or one-dimensional numpy array.')

        if not masks_shape == target_shape:
            raise ValueError('Masks and targets must be the same shape.')

        net_input = convert_features_to_1d_vector(net_input)
        targets = convert_vector_to_batchsize_format(targets)
        masks = convert_vector_to_batchsize_format(masks)

        net_input = match_n_features_to_n_targets(net_input, targets)

        return net_input, targets, masks

    except Exception as e:
        raise e

def get_privacy_settings(privacy_json):
    privacy_settings = privacy_json
    return privacy_settings


def add_payload_data_to_dict(data, model_tag, payload_data):
    if model_tag in data:
        data[model_tag].append(payload_data)
    else:
        data[model_tag] = [payload_data]
    return None


def parse_adserver_json_payload(json_input):
    if not json_input.get('function', '') == 'gradient_computation':
        return None, None

    data = {}
    privacy_settings = get_privacy_settings(json_input['privacy_settings'])

    for payload in json_input['gradient_service_payload_set']:
        relevant_data_info = payload['gradient_service_payload']['payload']
        model_tags = relevant_data_info['model_tag']

        model_features, targets, masks = sanitize_netinputs(relevant_data_info['model_features'],
                                                            relevant_data_info['model_label'],
                                                            relevant_data_info['mask'])

        data_dict = {
            'target': targets,
            'x': model_features,
            'mask': masks
        }
        if isinstance(model_tags, list):
            for model_tag in model_tags:
                add_payload_data_to_dict(data, model_tag, data_dict)
        else:
            add_payload_data_to_dict(data, model_tags, data_dict)

    model_set = {}
    for model in json_input['gradient_model_set']:
        pytorch_model = get_pytorch_model_from_onnxstring(model['model'])
        model_loss = get_loss(model['model_loss_function'])

        model_set[model['model_tag']] = dict(
            model=pytorch_model, loss=model_loss
        )
    return data, model_set, privacy_settings


def check_and_convert_input(json_dump, helpername):
    try:
        json_dump = json_dump.get_json()
    except Exception as E:
        logging.info(E)
    try:
        if not ((json_dump.get('function', '') == 'gradient_computation') or (json_dump.get('function', 
        '') == 'evaluation')) and not json_dump.get('origin', '') == helpername:
            logging.info("Invalid request made.")
            return None
        if not json_dump.get('origin') == helpername:
            logging.info("Origin and helper name mismatch.")
            return None
        json_string = json.dumps(json_dump)
        json_input = json.loads(json_string, object_hook=helper_payload_hook)
        return json_input
    except Exception as e:
        logging.info("Exception raised: {}".format(e))
        return None

def check_request_body_size(req):
    # body_size_limitation = int(float(os.environ["dataSize_threshold"]))
    # body_size = len(req._HttpRequest__body_bytes)
    # if(body_size > body_size_limitation):
        # return "The content size is " + str(body_size) + ", which is over the threshold " + str(body_size_limitation)
    return ''