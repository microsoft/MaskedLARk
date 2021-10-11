import json
import numpy as np
import onnx

from . import custom_torch_onnx_utils

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
        # Let the base class default method raise other TypeErrors
        super(PayloadEncoder, self).default(obj)


def helper_payload_hook(dct):
    # Reconstitute numpy arrays
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = np.array(dct['__ndarray__'], dct['dtype']).reshape(dct['shape'])
        return data
    # Convert the latin1 strings to onnx bytestreams
    if isinstance(dct, dict) and 'encoded_str' in dct:
        return bytes(dct['encoded_str'], dct['encoding'])
    return dct


def create_payload(net_input, target, mask_val, model_name, encryption_standard):
    payload = dict(
        encryption_standard=encryption_standard,
        payload=dict(
            model_features=net_input,
            model_label=target,
            model_mask=mask_val,
            model_tag=model_name
        )
    )
    return payload

def pytorch_to_onnx(pytorch_model, input):
    return custom_torch_onnx_utils._export_to_protobuf(pytorch_model, input)


def create_model_payload(onnx_model, model_name, loss_function, loss_kwargs):
    model_dict = dict(
        model_tag=model_name,
        model=onnx._serialize(onnx_model),
        model_loss_function=dict(
            loss_name=loss_function,
            loss_kwargs=loss_kwargs
        )
    )
    return model_dict
