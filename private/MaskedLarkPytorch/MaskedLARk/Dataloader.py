import json

import requests
import torch
import numpy as np
from torch.utils.data import Dataset

from . import payloadutils

class MLarkDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_queue = []
        self.task = 'binary_classifier'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data_target_pair = self.dataset[item]
        self.data_queue.append(data_target_pair)
        return data_target_pair[0] # return the data, not the target

    def set_diff_privacy(self, mechanism='laplacian', epsilon=1, norm_bound=1):
        self.dp_settings = dict(
            mechanism=mechanism,
            epsilon=epsilon,
            gradient_bound=norm_bound
        )

    def set_aggregation_privacy(self, mechanism='standard', threshold=10):
        self.agg_settings = dict(
            mechanism=mechanism,
            K=threshold
        )

    def get_privacy_settings(self):
        privacy_settings = dict(
            method=self.dp_settings['mechanism'],
            epsilon=self.dp_settings['epsilon'],
            norm_bound=self.dp_settings['gradient_bound'],
            K=self.agg_settings['K']
        )
        return privacy_settings

    def set_model_name(self, model_name):
        self.model_name = model_name

    def set_loss_fn(self, lossfn, loss_kwargs):
        self.loss_name = lossfn
        self.loss_kwargs = loss_kwargs

    def set_helper_names(self, helper_names):
        self.helper_names = helper_names

    def set_endpoints(self, endpoints):
        self.endpoints = endpoints

    def get_encryption_standard(self):
        return "cleartext"

    def get_model_settings(self):
        return self.model_name, self.loss_name, self.loss_kwargs

    def generate_masks(self, sum_to_one=True):
        a0 = np.random.uniform()
        if sum_to_one:
            a1 = 1.0-a0
        else:
            a1 = -a0
        return a0, a1

    def create_datapoint_payload(self, input_point, targets, mask, model_tag):
        if torch.is_tensor(input_point):
            inputs = input_point.squeeze().detach().cpu().numpy().tolist()#.astype(int).tolist()
        else:
            inputs = input_point.tolist()

        if torch.is_tensor(targets):
            targets = targets.squeeze().detach().cpu().numpy().tolist()#.astype(int).tolist()
        else:
            targets = targets

        payload = dict(
            aggregation_service_payload=dict(
                encryption_standard=self.get_encryption_standard(),
                payload=dict(
                    model_features=inputs,
                    model_label=targets,
                    mask=mask,
                    model_tag=model_tag
                )
            )
        )
        return payload

    def create_data_payload(self, inputs, targets, masks, model_tag, helpers):

        data_payloads = [self.create_datapoint_payload(input, target, mask, model_tag) for
                        input, target, mask in zip(inputs, targets, masks)]
        data_dict = {}
        for helper in helpers:
            data_dict[helper] = [x for helper_name, x in zip(helpers, data_payloads) if helper_name==helper]

        return data_dict

    def get_helper_names(self):
        # Make them declare entire string
        return self.helper_names

    def create_private_pseudodata(self, inputs, targets, helper_list):
        if self.task=='binary_classifier':
            inputs = np.repeat(inputs, 4, axis=0)
            masks = []
            helpers = []
            output_targets = [0.0, 1.0]*int(inputs.shape[0]/2)
            for ii, target in enumerate(targets):
                a0, a1 = self.generate_masks(sum_to_one=False)
                b0, b1 = self.generate_masks(sum_to_one=True)
                if target==1.0:
                    masks.extend([a0, b0, a1, b1])
                else:
                    masks.extend([b0, a0, b1, a1])
                if len(helper_list)==2:
                    helpers.extend([helper_list[0], helper_list[0], helper_list[1], helper_list[1]])
                else:
                    helpers.extend([helper_list[0]]*4)
            return inputs, output_targets, masks, helpers

    def get_data_payload(self, helper_list, model_name):
        inputs = np.concatenate([x[0][np.newaxis,:] for x in self.data_queue], axis=0)
        targets = np.asarray([x[1] for x in self.data_queue])
        
        inputs, targets, masks, helpers = self.create_private_pseudodata(inputs, targets, helper_list)

        n_inputs = len(masks)
        inputs = [inputs[ii, :] for ii in range(n_inputs)]
        data_payload = self.create_data_payload(inputs, targets, masks, model_name, helpers)
        
        return data_payload
    
    def clear_dataqueue(self):
        self.data_queue = []