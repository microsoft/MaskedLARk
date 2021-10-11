import requests
import torch
import numpy as np

from . import payloadutils

class Helper():
    def __init__(self, datasource, task='binary_classifier'):
        self.dp_settings = None
        self.agg_settings = None
        self.task = task
        self.datasource = datasource

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
            inputs = input_point

        if torch.is_tensor(targets):
            targets = targets.squeeze().detach().cpu().numpy().tolist()#.astype(int).tolist()

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


    def create_model_payload(self, model):
        model_tag, lossfn, lossfn_kwargs = self.get_model_settings()

        payload = [
            dict(
                model_tag=model_tag,
                model=dict(
                    encoded_str=model.decode('latin1'),
                    encoding='latin1'
                ),
                model_loss_function=dict(
                    loss_type="singleton",
                    loss_name=lossfn,
                    loss_kwargs=lossfn_kwargs
                )
            )
        ]
        return payload

    def get_helper_names(self):
        # Make them declare entire string
        return self.helper_names

    def parse_return_json(self, json_data):
        gradients = []
        for gradient in json_data['aggregation_model_set']:
            grad = {}
            for key, val in gradient['model_noisy_gradients'].items():
                grad[key] = torch.tensor(np.array(val['__ndarray__'], val['dtype']).reshape(val['shape']), dtype=torch.float32)
            gradients.append(grad)
        return gradients

    def query_helpers(self, data_payload, model_payload, endpoints):
        gradient_list = []

        helper_list = data_payload.keys()

        for ii, helper in enumerate(helper_list):
            needed_func = "gradient_computation"
            privacy_settings = self.get_privacy_settings()

            json_payload = dict(
                origin=helper,
                function=needed_func,
                privacy_settings=privacy_settings,
                aggregation_service_payload_set=data_payload[helper],
                aggregation_model_set=model_payload
            )

            return_val = requests.post(endpoints[ii], json=json_payload)
            gradients = self.parse_return_json(return_val.json())
            gradient_list.append(gradients)
        return gradient_list[0]

    def merge_gradients(self, gradient_list):
        accumulated_grads = gradient_list[0]
        if len(gradient_list) > 1:
            for grad in gradient_list[1:]:
                for param in accumulated_grads:
                    accumulated_grads[param] += grad[param]
        return accumulated_grads

    def normalize_gradients(self, gradients, normalizer):
        for param in gradients.keys():
            gradients[param] /= normalizer
        return gradients

    def create_private_pseudodata(self, inputs, targets, helper_list):
        if self.task=='binary_classifier':
            inputs = torch.repeat_interleave(inputs, 4, dim=0)
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



    def send_post_request(self, inputs, targets, model, normalize_grads = True):
        model = payloadutils.pytorch_to_onnx(model, inputs)
        n_inputs = len(targets)
        helper_list = self.get_helper_names()
        if not n_inputs == len(targets) and n_inputs == x.shape[0]:
            raise ValueError('We require n_targets==n_inputs')

        inputs, targets, masks, helpers = self.create_private_pseudodata(inputs, targets, helper_list)
        n_inputs = len(masks)
        inputs = [inputs[ii, :] for ii in range(n_inputs)]
        data_payload = self.create_data_payload(inputs, targets, masks, self.model_name, helpers)
        model_payload = self.create_model_payload(model)
        
        endpoints = self.endpoints
        
        gradient_list = self.query_helpers(data_payload, model_payload, endpoints)
        accumulated_gradients = self.merge_gradients(gradient_list)
        if normalize_grads:
            accumulated_gradients = self.normalize_gradients(accumulated_gradients, n_inputs)
        return accumulated_gradients

    def set_gradients(self, network: torch.Module, gradients: dict[str: np.array]):
        for grad, (name, param) in zip(gradients.keys(), network.named_parameters()):
            param.grad = gradients[grad]

    def fetch_gradients(self, model: torch.Module, normalize_grads = True):
        '''
        :param model:
        :param normalize_grads:
        :return: None
        '''
        inputs, target_list = map(list,zip(*self.datasource.data_queue))
        inputs = torch.tensor(np.asarray(inputs), dtype=torch.float32)
        targets = torch.tensor(np.asarray(target_list), dtype=torch.float32)
        self.datasource.clear_dataqueue()
        self.grads = self.send_post_request(inputs, targets, model, normalize_grads)
        self.model = model
        return None

    def backward(self):
        '''
        Modifies the state of self.model to have gradients contained in self.grads. Should only be called after
        self.fetch_gradients()
        :return: None
        '''
        self.set_gradients(self.model, self.grads)
        return None