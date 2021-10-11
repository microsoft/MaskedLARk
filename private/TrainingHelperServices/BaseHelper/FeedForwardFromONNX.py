import torch
import torch.nn as nn
import onnx
from onnx import numpy_helper
from functools import partial
import warnings
from torch.jit import TracerWarning
from torch.nn.modules.linear import Identity


from operations import *
from operations.base import OperatorWrapper

permissible_op_types = ['Gemm', 'Relu', 'Sigmoid']

TENSOR_PROTO_MAPPING = dict([i[::-1] for i in onnx.TensorProto.DataType.items()])

AttributeType = dict(
    UNDEFINED=0,
    FLOAT=1,
    INT=2,
    STRING=3,
    TENSOR=4,
    GRAPH=5,
    SPARSE_TENSOR=11,
    FLOATS=6,
    INTS=7,
    STRINGS=8,
    TENSORS=9,
    GRAPHS=10,
    SPARSE_TENSORS=12,
)


def extract_attr_values(attr):
    """Extract onnx attribute values."""
    if attr.type == AttributeType["INT"]:
        value = attr.i
    elif attr.type == AttributeType["FLOAT"]:
        value = attr.f
    elif attr.type == AttributeType["INTS"]:
        value = tuple(attr.ints)
    elif attr.type == AttributeType["FLOATS"]:
        value = tuple(attr.floats)
    elif attr.type == AttributeType["TENSOR"]:
        value = onnx.numpy_helper.to_array(attr.t)
    elif attr.type == AttributeType["STRING"]:
        value = attr.s.decode()
    else:
        raise NotImplementedError(
            "Extraction of attribute type {} not implemented.".format(attr.type)
        )
    return value

def extract_attributes(node):
    """Extract onnx attributes. Map onnx feature naming to pytorch."""
    kwargs = {}
    for attr in node.attribute:
        if attr.name == "dilations":
            kwargs["dilation"] = extract_attr_values(attr)
        elif attr.name == "group":
            kwargs["groups"] = extract_attr_values(attr)
        elif attr.name == "axis" and node.op_type == "Flatten":
            kwargs["start_dim"] = extract_attr_values(attr)
        elif attr.name == "axis" or attr.name == "axes":
            v = extract_attr_values(attr)
            if isinstance(v, (tuple, list)) and len(v) == 1:
                kwargs["dim"] = v[0]
            else:
                kwargs["dim"] = v
        elif attr.name == "keepdims":
            kwargs["keepdim"] = bool(extract_attr_values(attr))
        elif attr.name == "epsilon":
            kwargs["eps"] = extract_attr_values(attr)
        elif attr.name == "momentum":
            kwargs["momentum"] = extract_attr_values(attr)
        elif attr.name == "ceil_mode":
            kwargs["ceil_mode"] = bool(extract_attr_values(attr))
        elif attr.name == "value":
            kwargs["constant"] = extract_attr_values(attr)
        elif attr.name == "perm":
            kwargs["dims"] = extract_attr_values(attr)
        elif attr.name == "to":
            kwargs["dtype"] = TENSOR_PROTO_MAPPING[extract_attr_values(attr)].lower()
        elif attr.name == "mode":
            kwargs["mode"] = extract_attr_values(attr)
        elif attr.name == "transB":
            kwargs["transpose_weight"] = not extract_attr_values(attr)
        elif attr.name == "transA":
            kwargs["transpose_activation"] = bool(extract_attr_values(attr))
        elif attr.name == "alpha" and node.op_type == "LeakyRelu":
            kwargs["negative_slope"] = extract_attr_values(attr)
        elif attr.name == "alpha" and node.op_type == "Elu":
            kwargs["alpha"] = extract_attr_values(attr)
        elif attr.name == "alpha":
            kwargs["weight_multiplier"] = extract_attr_values(attr)
        elif attr.name == "beta":
            kwargs["bias_multiplier"] = extract_attr_values(attr)
        elif attr.name == "starts":
            kwargs["starts"] = extract_attr_values(attr)
        elif attr.name == "ends":
            kwargs["ends"] = extract_attr_values(attr)
        else:
            raise NotImplementedError(
                "Extraction of attribute {} not implemented.".format(attr.name)
            )
    return kwargs


def fill_linear_layer(layer, weight, bias):
    """Load weight and bias to a given layer from onnx format."""
    with torch.no_grad():
        layer.weight.data = torch.from_numpy(onnx.numpy_helper.to_array(weight))
        if bias is not None:
            layer.bias.data = torch.from_numpy(onnx.numpy_helper.to_array(bias))

def convert_linear_layer(node, params):
    """Convert linear layer from onnx node and params."""
    # Default Gemm attributes
    dc = dict(
        transpose_weight=True,
        transpose_activation=False,
        weight_multiplier=1,
        bias_multiplier=1,
    )
    dc.update(extract_attributes(node))
    for attr in node.attribute:
        if attr.name in ["transA"] and extract_attr_values(attr) != 0:
            raise NotImplementedError(
                "Not implemented for attr.name={} and value!=0.".format(attr.name)
            )

    kwargs = {}
    weight, bias = extract_params(params)
    kwargs["bias"] = bias is not None
    kwargs["in_features"] = weight.dims[1]
    kwargs["out_features"] = weight.dims[0]

    # initialize layer and load weights
    layer = nn.Linear(**kwargs)
    fill_linear_layer(layer, weight, bias)

    # apply onnx gemm attributes
    if dc.get("transpose_weight"):
        layer.weight.data = layer.weight.data.t()

    layer.weight.data *= dc.get("weight_multiplier")
    if layer.bias is not None:
        layer.bias.data *= dc.get("bias_multiplier")

    return layer


def extract_params(params):
    """Extract weights and biases."""
    param_length = len(params)
    if param_length == 1:
        weight = params[0]
        bias = None
    elif param_length == 2:
        weight = params[0]
        bias = params[1]
    else:
        raise ValueError("Unexpected number of parameters: {}. Only Linear and activation function layers are supported at this time.".format(param_length))
    return weight, bias

def value_wrapper(value):
    def callback(*args, **kwargs):
        return value

    return callback



def convert_operations(onnx_model, batch_dim=0):
    """
    Convert onnx model operations. Yields onnx's operator_id, operator_name and
    converted pytorch operator.

    Parameters
    ----------
    onnx_model: onnx.ModelProto
        Loaded onnx model.
    batch_dim: int
        Usually 0 for computer vision models and 1 for NLP models.

    Returns
    -------
    iterator: (op_id, op_name, op)
    """
    weights = {tensor.name: tensor for tensor in onnx_model.graph.initializer}
    opset_version = onnx_model.opset_import[0].version

    for i, node in enumerate(onnx_model.graph.node):
        # extract only useful inputs
        params = [weights[par_name] for par_name in node.input if par_name in weights]

        if node.op_type == "Conv":
            op = convert_layer(node, "Conv", params)
        elif node.op_type == "Relu":
            op = nn.ReLU(inplace=True)
        elif node.op_type == "LeakyRelu":
            op = nn.LeakyReLU(**extract_attributes(node), inplace=True)
        elif node.op_type == "Elu":
            op = nn.ELU(**extract_attributes(node), inplace=True)
        elif node.op_type == "Sigmoid":
            op = nn.Sigmoid()
        elif node.op_type == "MaxPool":
            op = convert_layer(node, "MaxPool")
        elif node.op_type == "AveragePool":
            op = convert_layer(node, "AvgPool")
        elif node.op_type == "Flatten":
            op = Flatten(**extract_attributes(node))
        elif node.op_type == "Gemm":
            op = convert_linear_layer(node, params)
            op.feature_dim = batch_dim + 1  # Necessary for transformers
        elif node.op_type == "BatchNormalization":
            op = convert_batch_norm_layer(node, params=params)
        elif node.op_type == "InstanceNormalization":
            op = convert_instance_norm_layer(node, params=params)
        elif node.op_type == "Concat":
            op = partial(torch.cat, **extract_attributes(node))
        elif node.op_type == "Constant":
            op = value_wrapper(torch.from_numpy(extract_attributes(node)["constant"]))
        elif node.op_type == "Reshape":
            shape = list(
                filter(lambda x: x.name == node.input[1], onnx_model.graph.initializer)
            )
            shape = numpy_helper.to_array(shape[0]) if shape else None
            op = Reshape(shape)
        elif node.op_type == "Shape":
            op = Shape()
        elif node.op_type == "Expand":
            op = Expand()
        elif node.op_type == "Gather":
            op = Gather(**extract_attributes(node))
        elif node.op_type == "Squeeze":
            op = Squeeze(opset_version=opset_version, **extract_attributes(node))
        elif node.op_type == "Unsqueeze":
            op = Unsqueeze(opset_version=opset_version, **extract_attributes(node))
        elif node.op_type == "ConstantOfShape":
            op = ConstantOfShape(**extract_attributes(node))
        elif node.op_type == "Range":
            op = Range()
        elif node.op_type == "Slice":
            op = Slice(**extract_attributes(node))
        elif node.op_type == "Cast":
            op = Cast(**extract_attributes(node))
        elif node.op_type == "Where":
            op = torch.where
        elif node.op_type == "Equal":
            op = torch.eq
        elif node.op_type == "Mul":
            op = torch.mul
        elif node.op_type == "Div":
            op = torch.true_divide
        elif node.op_type == "MatMul":
            if params:
                weight = torch.from_numpy(numpy_helper.to_array(params[0]))
                op = nn.Linear(weight.shape[0], weight.shape[1], bias=False)
                op.weight.data = weight.t()

                # check if next node Add to add bias
                next_node = onnx_model.graph.node[i + 1]
                next_params = [
                    weights[par_name]
                    for par_name in next_node.input
                    if par_name in weights
                ]
                if next_params and next_node.op_type == "Add":
                    bias = torch.from_numpy(numpy_helper.to_array(next_params[0]))
                    op.bias = nn.Parameter(bias)
                    node.output.pop()
                    node.output.extend(next_node.output)
                    onnx_model.graph.node.pop(i + 1)  # remove next node
            else:
                op = torch.matmul
        elif node.op_type == "Sub":
            op = torch.sub
        elif node.op_type == "Pow":
            op = torch.pow
        elif node.op_type == "Sqrt":
            op = torch.sqrt
        elif node.op_type == "Softmax":
            op = nn.Softmax(**extract_attributes(node))
        elif node.op_type == "Transpose":
            op = partial(torch.Tensor.permute, **extract_attributes(node))
        elif node.op_type == "Split":
            kwargs = extract_attributes(node)
            # if the split_size_or_sections is not in node attributes,
            # the number_of_splits becomes the number of node outputs
            if "split_size_or_sections" not in kwargs:
                kwargs["number_of_splits"] = len(node.output)
            op = Split(**kwargs)
        elif node.op_type == "ReduceMean":
            kwargs = dict(keepdim=True)
            kwargs.update(extract_attributes(node))
            op = partial(torch.mean, **kwargs)
        elif node.op_type == "Add":
            op = Add(feature_dim=batch_dim + 1)  # 0 for CV models and 1 for NLP
        elif node.op_type == "Identity":
            op = nn.Identity()
        elif node.op_type == "Resize":
            op = Resize(**extract_attributes(node))
        elif node.op_type == "OneHot":
            op = OneHot(**extract_attributes(node))
        elif node.op_type == "Clip":
            op = OperatorWrapper(torch.clamp)
        elif node.op_type == "Tanh":
            op = OperatorWrapper(torch.tanh)
        elif node.op_type == "Erf":
            op = OperatorWrapper(torch.erf)
        elif node.op_type == "Log":
            op = OperatorWrapper(torch.log)
        elif node.op_type == "Exp":
            op = OperatorWrapper(torch.exp)
        elif node.op_type == "Reciprocal":
            op = OperatorWrapper(torch.reciprocal)
        elif node.op_type == "And":
            op = OperatorWrapper(torch.logical_and)
        elif node.op_type == "Or":
            op = OperatorWrapper(torch.logical_or)
        elif node.op_type == "Not":
            op = OperatorWrapper(torch.logical_not)
        else:
            op = getattr(torch, node.op_type.lower(), None)
            if op is None:
                raise NotImplementedError(
                    "Conversion not implemented for op_type={}.".format(node.op_type)
                )
            else:
                print(
                    "Automatic inference of operator: {}".format(node.op_type.lower())
                )

        op_name = "{}_{}".format(node.op_type, node.output[0])
        op_id = node.output[0]
        yield op_id, op_name, op

class InitParameters(dict):
    """Use for parameters that are hidden."""

    def __getitem__(self, item):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", TracerWarning)
            return torch.from_numpy(numpy_helper.to_array(super().__getitem__(item)))

    def get(self, item, default):
        if item in self:
            return self[item]
        else:
            return default

def get_inputs_names(onnx_model):
    param_names = set([x.name for x in onnx_model.graph.initializer])
    input_names = [x.name for x in onnx_model.graph.input]
    input_names = [x for x in input_names if x not in param_names]
    return input_names

class ConvertModel(nn.Module):
    def __init__(
        self, onnx_model: onnx.ModelProto, batch_dim=0, experimental=False, debug=False
    ):
        """
        Convert onnx model to pytorch.

        Parameters
        ----------
        onnx_model: onnx.ModelProto
            Loaded onnx model.
        batch_dim: int
            Dimension of the batch.
        experimental: bool
            Experimental implementation allows batch_size > 1. However,
            batchnorm layers could potentially produce false outputs.

        Returns
        -------
        model: torch.nn.Module
            A converted pytorch model.
        """
        super().__init__()
        self.onnx_model = onnx_model
        self.batch_dim = batch_dim
        self.experimental = experimental
        self.debug = debug
        self.mapping = {}
        for op_id, op_name, op in convert_operations(onnx_model, batch_dim):
            setattr(self, op_name, op)
            self.mapping[op_id] = op_name

        self.init_parameters = InitParameters(
            {tensor.name: tensor for tensor in self.onnx_model.graph.initializer}
        )

        self.input_names = get_inputs_names(onnx_model)

    def forward(self, *input):
        # TODO figure out how to store only necessary activations.
        activations = dict(zip(self.input_names, input))

        for node in self.onnx_model.graph.node:
            # Identifying the layer ids and names
            out_op_id = node.output[0]
            out_op_name = self.mapping[out_op_id]

            # getting correct layer
            op = getattr(self, out_op_name)

            # if first layer choose input as in_activations
            # if not in_op_names and len(node.input) == 1:
            #    in_activations = input
            layer_types = (nn.Linear)
            if isinstance(op, layer_types) or (
                isinstance(op, nn.Sequential)
                and any(isinstance(x, layer_types) for x in op.modules())
            ):
                in_activations = [
                    activations[in_op_id]
                    for in_op_id in node.input
                    if in_op_id in activations
                ]
            else:
                in_activations = [
                    activations[in_op_id] if in_op_id in activations
                    # if in_op_id not in activations neither in parameters then
                    # it must be the initial input
                    # TODO loading parameters in forward func might be very slow!
                    else self.init_parameters.get(in_op_id, input[0])
                    for in_op_id in node.input
                ]

            # store activations for next layer
            if isinstance(op, partial) and op.func == torch.cat:
                activations[out_op_id] = op(in_activations)
            elif isinstance(op, Split):
                for out_op_id, output in zip(node.output, op(*in_activations)):
                    activations[out_op_id] = output
            elif isinstance(op, Identity):
                # After batch norm fusion the batch norm parameters
                # were all passed to identity instead of first one only
                activations[out_op_id] = op(in_activations[0])
            else:
                activations[out_op_id] = op(*in_activations)

        # collect all outputs
        outputs = [activations[x.name] for x in self.onnx_model.graph.output]
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs