import torch.onnx.utils as onnxutils
import torch


'''
This is taken wholesale from https://github.com/pytorch/pytorch/blob/master/torch/onnx/utils.py
'''

__IN_ONNX_EXPORT = False


def _export_to_protobuf(model, args, export_params=True, verbose=False, training=None,
            input_names=None, output_names=None, operator_export_type=None,
            export_type=torch.onnx.ExportTypes.PROTOBUF_FILE, example_outputs=None,
            opset_version=None, _retain_param_name=False, do_constant_folding=True,
            strip_doc_string=True, dynamic_axes=None, keep_initializers_as_inputs=None,
            fixed_batch_size=False, custom_opsets=None, add_node_names=True,
            enable_onnx_checker=True,
            onnx_shape_inference=True):

    if isinstance(model, torch.nn.DataParallel):
        raise ValueError("torch.nn.DataParallel is not supported by ONNX "
                         "exporter, please use 'attribute' module to "
                         "unwrap model from torch.nn.DataParallel. Try "
                         "torch.onnx.export(model.module, ...)")
    global __IN_ONNX_EXPORT
    assert __IN_ONNX_EXPORT is False
    __IN_ONNX_EXPORT = True
    try:
        from torch.onnx.symbolic_helper import _set_onnx_shape_inference
        _set_onnx_shape_inference(onnx_shape_inference)

        from torch.onnx.symbolic_helper import _default_onnx_opset_version, _set_opset_version
        from torch.onnx.symbolic_helper import _set_operator_export_type
        if opset_version is None:
            opset_version = _default_onnx_opset_version
        if not operator_export_type:
            if torch.onnx.PYTORCH_ONNX_CAFFE2_BUNDLE:
                operator_export_type = torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
            else:
                operator_export_type = torch.onnx.OperatorExportTypes.ONNX

        # By default, training=None, (which defaults to TrainingMode.EVAL),
        # which is good because running a model in training mode could result in
        # internal buffers getting updated, dropout getting applied, etc.
        # If you really know what you're doing, you can turn
        # training=TrainingMode.TRAINING or training=TrainingMode.PRESERVE,
        # (to preserve whatever the original training mode was.)
        _set_opset_version(opset_version)
        _set_operator_export_type(operator_export_type)
        with onnxutils.select_model_mode_for_export(model, training):
            val_keep_init_as_ip = onnxutils._decide_keep_init_as_input(keep_initializers_as_inputs,
                                                             operator_export_type,
                                                             opset_version)
            val_add_node_names = onnxutils._decide_add_node_names(add_node_names, operator_export_type)
            val_do_constant_folding = onnxutils._decide_constant_folding(do_constant_folding, operator_export_type, training)
            args = onnxutils._decide_input_format(model, args)
            if dynamic_axes is None:
                dynamic_axes = {}
            onnxutils._validate_dynamic_axes(dynamic_axes, model, input_names, output_names)

            graph, params_dict, torch_out = \
                onnxutils._model_to_graph(model, args, verbose, input_names,
                                output_names, operator_export_type,
                                example_outputs, _retain_param_name,
                                val_do_constant_folding,
                                fixed_batch_size=fixed_batch_size,
                                training=training,
                                dynamic_axes=dynamic_axes)

            defer_weight_export = export_type is not torch.onnx.ExportTypes.PROTOBUF_FILE
            if custom_opsets is None:
                custom_opsets = {}

            if export_params:
                proto, export_map = graph._export_onnx(
                    params_dict, opset_version, dynamic_axes, defer_weight_export,
                    operator_export_type, strip_doc_string, val_keep_init_as_ip, custom_opsets,
                    val_add_node_names)
            else:
                proto, export_map = graph._export_onnx(
                    {}, opset_version, dynamic_axes, False, operator_export_type,
                    strip_doc_string, val_keep_init_as_ip, custom_opsets, val_add_node_names)

            if enable_onnx_checker and \
                operator_export_type is torch.onnx.OperatorExportTypes.ONNX:
                # Only run checker if enabled and we are using ONNX export type and
                # large model format export in not enabled.
                onnxutils._check_onnx_proto(proto)
    finally:
        assert __IN_ONNX_EXPORT
        __IN_ONNX_EXPORT = False
    return proto