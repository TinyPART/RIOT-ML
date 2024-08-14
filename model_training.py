import tvm
import tvm.micro
import tvm.micro.testing
from tvm import relay
from tvm.micro import export_model_library_format
from tvm.micro.testing.utils import (
    # create_header_file,
    mlf_extract_workspace_size_bytes,
    aot_transport_init_wait,
    aot_transport_find_message,
)

# from tvm.relay.op.op import register_gradient
# from tvm.relay.op import nn as _nn
# from tvm.relay.op.tensor import zeros_like
# # dummy gradients as an placeholder
# @register_gradient("nn.adaptive_avg_pool2d")
# def adaptive_avg_pool2d_grad(orig, grad):
#     """Returns the gradient of avg_pool2d."""
#     return [zeros_like(_) for _ in orig.args]


test_list = ['classifier_3_weight']

labels = relay.var('labels', relay.TensorType((1, 10), "float32"))

#log_softmax = relay.nn.log_softmax(predictions)
log_softmax = relay.nn.log_softmax(mod['main'].body)

loss = relay.nn.cross_entropy_with_logits(log_softmax, labels)
loss = relay.Function(relay.analysis.free_vars(loss), loss)

loss_mod = tvm.IRModule.from_expr(loss)
# loss_func = relay.transform.Defunctionalization(mod["main"], mod)
# loss_mod = relay.transform.ToANormalForm()(loss_mod)
# loss_mod = tvm.transform.Sequential([
#     # relay.transform.InferType(),
#     # relay.transform.PartialEvaluate(),
#     # relay.transform.DeadCodeElimination(),
#     relay.transform.ToGraphNormalForm(),
# ])(loss_mod)
loss_mod = relay.transform.InferType()(loss_mod)
print((loss_mod))
breakpoint()

#bwd_loss_mod = relay.transform.FirstOrderGradient(['predictions'])(loss_mod)
bwd_loss_mod = relay.transform.FirstOrderGradient(test_list, False, False)(loss_mod)

bwd_loss_mod = tvm.transform.Sequential([
    # relay.transform.PartialEvaluate(),
    relay.transform.DeadCodeElimination(),
    relay.transform.ToGraphNormalForm(),
])(bwd_loss_mod)

def build_sgd(param_shape, lr=None, weight_decay=None):
    param = relay.var("input_param", shape=param_shape, dtype="float32")
    grad = relay.var("input_grad", shape=param_shape, dtype="float32")
    lr = relay.constant(lr, dtype="float32") if lr is not None else relay.var("lr", "float32") 
    weight_decay = relay.constant(weight_decay, dtype="float32") if weight_decay is not None else relay.var("weight_decay", "float32") 

    new_param = relay.multiply(weight_decay, param)
    new_param = relay.add(grad, new_param)
    new_param = relay.multiply(lr, new_param)
    new_param = relay.subtract(param, new_param)
    return relay.Function(relay.analysis.free_vars(new_param), new_param)


sgd_func = build_sgd(params['classifier_3_weight'].shape)
sgd_mod = tvm.IRModule.from_expr(sgd_func)
sgd_mod = relay.transform.InferType()(sgd_mod)