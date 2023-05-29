import torch
import numpy as np


def to_dense_prediction_model(model, axis=(2, 3)):
    """
    Transform a sequential model with strides to a model that outputs
    dense predictions by removing the strides and instead inserting dilations.
    Modifies model in-place.

    Parameters
    ----------
    model: torch.nn.Module
        Model which modules will be modified
    axis: int or (int,int)
        Axis to transform (in terms of intermediate output axes)
        can either be 2, 3, or (2,3).

    Notes
    -----
    Does not yet work correctly for average pooling.
    Prior to version 0.1.7, there had been a bug that could move strides
    backwards one layer.

    """
    if not hasattr(axis, "__len__"):
        axis = [axis]
    assert all([ax in [2, 3] for ax in axis]), "Only 2 and 3 allowed for axis"
    axis = np.array(axis) - 2
    stride_so_far = np.array([1, 1])
    for module in model.modules():
        if hasattr(module, "dilation"):
            assert module.dilation == 1 or (module.dilation == (1, 1)), (
                "Dilation should equal 1 before conversion, maybe the model is "
                "already converted?"
            )
            new_dilation = [1, 1]
            for ax in axis:
                new_dilation[ax] = int(stride_so_far[ax])
            module.dilation = tuple(new_dilation)
        if hasattr(module, "stride"):
            if not hasattr(module.stride, "__len__"):
                module.stride = (module.stride, module.stride)
            stride_so_far *= np.array(module.stride)
            new_stride = list(module.stride)
            for ax in axis:
                new_stride[ax] = 1
            module.stride = tuple(new_stride)


def get_output_shape(model, in_chans, input_window_samples):
    """Returns shape of neural network output for batch size equal 1.

    Returns
    -------
    output_shape: tuple
        shape of the network output for `batch_size==1` (1, ...)
    """
    with torch.no_grad():
        dummy_input = torch.ones(
            1, in_chans, input_window_samples,
            dtype=next(model.parameters()).dtype,
            device=next(model.parameters()).device,
        )
        output_shape = model(dummy_input).shape
    return output_shape

class ThrowAwayIndexLoader(object):
    def __init__(self, net, loader, is_regression):
        self.net = net
        self.loader = loader
        self.last_i = None
        self.is_regression = is_regression

    def __iter__(self, ):
        normal_iter = self.loader.__iter__()
        for batch in normal_iter:
            if len(batch) == 3:
                x, y, i = batch
                # Store for scoring callbacks
                self.net._last_window_inds = i
            else:
                x, y = batch

            # TODO: should be on dataset side
            if hasattr(x, 'type'):
                x = x.type(torch.float32)
                if self.is_regression:
                    y = y.type(torch.float32)
                else:
                    y = y.type(torch.int64)
            yield x, y


def update_estimator_docstring(base_class, docstring):
    base_doc = base_class.__doc__.replace(' : ', ': ')
    idx = base_doc.find('callbacks:')
    idx_end = idx + base_doc[idx:].find('\n\n')
    # remove callback descripiton already included in braindecode docstring
    filtered_doc = base_doc[:idx] + base_doc[idx_end+6:]
    splitted = docstring.split('Parameters\n    ----------\n    ')
    out_docstring = splitted[0] + \
                    filtered_doc[filtered_doc.find('Parameters'):filtered_doc.find('Attributes')] + \
                    splitted[1] + \
                    filtered_doc[filtered_doc.find('Attributes'):]
    return out_docstring
