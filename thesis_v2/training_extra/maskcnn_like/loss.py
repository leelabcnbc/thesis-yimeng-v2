# follow https://github.com/leelabcnbc/thesis-proposal-yimeng-201808/blob/master/thesis_proposal_201804/maskcnn/cnn_aux.py#L158-L194  # noqa: E501

from copy import deepcopy

from torchnetjson.net import JSONNet
from . import loss_terms as optim_maskcnn
from .opt_terms import sanity_check_opt_config
from ..evaluation import get_output_loss


def get_loss(*, opt_config: dict,
             return_dict: bool = False,
             device=None, handle_nan=False):
    assert sanity_check_opt_config(opt_config)
    opt_config = deepcopy(opt_config)

    group_list = [x['group'] for x in opt_config['conv']]
    smooth_list = [x['smoothness'] for x in opt_config['conv']]

    def loss_func_inner(yhat, y, model: JSONNet):
        # get it out from packed labels.
        # first one is neural. second one is aux.
        assert len(yhat) == len(y) == 1
        yhat_neural = yhat[0]
        y_neural = y[0]

        conv_module_list_this = model.extradict['conv_layers']
        group_sparsity = optim_maskcnn.maskcnn_loss_v1_kernel_group_sparsity(
            conv_module_list_this, group_list)
        smooth_sparsity = optim_maskcnn.maskcnn_loss_v1_kernel_smoothness(
            conv_module_list_this, smooth_list, device=device)
        readout_reg = optim_maskcnn.maskcnn_loss_v1_weight_readout(
            model.get_module('fc'),
            scale=opt_config['fc']['scale'],
            legacy=opt_config['legacy'])
        output_loss = get_output_loss(yhat=yhat_neural,
                                      y=y_neural,
                                      loss_type=opt_config['loss'],
                                      legacy=opt_config['legacy'], handle_nan=handle_nan)

        if return_dict:
            # for debugging
            result_dict = {
                'group_sparsity': group_sparsity,
                'smooth_sparsity': smooth_sparsity,
                'readout_reg': readout_reg,
                'poisson': output_loss,
                'total_loss': (group_sparsity + smooth_sparsity +
                               readout_reg + output_loss)
            }
            for x1, x2 in result_dict.items():
                result_dict[x1] = x2.data.item()
            return result_dict
        else:
            return group_sparsity + smooth_sparsity + readout_reg + output_loss

    return loss_func_inner
