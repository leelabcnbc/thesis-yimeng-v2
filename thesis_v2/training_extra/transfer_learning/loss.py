# follow https://github.com/leelabcnbc/thesis-proposal-yimeng-201808/blob/master/thesis_proposal_201804/maskcnn/cnn_aux.py#L158-L194  # noqa: E501

from copy import deepcopy

from torchnetjson.net import JSONNet
from ..evaluation import get_output_loss
from .opt_terms import sanity_check_opt_config
from ...blocks.maskcnn.nn_modules import FactoredLinear2D


from ..maskcnn_like.loss_terms import maskcnn_loss_v1_weight_readout


def get_loss(*, opt_config: dict,
             return_dict: bool = False):
    assert sanity_check_opt_config(opt_config)
    opt_config = deepcopy(opt_config)

    def loss_func_inner(yhat, y, model: JSONNet):
        # get it out from packed labels.
        # first one is neural. second one is aux.
        yhat_neural = yhat[0]
        y_neural = y[0]

        sparse_reg = _get_sparsity_loss(model.get_module('fc'),
                                        opt_config['fc']['sparse'])

        output_loss = get_output_loss(
            yhat=yhat_neural,
            y=y_neural,
            loss_type=opt_config['loss']
        )

        if return_dict:
            # for debugging
            result_dict = {
                'sparse_reg': sparse_reg,
                'poisson': output_loss,
                'total_loss': sparse_reg + output_loss,
            }
            # this cannot do backprop, since only item() is left.
            for x1, x2 in result_dict.items():
                result_dict[x1] = x2.data.item()
            return result_dict
        else:
            return sparse_reg + output_loss

    return loss_func_inner


def _get_sparsity_loss(module, sparse: float):
    if isinstance(module, FactoredLinear2D):
        return maskcnn_loss_v1_weight_readout(module,
                                              scale=sparse,
                                              legacy=False)
    else:
        raise RuntimeError
