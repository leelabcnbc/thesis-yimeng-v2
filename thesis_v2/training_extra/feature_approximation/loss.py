from torchnetjson.net import JSONNet
from torch.nn.functional import mse_loss, l1_loss
from .opt_terms import sanity_check_opt_config
from copy import deepcopy


def get_loss(*, opt_config=None):
    assert sanity_check_opt_config(opt_config)
    opt_config = deepcopy(opt_config)

    loss_set = {'l1', 'mse'}
    assert opt_config['loss'] in loss_set

    def loss_func_inner(yhat, y, model: JSONNet):
        # get it out from packed labels.
        # first one is neural. second one is aux.
        assert len(yhat) == len(y) == 1
        if opt_config['loss'] == 'mse':
            return mse_loss(yhat[0], y[0], reduction='mean')
        elif opt_config['loss'] == 'l1':
            return l1_loss(yhat[0], y[0], reduction='mean')
        else:
            raise ValueError

    return loss_func_inner
