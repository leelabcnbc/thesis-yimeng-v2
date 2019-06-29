"""rewrite of reference prednet using my new generic interface

"""
import torch
from torch import nn
from torch.nn import functional
from .base import PredBlockBase


class PcConvBp(PredBlockBase):
    def __init__(self, inchan, outchan, kernel_size=3, stride=1, padding=1,
                 cls=0, bias=False,
                 act_fn='relu',
                 bypass=True,
                 no_act=False,
                 b0_init=1.0,
                 ):
        super().__init__(cls)
        self.FFconv = nn.Conv2d(inchan, outchan, kernel_size, stride, padding,
                                bias=bias)
        self.FBconv = nn.ConvTranspose2d(outchan, inchan, kernel_size, stride,
                                         padding, bias=bias)
        # noinspection PyUnresolvedReferences
        self.b0 = nn.Parameter(torch.ones(1, outchan, 1, 1)*b0_init)
        self.cls = cls
        if bypass:
            self.bypass = nn.Conv2d(inchan, outchan, kernel_size=1,
                                    stride=1,
                                    # bias=False or True should not matter,
                                    # I guess.
                                    bias=False)
        else:
            self.bypass = None

        if not no_act:
            if act_fn == 'relu':
                self.act_fn = nn.ReLU(inplace=True)
            elif act_fn == 'softplus':
                self.act_fn = nn.Softplus()
            else:
                raise NotImplementedError
        else:
            assert act_fn is None
            # https://stackoverflow.com/questions/8748036/is-there-a-builtin-identity-function-in-python
            self.act_fn = lambda x: x

    def forward_fb(self, input_higher):
        return self.FBconv(input_higher)

    def forward_init(self, input_lower):
        return self.act_fn(self.FFconv(input_lower))

    def forward_post(self, input_higher, input_lower):
        if self.bypass is not None:
            return input_higher + self.bypass(input_lower)
        else:
            return input_higher

    def forward_update(self, input_higher, input_lower, prediction):
        return self.FFconv(self.act_fn(input_lower - prediction)) * functional.relu(
            self.b0) + input_higher
