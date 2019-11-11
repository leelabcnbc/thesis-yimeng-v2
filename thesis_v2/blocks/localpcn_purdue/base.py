from abc import ABC, abstractmethod
from torch import nn
from ..util import LambdaSingle

class PredBlockBase(nn.Module, ABC):
    def __init__(self, num_cycle):
        super().__init__()
        self.num_cycle = num_cycle
        # _in captures every blob with same shape as input
        # _out captures every blob with same shape as output
        self.lambda_in = LambdaSingle()
        self.lambda_out = LambdaSingle()

    @abstractmethod
    def forward_update(self, input_higher, input_lower, prediction):
        # generate
        raise NotImplementedError

    @abstractmethod
    def forward_init(self, input_lower):
        # generate first iteration of high layers.
        # it can be simply using forward_ff.
        raise NotImplementedError

    @abstractmethod
    def forward_fb(self, input_higher):
        # from higher layer to generate predictions.
        raise NotImplementedError

    @abstractmethod
    def forward_post(self, input_higher, input_lower):
        # these two can be list or tuple of Tensors.
        raise NotImplementedError

    def forward(self, input_lower):
        # for num_cycle = T

        # we have T+2 lambda_out
        # and T lambda_pred
        input_lower = self.lambda_in(input_lower)

        # input_lower can be a Tensor or a list/tuple of Tensors.
        higher = self.lambda_out(self.forward_init(input_lower))

        # update

        for _ in range(self.num_cycle):
            prediction = self.lambda_in(self.forward_fb(higher))
            higher = self.lambda_out(self.forward_update(higher, input_lower, prediction))

        higher = self.lambda_out(self.forward_post(higher, input_lower))
        return higher
