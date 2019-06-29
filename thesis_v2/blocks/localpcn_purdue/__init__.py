"""this is a generic version of the prednet block in

https://github.com/libilab/PCN-with-Local-Recurrent-Processing

to be generic, we need to support other types of loss function other than
MSE loss, and bypass loss should be optional.

the motivation is to find the best prednet block, and make connections to
sparse coding, and other older studies.


TODO: feedback connections so that lower layer representations got changed.
      right now the original paper claims that the model has feedback;
      however, the feedback connections does not change lower layer.
      To enable change of lower layer representation, maybe we can
      optimize an objective function for lower layer as well, except that
      there we may optimize less, or whatever.
"""


from . import nn_modules
