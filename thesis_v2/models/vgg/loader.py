from torchvision.models import (
    vgg16, vgg16_bn, vgg19, vgg19_bn,
    vgg11, vgg11_bn, vgg13, vgg13_bn
)


def get_pretrained_network(net_name):
    a = {'vgg16': vgg16, 'vgg19': vgg19, 'vgg16_bn': vgg16_bn,
         'vgg19_bn': vgg19_bn,
         'vgg11': vgg11, 'vgg13': vgg13, 'vgg11_bn': vgg11_bn,
         'vgg13_bn': vgg13_bn,
         }[net_name](pretrained=True)
    # a.cuda()
    a.eval()
    return a
