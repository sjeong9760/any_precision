import numpy as np

def conv(X, filters, stride=1, pad=0): 
    n, c, h, w = X.shape 
    n_f, _, filter_h, filter_w = filters.shape 
    out_h = (h + 2 * pad - filter_h) // stride + 1 
    out_w = (w + 2 * pad - filter_w) // stride + 1 # add padding to height and width. 
    in_X = np.pad(X, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant') 
    out = np.zeros((n, n_f, out_h, out_w)) 
    for i in range(n): # for each image. 
        for c in range(n_f): # for each channel. 
            for h in range(out_h): # slide the filter vertically. 
                h_start = h * stride 
                h_end = h_start + filter_h 
                for w in range(out_w): # slide the filter horizontally. 
                    w_start = w * stride 
                    w_end = w_start + filter_w 
                    # Element-wise multiplication. 
                    out[i, c, h, w] = np.sum(in_X[i, :, h_start:h_end, w_start:w_end] * filters[c]) 
    return out

def activate(x):
    ## Relu ##
    return np.clip.acti(x, 0, None)

def bn(x, num_features, gamma, beta):
    eps = 1e-05
    N, H, W, C = x.shape
    if C != num_features:
        print('Channel != num_features')
        raise ValueError
    out = np.zeros((N,H,W,C))
    for c in range(C):
        mean = np.mean(x[:,:,:,c])
        var = np.var(x[:,:,:,c])
        out[:,:,:,c] = gamma[c]* (x[n,:,:,c] - mean) / (np.sqrt(var + eps)) + beta[c]

    return out

def linear(x, in_features, out_features, weight, bias):
    return x@weight.T + bias


skip_conv = conv

class PreActBasicBlockQ_np():
    def __init__(self, num, bit_list, in_planes, out_planes, weight, stride=1, quan=False):

        ## Quantize ##
        if quan:

            self.bit_list = bit_list
            self.wbit = self.bit_list[-1]
            self.abit = self.bit_list[-1]
        
        self.weight = weight
        self.num = num
        self.out_planes = out_planes
        self.stride = stride
        self.in_planes = in_planes

        self.skip_conv = None
        if stride != 1:
            self.skip_conv = True

    def bn0(self, x, c, weight, b):
        return bn(x,c,weight,b)

    def act0(self, x):
        return activate(x)

    def conv0(self, x, weight, stride, pad):
        return conv(x, weight, stride, pad)

    def bn1(self, x, c, weight, b):
        return bn(x,c, weight,b)

    def act1(self, x):
        return activate(x)

    def conv1(self, x, weight, stride, pad):
        return conv(x, weight, stride, pad)

    def skip_conv(self, x, weight, stride, pad):
        return skip_conv(x, weight, stride, pad)

    def skip_bn(self, x, c, weight, b):
        return bn(x,c, weight,b)

    def forward(self, x, weight_dict):
        out = self.bn0(x, self.in_planes, weight_dict[f'layers.{self.num}.bn0.bn_dict.1.weight'], weight_dict[f'layers.{self.num}.bn0.bn_dict.1.weight'])
        out = self.act0(out)

        if self.skip_conv is not None:
            if self.stride != 1:
                padding = 0
            else:
                padding = 1
            shortcut = self.skip_conv(out, weight_dict[f'layers.{self.num}.skip_conv.weight'], stride = self.stride, pad = padding)
            shortcut = self.skip_bn(out, self.out_planes, weight_dict[f'layers.{self.num}.skip_bn.weight'], weight_dict[f'layers.{self.num}.skip_bn.bias'])

        else:
            shortcut = x


        out = self.conv0(out, weight_dict[f'layers.{self.num}.conv0.weight'], stride = self.stride, pad = 1)
        out = self.bn1(out, self.out_planes, weight_dict[f'layers.{self.num}.bn1.bn_dict.1.weight'], weight_dict[f'layers.{self.num}.bn0.bn_dict.1.bias'])
        out = self.act1(out)

        out = self.conv1(out, weight_dict[f'layers.{self.num}.conv1.weight'], stride = 1, pad = 1)

        out += shortcut
        return out

class PreActResNet_np():
    def __init__(self, block, num_units, bit_list, num_classes, expand=5):
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        self.abit = self.bit_list[-1]
        self.expand = expand
        self.num_classes = num_classes

        ep = self.expand
        
        ## selected layer list ##
        layer_list = [0, 3, 4, 6, 7]

        strides = [1] * num_units[0] + [2] + [1] * (num_units[1] - 1) + [2] + [1] * (num_units[2] - 1)
        channels = [16 * ep] * num_units[0] + [32 * ep] * num_units[1] + [64 * ep] * num_units[2]
        in_planes = 16 * ep

        strides = list(np.array(strides)[layer_list])
        channels = list(np.array(channels)[layer_list])

        self.layers = []

        for i, (stride, channel) in enumerate(zip(strides, channels)):
            self.layers.append(block(i, self.bit_list, in_planes, channel, stride))
            in_planes = channel

    def conv0(self, x, weight, stride, pad):
        return conv(x, weight, stride, pad)

    def fc(self, x, in_features, out_features, weight, bias):
        return linear(x, in_features, out_features, wieght, bias)

    def forward(self, x, weight_list):
        out = self.conv0(x, weight_list['conv0.weight'], 1, 1)
        for layer in self.layers:
            out = layer.forward(out, weight_list)
        out = bn(out, 64*self.expand, weight_list['bn.bn_dict.1.weight'], weight_list['bn.bn_dict.1.bias'])
        out = out.mean(dim=2).mean(dim=2)
        out = self.fc(out, 64*self.expand,num_classes, weight_list['fc.weight'], weight_list['fc.bias'])
        return out



# For CIFAR10
def resnet20q_np(bit_list, num_classes=10):
    return PreActResNet_np(PreActBasicBlockQ_np, [3, 3, 3], bit_list, num_classes=num_classes)









