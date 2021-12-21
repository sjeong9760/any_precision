import pickle
import argparse
import torch
import numpy as np
import math

import models

def quantize_weight(data):
    E = torch.mean(torch.abs(data)).detach()
    weight = torch.tanh(data)
    weight = weight / 2 / torch.max(torch.abs(weight)) +0.5
    weight_q = 2 * torch.round(weight) - 1
    weight_q = weight_q * E
    return weight_q

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', default = 'custom_pretrained_5/114_epoch_1_bit.pth.tar')
    parser.add_argument('--save', action = 'store_true', default = False)
    args = parser.parse_args()

    # Load pretrained parameters
    checkpoint = torch.load(args.pretrain, map_location = torch.device('cpu'))

    # Init weights
    model = models.__dict__['resnet20q']([1], 10)
    model.load_state_dict(checkpoint['state_dict'])

    # Collect BN statistics
    bn = {}
    for i, x in enumerate(model.modules()):
        if isinstance(x, torch.nn.BatchNorm2d):
            bn[i] = {}
            bn[i]['param'] = x.weight.detach().numpy()
            bn[i]['mean'] = x.running_mean.numpy()
            bn[i]['var'] = x.running_var.numpy()
            bn[i]['eps'] = x.eps

    params = {}
    for i, (name, param) in enumerate(model.named_parameters()):
        params[name] = {}
        # Module list (module list always contains quantized weights for conv)
        if 'layers' in name:
            layer_num = int(name.split('.')[1])
            # Ignore 1, 2, 5, 8 layers
            if not layer_num in [1, 2, 5, 8]:
                if 'conv' in name:
                    # We skip this layer
                    if 'conv0' in name and layer_num == 7:
                        continue
                    quant_weight = quantize_weight(param.data)
                    # Get scale factor
                    scale = torch.abs(quant_weight[0][0][0][0]).item()
                    # Get binary value
                    quant_weight = quant_weight.apply_(lambda x : False if x <= 0.0 else True)
                    # From NCHW to NHWC
                    quant_weight = quant_weight.permute(0, 2, 3, 1)
                    # To numpy array
                    quant_weight = quant_weight.numpy().astype(np.uint8)
                    params[name]['shape'] = quant_weight.shape
                    # Pack bits
                    quant_weight = np.packbits(quant_weight.flatten())
                    params[name]['param'] = quant_weight
                    params[name]['scale'] = scale
                # batchnorm
                else:
                    # We skip this layer
                    if 'bn1' in name and layer_num == 7:
                        continue
                    if 'bn' in name and 'weight' in name:
                        params[name]['param'] = param.data.numpy()
                        params[name]['shape'] = params[name]['param'].shape
                        # Find matching bn statistics
                        did = True
                        for idx, bn_param in bn.items():
                            if np.array_equal(bn_param['param'], params[name]['param']):
                                params[name]['mean'] = bn_param['mean']
                                params[name]['var'] = bn_param['var']
                                params[name]['eps'] = bn_param['eps']
                                did = True
                                break
                        assert did
                    else: # bn bias
                        params[name]['param'] = param.data.numpy()
                        params[name]['shape'] = params[name]['param'].shape
            else:
                del params[name]
        else: # bn and fc
            if 'bn' in name and 'weight' in name: # bn weight
                params[name]['param'] = param.data.numpy()
                params[name]['shape'] = params[name]['param'].shape
                # Find matching bn statistics
                did = True
                for idx, bn_param in bn.items():
                    if np.array_equal(bn_param['param'], params[name]['param']):
                        params[name]['mean'] = bn_param['mean']
                        params[name]['var'] = bn_param['var']
                        params[name]['eps'] = bn_param['eps']
                        did = True
                        break
                assert did
            else: 
                if 'fc' in name and 'weight' in name: # fc
                    quant_weight = quantize_weight(param.data)
                    # Get scale factor
                    scale = torch.abs(quant_weight[0][0]).item()
                    # Get binary value
                    quant_weight = quant_weight.apply_(lambda x : False if x <= 0.0 else True)
                    # To numpy array
                    quant_weight = quant_weight.numpy().astype(np.uint8)
                    params[name]['shape'] = quant_weight.shape
                    # Pack bits
                    quant_weight = np.packbits(quant_weight.flatten())
                    params[name]['param'] = quant_weight
                    params[name]['scale'] = scale
                else: # bn bias
                    params[name]['param'] = param.data.numpy()
                    params[name]['shape'] = params[name]['param'].shape

    #print(params)
    # fuse conv and bn
    fused_params = {}
    layer_nums = [0, 3, 4, 6, 7]
    for layer in layer_nums:
        group_parameters = {}
        for name, param in params.items():
            if 'layers' in name:
                layer_num = int(name.split('.')[1])
                if layer == layer_num:
                    if 'bn0' in name:
                        if not 'bn' in group_parameters:
                            group_parameters['bn'] = {}
                        if 'weight' in name:
                            group_parameters['bn']['alpha'] = param['param']
                            group_parameters['bn']['mean'] = param['mean']
                            group_parameters['bn']['var'] = param['var']
                            group_parameters['bn']['eps'] = param['eps']
                            group_parameters['bn']['bn_shape'] = param['shape']
                        elif 'bias' in name:
                            group_parameters['bn']['beta'] = param['param']
                    elif ('bn1' in name or 'conv0' in name) and layer_num != 7:
                        if not 'conv_bn' in group_parameters:
                            group_parameters['conv_bn'] = {}
                        if 'conv0' in name:
                            group_parameters['conv_bn']['scale'] = param['scale']
                            group_parameters['conv_bn']['weight'] = param['param']
                            group_parameters['conv_bn']['conv_shape'] = param['shape']
                        elif 'bn1' in name and 'weight' in name:
                            group_parameters['conv_bn']['mean'] = param['mean']
                            group_parameters['conv_bn']['var'] = param['var']
                            group_parameters['conv_bn']['eps'] = param['eps']
                            group_parameters['conv_bn']['alpha'] = param['param']
                            group_parameters['conv_bn']['bn_shape'] = param['shape']
                        elif 'bn1' in name and 'bias' in name:
                            group_parameters['conv_bn']['beta'] = param['param']
                    elif 'skip_' in name:
                        if not 'skip_conv_bn' in group_parameters:
                            group_parameters['skip_conv_bn'] = {}
                        if 'conv' in name:
                            group_parameters['skip_conv_bn']['scale'] = param['scale']
                            group_parameters['skip_conv_bn']['weight'] = param['param']
                            group_parameters['skip_conv_bn']['conv_shape'] = param['shape']
                        elif 'bn' in name and 'weight' in name:
                            group_parameters['skip_conv_bn']['mean'] = param['mean']
                            group_parameters['skip_conv_bn']['var'] = param['var']
                            group_parameters['skip_conv_bn']['eps'] = param['eps']
                            group_parameters['skip_conv_bn']['alpha'] = param['param']
                            group_parameters['skip_conv_bn']['bn_shape'] = param['shape']
                        elif 'bn' in name and 'bias' in name:
                            group_parameters['skip_conv_bn']['beta'] = param['param']
                    elif 'conv1'in name:
                        group_parameters['conv'] = {}
                        group_parameters['conv']['weight'] = param['param']
                        group_parameters['conv']['scale'] = param['scale']
                        group_parameters['conv']['conv_shape'] = param['shape']

        for gn in ['bn', 'conv_bn', 'skip_conv_bn', 'conv']:
            if gn in group_parameters:
                name = 'layers.{}.{}'.format(layer, gn)
                fused_params[name] = {}

                if gn != 'conv':
                    param = group_parameters[gn]
                    alpha = param['alpha'] / np.vectorize(math.sqrt)(param['var'] + param['eps'])
                    beta = param['beta'] - ((param['alpha'] * param['mean']) / np.vectorize(math.sqrt)(param['var'] + param['eps']))
                    fused_params[name]['alpha'] = alpha if gn == 'bn' else param['scale'] * alpha
                    fused_params[name]['beta'] = beta
                    if 'weight' in param:
                        fused_params[name]['weight'] = param['weight'] 
                    if 'bn_shape' in param:
                        fused_params[name]['bn_shape'] = param['bn_shape']
                    if 'conv_shape' in param:
                        fused_params[name]['conv_shape'] = param['conv_shape']
                else:
                    fused_params[name] = group_parameters[gn]

    # fuse outer bn
    group_parameters = {'bn' : {}, 'fc': {}}
    for name, param in params.items():
        if 'bn.bn_dict.1' in name:
            if 'weight' in name:
                group_parameters['bn']['alpha'] = param['param']
                group_parameters['bn']['mean'] = param['mean']
                group_parameters['bn']['var'] = param['var']
                group_parameters['bn']['eps'] = param['eps']
                group_parameters['bn']['bn_shape'] = param['shape']
            elif 'bias' in name:
                group_parameters['bn']['beta'] = param['param']
        elif 'fc' in name:
            if 'weight' in name:
                group_parameters['fc']['weight'] = param['param']
                group_parameters['fc']['weight_shape'] = param['shape']
            elif 'bias' in name:
                group_parameters['fc']['bias'] = param['param']
                group_parameters['fc']['bias_shape'] = param['shape']

    bn_param = group_parameters['bn']
    fused_params['outer_bn'] = {}
    fused_params['outer_bn']['alpha'] = bn_param['alpha'] / np.vectorize(math.sqrt)(bn_param['var'] + bn_param['eps'])
    fused_params['outer_bn']['beta'] = bn_param['beta'] - ((bn_param['alpha'] * bn_param['mean']) / np.vectorize(math.sqrt)(bn_param['var'] + bn_param['eps']))
    fused_params['outer_bn']['bn_shape'] = bn_param['bn_shape']
    fused_params['outer_fc'] = group_parameters['fc']

    # for name, param in fused_params.items():
    #     print(name)
    #     if 'alpha' in param:
    #         print('alpha', param['bn_shape'])
    #     if 'weight' in param and not 'fc' in name:
    #         print('weight', param['conv_shape'])
    #
    #     if name == 'outer_bn':
    #         print('alpha', param['bn_shape'])
    #     elif name == 'outer_fc':
    #         print(param)




    if args.save:
        with open('params.pkl', 'wb') as p:
            pickle.dump(fused_params, p, protocol = pickle.HIGHEST_PROTOCOL)
