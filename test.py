import numpy as np
import pickle
import torch
import argparse

import models

def load_data(params_path, data_path, short=False):
    with open(params_path, "rb") as f:
        params = pickle.load(f)

    with open(data_path, "rb") as f:
        img, target = pickle.load(f)

    if short:
        b,c,h,w = img.shape
        img = img[0, :, :, :].numpy().reshape(1,c,h,w)

    return params, img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default = 'params.pkl')
    parser.add_argument('--data', default = 'testset/data.pkl')
    args = parser.parse_args()

    params, img = load_data(args.param, args.data, True)

    np_model = models.__dict__['resnet20q_np']([1], params, 10)
    np_out = np_model.forward(img)

    print(np_out)

