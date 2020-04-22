# -*- coding: utf-8 -*-
import argparse
import h5py
import numpy as np


# weights to int8
def w_quantize(x):
    min_weights = np.min(x)
    max_weights = np.max(x)
    weight_scale = (max_weights-min_weights) / 127.
    temp_dict = {}
    for frac_len in range(-4, 4):
        c = x / weight_scale
        a = np.floor(c * (2 ** frac_len) + 0.5)
        q_weights = a.astype(np.int8)
        d = c - q_weights / (2 ** frac_len)
        temp_dict[np.var(d)] = {'frac_len': frac_len, 'variance': np.var(d)}
    frac_len = temp_dict[min(temp_dict)]['frac_len']
    print(temp_dict[min(temp_dict)])
    c = x / weight_scale
    a = np.floor(c * (2 ** frac_len) + 0.5)
    q_weights = a.astype(np.int8)
    return q_weights, weight_scale, frac_len


# bias to float16
def b_quantize(x):
    min_bias = np.min(x)
    max_bias = np.max(x)
    bias_scale = (max_bias - min_bias) / 65535.
    temp_dict_b = {}
    for frac_len in range(-4, 4):
        c = x / bias_scale
        a = np.floor(c * (2 ** frac_len) + 0.5)
        q_bias = a.astype(np.float16)
        d = c - q_bias / (2 ** frac_len )
        temp_dict_b[np.var(d)] = {'frac_len': frac_len, 'variance': np.var(d)}
    frac_len = temp_dict_b[min(temp_dict_b)]['frac_len']
    print(temp_dict_b[min(temp_dict_b)])
    c = x / bias_scale
    a = np.floor(c * (2 ** frac_len) + 0.5)
    q_bias = a.astype(np.float16)
    return q_bias, bias_scale, frac_len


def q_weights(weight_file, output_weights):
    weights = h5py.File(weight_file, mode='r')
    q_weights = h5py.File(output_weights, mode='w')

    try:
        layers = weights.attrs['layer_names']
    except():
        raise ValueError("weights file must contain attribution: 'layer_names'")

    q_weights.attrs['layer_names'] = [name for name in weights.attrs['layer_names']]
    scales = []

    for layer_name in layers:
        f = q_weights.create_group(layer_name)
        g = weights[layer_name]

        f.attrs['weight_names'] = g.attrs['weight_names']
        for weight_name in g.attrs['weight_names']:
            print('{}, Start quantize!'.format(weight_name))
            weight_value = g[weight_name]
            name = str(weight_name)
            name = name.split(':')[0]
            name = name.split('_')[-2]

            if name == 'W':
                new_weight_value, weight_scale, frac_len = w_quantize(weight_value)
                scales.append(weight_scale)

                new_list = f.create_dataset(weight_name, new_weight_value.shape, dtype=np.float32)
                new_list[:] = new_weight_value

            else:
                new_bias_value, bias_scale, frac_len = b_quantize(weight_value)
                new_list = f.create_dataset(weight_name, new_bias_value.shape, dtype=np.float32)
                new_list[:] = weight_value

    q_weights.flush()
    q_weights.close()
    print('All quantize finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='quantize')
    parser.add_argument('--input-weights', type=str,
                        default='./weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

    parser.add_argument('--output-weights', type=str, default='./weights/vgg16_q_weights.h5')

    args = parser.parse_args()
    q_weights(args.input_weights, args.output_weights)
