import argparse
import glob
import os
import typing as tp

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ksac import create_ksac_net, _BACKBONES


def load_xy(x_path: str, y_path: str, n_classes: int, size=None):
    x = tf.io.read_file(x_path)
    x = tf.image.decode_jpeg(x, channels=3)
    x = tf.image.convert_image_dtype(x, tf.uint8)
    y = tf.io.read_file(y_path)
    y = tf.image.decode_png(y, channels=1)
    y = tf.where(y >= n_classes, n_classes - 1, y)
    y = tf.image.convert_image_dtype(y, tf.int64)
    if size is not None:
        x = tf.image.resize(x, size)
        y = tf.image.resize(y, size, method='nearest')
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)
    y = tf.squeeze(y, [-1])
    return x, y


def load_minibatch(
        x_paths: tp.Iterable[str],
        y_paths: tp.Iterable[str],
        n_classes: int,
        size=None):
    x_batch = []
    y_batch = []
    for x_path, y_path in zip(x_paths, y_paths):
        x, y = load_xy(x_path, y_path, n_classes, size)
        x_batch.append(x)
        y_batch.append(y)
    return tf.stack(x_batch), tf.stack(y_batch)


def focal_loss(labels, logits, gamma: float = 2.0):
    q = tf.one_hot(labels, logits.shape[-1])
    lnp = tf.nn.log_softmax(logits)
    p = tf.math.exp(lnp)
    return -tf.reduce_sum(
        q * lnp * tf.clip_by_value((1 - p), 1e-8, 1) ** gamma, -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--images', type=str, help='Directory of training images')
    parser.add_argument(
        '--labels', type=str, help='Directory of training labels')
    parser.add_argument(
        '--outputs', type=str, help='Directory to save trained models')
    parser.add_argument(
        '--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument(
        '--batch', type=int, default=32, help='Minibatch size')
    parser.add_argument(
        '--input_shape', type=int, nargs=2, default=[480, 480],
        help='Input image shape')
    parser.add_argument(
        '--n_classes', type=int, default=20, help='Number of classes')
    parser.add_argument(
        '--filters', type=int, default=128, help='Number of filters')
    parser.add_argument(
        '--dilations', type=int, default=[6, 12, 18], nargs='*',
        help='Dilation rates')
    parser.add_argument(
        '--backbone', type=str, default=list(_BACKBONES)[0],
        choices=list(_BACKBONES), help='Backbone network')
    args = parser.parse_args()

    x_list = glob.glob(os.path.join(args.images, '*.jpg'))
    x_list.sort()
    x_list = np.array(x_list)
    y_list = glob.glob(os.path.join(args.labels, '*.png'))
    y_list.sort()
    y_list = np.array(y_list)
    assert len(x_list) == len(y_list)

    model = create_ksac_net(
        args.input_shape + [3], args.n_classes, args.filters,
        dilation_rates=args.dilations, backbone=args.backbone)
    optimizer = tf.optimizers.Adam()

    running_accuracy = 0
    running_loss = 0
    for e in range(1, args.epochs + 1):
        indices = np.random.permutation(len(x_list))
        pbar = tqdm(range(0, len(x_list), args.batch))
        for i in pbar:
            x_batch = x_list[indices[i: i + args.batch]]
            y_batch = y_list[indices[i: i + args.batch]]
            x_batch, y_batch = load_minibatch(
                x_batch, y_batch, args.n_classes, args.input_shape)
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss = tf.reduce_mean(focal_loss(y_batch, logits))
            running_loss = running_loss * 0.99 + loss * 0.01
            accuracy = np.mean(
                np.argmax(logits.numpy(), -1) == y_batch.numpy())
            running_accuracy = running_accuracy * 0.99 + accuracy * 0.01
            if i % (10 * args.batch) == 0:
                pbar.set_description(
                    f'Epoch={e:3d}, Accuracy={int(accuracy * 100):3d}% ',
                    f'Loss={running_loss:g}')
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        model.save(os.path.join(args.outputs, f'epoch_{e:03d}'))
