import glob
import os
import typing as tp

import numpy as np
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf
from tqdm import tqdm


def load_xy(
        x_path: str,
        y_path: str,
        n_classes: int,
        crop_to=None,
        resize_to=None):
    x = imread(x_path)
    y = imread(y_path)
    y[y >= n_classes] = n_classes - 1
    if crop_to is not None:
        h = np.random.randint(0, x.shape[0] - crop_to[0])
        w = np.random.randint(0, y.shape[1] - crop_to[1])
        x = x[h: y.shape[0] - h, w: y.shape[1] - w]
        y = y[h: y.shape[0] - h, w: y.shape[1] - w]
        assert x.shape[:2] == y.shape == tuple(crop_to)
    if resize_to is not None:
        x = resize(
            x, resize_to, order=1, preseve_range=True).astype(np.float32)
        y = resize(
            y, resize_to, order=0, preseve_range=True).astype(np.int)
    if np.random.uniform() > 0.5:
        x = x[:, ::-1]
        y = y[:, ::-1]
    return x, y


def load_minibatch(
        x_paths: tp.Iterable[str],
        y_paths: tp.Iterable[str],
        n_classes: int,
        crop_to=None,
        resize_to=None):
    x_batch = []
    y_batch = []
    for x_path, y_path in zip(x_paths, y_paths):
        x, y = load_xy(x_path, y_path, n_classes, crop_to, resize_to)
        x_batch.append(x)
        y_batch.append(y)
    return np.asarray(x_batch), np.asarray(y_batch)


def focal_loss(labels, logits, gamma: float = 2.0):
    q = tf.one_hot(labels, logits.shape[-1])
    lnp = tf.nn.log_softmax(logits)
    p = tf.math.exp(lnp)
    return -tf.reduce_sum(
        q * lnp * tf.clip_by_value((1 - p), 1e-8, 1) ** gamma, -1)


def train_model(
        model: tf.keras.Model,
        x_train_dir: str,
        y_train_dir: str,
        output_dir: str,
        epochs: int,
        batchsize: int,
        *,
        crop_to=None,
        progress_bar: tqdm = None,
        description_template='Epoch={:3d}, Acc={:3d}%, Loss={:6.04f}'):
    input_shape = model.inputs[0].shape
    n_classes = model.outputs[0].shape[-1]
    x_train = glob.glob(os.path.join(x_train_dir, '*.jpg'))
    x_train.sort()
    y_train = glob.glob(os.path.join(y_train_dir, '*.png'))
    y_train.sort()
    assert len(x_train) == len(y_train)
    x_train, y_train = np.array(x_train), np.array(y_train)

    running_accuracy = 0
    x, y = load_minibatch(
        x_train[:batchsize], y_train[:batchsize],
        n_classes=n_classes, crop_to=crop_to, resize_to=input_shape[1:-1])
    running_loss = np.mean(focal_loss(y, model.predict(x)))
    optimizer = tf.optimizers.Adam()
    for e in range(1, epochs + 1):
        indices = np.random.permutation(len(x_train))
        iterator = range(0, len(x_train), batchsize)
        if progress_bar is not None:
            iterator = progress_bar(iterator)
        for i in pbar:
            x, y = load_minibatch(
                x_train[indices[i: i + batchsize]],
                y_train[indices[i: i + batchsize]],
                n_classes=n_classes, size=input_shape[1:-1])
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss = tf.reduce_mean(focal_loss(y, logits))
            running_loss = running_loss * 0.99 + loss * 0.01
            accuracy = np.mean(
                np.argmax(logits.numpy(), -1) == y_batch.numpy())
            running_accuracy = running_accuracy * 0.99 + accuracy * 0.01
            if i % (10 * batchsize) == 0 and progress_bar is not None:
                iterator.set_description(
                    description_template.format(
                        e, int(running_accuracy * 100), running_loss))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        os.makedirs(output_dir, exist_ok=True)
        model.save(os.path.join(output_dir, f'epoch_{e:03d}'))


if __name__ == '__main__':
    import argparse
    from model import create_ksac_net, _BACKBONES

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
        '--batchsize', type=int, default=32, help='Minibatch size')
    parser.add_argument(
        '--input_shape', type=int, nargs=2, default=[512, 512],
        help='Input image shape')
    parser.add_argument(
        '--n_classes', type=int, default=20, help='Number of classes')
    parser.add_argument(
        '--dilations', type=int, default=[6, 12, 18], nargs='*',
        help='Dilation rates')
    parser.add_argument(
        '--strides', type=int, nargs=2, default=[4, 16],
        help='Output strides of feature maps from a backbone network')
    parser.add_argument(
        '--filters', type=int, nargs=2, default=[32, 64],
        help='Number of filters of KSAC module for each resolution')
    parser.add_argument(
        '--crop_to', type=int, nargs=2, default=None,
        help='Crop image to given size then resize if given')
    parser.add_argument(
        '--backbone', type=str, default=list(_BACKBONES)[0],
        choices=list(_BACKBONES), help='Backbone network')
    args = parser.parse_args()

    model = create_ksac_net(
        args.input_shape + [3], args.n_classes, args.filters,
        dilation_rates=args.dilations, backbone=args.backbone)
    train_model(
        model, args.images, args.labels, args.outputs, args.epochs,
        args.batchsize, crop_to=args.crop_to, progress_bar=tqdm)
