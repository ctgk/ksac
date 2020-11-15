import typing as tp

import tensorflow as tf


__all__ = ['create_ksac_net', 'KernelSharingAtrousConvolution']

_BACKBONES = {
    'mobilenetv2': {
        'preprocess': tf.keras.applications.mobilenet_v2.preprocess_input,
        'net': tf.keras.applications.MobileNetV2,
        'x2_layer_name': 'block_1_expand_relu',
        'x4_layer_name': 'block_3_expand_relu',
        'x8_layer_name': 'block_6_expand_relu',
        'x16_layer_name': 'block_13_expand_relu',
    },
    'xception': {
        'preprocess': tf.keras.applications.xception.preprocess_input,
        'net': tf.keras.applications.Xception,
        'x4_layer_name': 'block4_sepconv2_bn',
        'x16_layer_name': 'block13_sepconv2_bn',
    },
    'efficientnetb0': {
        'preprocess': tf.keras.applications.efficientnet.preprocess_input,
        'net': tf.keras.applications.EfficientNetB0,
        'x4_layer_name': 'block3a_expand_activation',
        'x16_layer_name': 'block6a_expand_activation',
    }
}


class _KSAC33(tf.keras.layers.Layer):

    def __init__(
            self,
            filters: int,
            dilation_rates: tp.Iterable[int] = (6, 12, 18),
            use_bn: bool = True,
            separable: bool = True,
            kernel_initializer='glorot_uniform'):
        super().__init__()
        self._filters = filters
        self._dilation_rates = dilation_rates
        self._separable = separable
        self._kernel_initializer = kernel_initializer
        if use_bn:
            self.bns = [
                tf.keras.layers.BatchNormalization(center=False)
                for _ in self._dilation_rates]

    def build(self, input_shape):
        k_init = eval('tf.keras.initializers.' + self._kernel_initializer)()
        if self._separable:
            d_shape = (3, 3, input_shape[-1], 1)
            p_shape = (1, 1, input_shape[-1], self._filters)
            self.depthwise_kernel = tf.Variable(
                k_init(d_shape), trainable=True, name='depthwise_kernel')
            self.pointwise_kernel = tf.Variable(
                k_init(p_shape), trainable=True, name='pointwise_kernel')
        else:
            k_shape = (3, 3, input_shape[-1], self._filters)
            self.kernel = tf.Variable(
                k_init(k_shape), trainable=True, name='kernel')
        if hasattr(self, 'bns'):
            for bn in self.bns:
                bn.build(input_shape[:-1] + [self._filters])

    def call(self, x, training: bool = False):
        if self._separable:
            feature_maps = [
                tf.nn.separable_conv2d(
                    x, self.depthwise_kernel, self.pointwise_kernel, [1] * 4,
                    'SAME', dilations=[d, d])
                for d in self._dilation_rates]
        else:
            feature_maps = [
                tf.nn.conv2d(x, self.kernel, (1, 1), 'SAME', dilations=d)
                for d in self._dilation_rates]
        if hasattr(self, 'bns'):
            feature_maps = [
                bn(h, training=training)
                for h, bn in zip(feature_maps, self.bns)]
        return sum(feature_maps)


class _KSACPooling(tf.keras.layers.Layer):

    def __init__(
            self,
            filters: int,
            use_bn: bool = True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'):
        super().__init__()
        self._filters = filters
        self.conv = tf.keras.layers.Conv2D(filters, 1, (1, 1), use_bias=False)
        if use_bn:
            self.bn = tf.keras.layers.BatchNormalization(center=False)

    def build(self, input_shape):
        self.conv.build([input_shape[0], 1, 1, input_shape[-1]])
        if hasattr(self, 'bn'):
            self.bn.build([input_shape[0], 1, 1, self._filters])

    def call(self, x, training: bool = False):
        x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = self.conv(x)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        return tf.image.resize(images=x, size=x.shape[1:-1])


class KernelSharingAtrousConvolution(tf.keras.layers.Layer):
    
    def __init__(
            self, 
            filters: int, 
            dilation_rates: tp.Iterable[int] = (6, 12, 18), 
            use_bn: bool = True):
        super().__init__()
        self._filters = filters
        self.ksac_11 = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(filters, 1, (1, 1), use_bias=False)]
            + ([tf.keras.layers.BatchNormalization(center=False)]
                if use_bn else []))
        self.ksac_33 = _KSAC33(
            filters=filters, dilation_rates=dilation_rates, use_bn=use_bn)
        self.ksac_pool = _KSACPooling(filters=filters, use_bn=use_bn)

    def build(self, input_shape):
        self.ksac_11.build(input_shape)
        self.ksac_33.build(input_shape)
        self.ksac_pool.build(input_shape)
        init = tf.zeros_initializer()
        self.bias = tf.Variable(
            init((self._filters,)), trainable=True, name='bias')

    def call(self, x, training: bool = False):
        return tf.nn.relu(
            self.ksac_11(x, training=training)
            + self.ksac_33(x, training=training)
            + self.ksac_pool(x, training=training)
            + self.bias)


def deeplabv3_plus_decoder(
        h4, h16, n_classes: int, filters: int, out_size, use_bn: bool = True):
    h4 = tf.keras.layers.Conv2D(filters, 1, (1, 1), use_bias=not use_bn)(h4)
    if use_bn:
        h4 = tf.keras.layers.BatchNormalization()(h4)
    h4 = tf.nn.relu(h4)
    h4 = tf.concat(
        [h4, tf.image.resize(images=h16, size=h4.shape[1:-1])], axis=-1)
    h4 = tf.keras.layers.Conv2D(n_classes, 3, (1, 1), 'SAME')(h4)
    return tf.image.resize(images=h4, size=out_size)


def decoder(hr: list, n_classes: int, out_size):
    h = tf.concat(
        [hr[0]]
        + [tf.image.resize(images=h, size=hr[0].shape[1:-1]) for h in hr[1:]],
        axis=-1)
    h = tf.keras.layers.Conv2D(n_classes, 3, (1, 1), 'SAME')(h)
    return tf.image.resize(images=h, size=out_size)


def create_ksac_net(
        input_shape: tp.List[int],
        n_classes: int,
        resolutions: tp.List[int] = [2, 4],
        filters: tp.List[int] = [64, 128],
        dilation_rates: tp.List[int] = [6, 12, 18],
        use_bn: bool = True,
        backbone='mobilenetv2'):
    pretrained = _BACKBONES[backbone]['net'](
        include_top=False, input_shape=input_shape)
    pretrained = tf.keras.Model(
        inputs=pretrained.inputs,
        outputs=[
            pretrained.get_layer(
                _BACKBONES[backbone][f'x{r}_layer_name']).output
            for r in resolutions
        ]
    )
    x = tf.keras.Input(input_shape, dtype=tf.float32)
    h = _BACKBONES[backbone]['preprocess'](x)
    hr = pretrained(h)
    hr = [
        KernelSharingAtrousConvolution(
            filters=f, dilation_rates=dilation_rates, use_bn=use_bn)(h)
        for f, h in zip(filters, hr)
    ]
    logits = decoder(hr, n_classes, out_size=input_shape[:-1])
    return tf.keras.Model(inputs=[x], outputs=[logits])


if __name__ == '__main__':
    import argparse
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph
    )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input_shape', type=int, nargs=2, default=[240, 320],
        help='Input image shape')
    parser.add_argument(
        '--n_classes', type=int, default=20, help='Number of classes')
    parser.add_argument(
        '--dilation_rates', type=int, nargs='*', default=[6, 12, 18],
        help='Dilation rate for each atrous convolution in KSAC module')
    parser.add_argument(
        '--resolutions', type=int, nargs='*', default=[4, 16],
        help='Resolutions of feature maps from a backbone network')
    parser.add_argument(
        '--filters', type=int, nargs='*', default=[32, 64],
        help='Number of filters of KSAC module for each resolution')
    parser.add_argument(
        '--backbone', type=str, default=list(_BACKBONES)[0],
        choices=list(_BACKBONES), help='Backbone network')
    args = parser.parse_args()

    model = create_ksac_net(
        args.input_shape + [3], args.n_classes, args.resolutions, args.filters,
        args.dilation_rates, backbone=args.backbone)
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1] + args.input_shape + [3])])
    _, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=graph, run_meta=run_meta, cmd='op', options=opts)
    print(model.summary())
    print(f'{flops.total_float_ops:,} FLOPs')
    tf.keras.utils.plot_model(model, show_shapes=True)
    print(model.outputs[0].shape)
