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
            dilations: tp.Iterable[int] = (6, 12, 18),
            use_bn: bool = True,
            separable: bool = True,
            kernel_initializer='glorot_uniform'):
        super().__init__()
        self._filters = filters
        self._dilations = dilations
        self._separable = separable
        self._kernel_initializer = kernel_initializer
        if use_bn:
            self.bns = [
                tf.keras.layers.BatchNormalization(center=False)
                for _ in self._dilations]

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
                for d in self._dilations]
        else:
            feature_maps = [
                tf.nn.conv2d(x, self.kernel, (1, 1), 'SAME', dilations=d)
                for d in self._dilations]
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
            dilations: tp.Iterable[int] = (6, 12, 18),
            use_bn: bool = True):
        super().__init__()
        self._filters = filters
        self.ksac_11 = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(filters, 1, (1, 1), use_bias=False)]
            + ([tf.keras.layers.BatchNormalization(center=False)]
                if use_bn else []))
        self.ksac_33 = _KSAC33(
            filters=filters, dilations=dilations, use_bn=use_bn)
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
        h0, h1, n_classes: int, filters: int, out_size, use_bn: bool = True):
    h0 = tf.keras.layers.Conv2D(filters, 1, (1, 1), use_bias=not use_bn)(h0)
    if use_bn:
        h0 = tf.keras.layers.BatchNormalization()(h0)
    h0 = tf.nn.relu(h0)
    h0 = tf.concat(
        [h0, tf.image.resize(images=h1, size=h0.shape[1:-1])], axis=-1)
    h0 = tf.keras.layers.SeparableConv2D(n_classes, 3, (1, 1), 'SAME')(h0)
    return tf.image.resize(images=h0, size=out_size)


def create_ksac_net(
        input_shape: tp.List[int],
        n_classes: int,
        strides: tp.List[int] = [4, 16],
        filters: tp.List[int] = [64, 128],
        dilations: tp.List[int] = [6, 12, 18],
        use_bn: bool = True,
        backbone='mobilenetv2'):
    assert len(strides) == len(filters) == 2
    pretrained = _BACKBONES[backbone]['net'](
        include_top=False, input_shape=input_shape)
    pretrained = tf.keras.Model(
        inputs=pretrained.inputs,
        outputs=[
            pretrained.get_layer(
                _BACKBONES[backbone][f'x{s}_layer_name']).output
            for s in strides
        ]
    )
    x = tf.keras.Input(input_shape, dtype=tf.float32)
    h = _BACKBONES[backbone]['preprocess'](x)
    h0, h1 = pretrained(h)
    h1 = KernelSharingAtrousConvolution(
        filters=filters[1], dilations=dilations, use_bn=use_bn)(h1)
    logits = deeplabv3_plus_decoder(
        h0, h1, n_classes, filters[0], input_shape[:-1], use_bn)
    return tf.keras.Model(inputs=[x], outputs=[logits])


if __name__ == '__main__':
    import argparse
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph
    )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input_shape', type=int, nargs=2, default=[512, 512],
        help='Input image shape')
    parser.add_argument(
        '--n_classes', type=int, default=20, help='Number of classes')
    parser.add_argument(
        '--dilations', type=int, nargs='*', default=[6, 12, 18],
        help='Dilation rates for each atrous convolution in KSAC module')
    parser.add_argument(
        '--strides', type=int, nargs=2, default=[4, 16],
        help='Output strides of feature maps from a backbone network')
    parser.add_argument(
        '--filters', type=int, nargs=2, default=[32, 64],
        help='Number of filters of KSAC module for each resolution')
    parser.add_argument(
        '--backbone', type=str, default=list(_BACKBONES)[0],
        choices=list(_BACKBONES), help='Backbone network')
    parser.add_argument(
        '--plot', type=str, default=None, help='Plot model and save png file')
    parser.add_argument(
        '--expand_nested', action='store_true',
        help='Expand nested model when plotting')
    args = parser.parse_args()

    model = create_ksac_net(
        args.input_shape + [3], args.n_classes, args.strides, args.filters,
        args.dilations, backbone=args.backbone)
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
    if args.plot is not None:
        if not args.plot.endswith('.png'):
            raise ValueError('Plotting functionality supports png format only')
        tf.keras.utils.plot_model(
            model, show_shapes=True, expand_nested=args.expand_nested)
