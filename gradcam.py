import os
import sys
import cv2
import keras
import keras.backend as K
from keras.layers.core import Lambda

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.framework import ops

# Load few images:
n = 25
data = np.load('../simu/synthetic_imgs/test.npz')

few = np.random.choice(data['test_x'].shape[0], n, replace=False)
test_x  = np.transpose(data['test_x'][few], (0,2,3,1))
test_y  = data['test_y'][few]

plt.imshow(test_x[16])
_ = plt.hist(test_x[0].flatten(), 55)

### Grad-CAM ###
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def target_loss_out_shape(input_shape):
    return input_shape

def target_loss(x, risk):
    if risk > 0.5:
        return x
    if risk < -0.5:
        return -x
    else:
        return tf.matrix_inverse(x) * tf.sign(risk)

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = keras.models.load_model('model.h5')
    return new_model

def compile_saliency_function(model):
    input_img = model.input
    layer_output = model.layers[7].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_cam(input_model, img, prediction):
    model = input_model

    target_layer = lambda x: target_loss(x, prediction)
    model.add(Lambda(target_layer, output_shape = target_loss_out_shape))

    loss = K.sum(model.layers[-1].output)
    conv_output =  model.layers[7].output

    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_func = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_func([ img ])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (111, 111))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    image = img[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    cam = np.uint8(cam)

    return cam, heatmap


# Load a model & obtain predictions:
model = keras.models.load_model('model.h5')
model.model.summary()

preds = model.predict(test_x)
plt.plot(test_y[:,0], preds, '.')

# Pick an image:
img_index = 15 # 19
img = test_x[img_index:img_index+1]

cam, heatmap = grad_cam(model, img, preds[img_index])

plt.imshow(img[0])
plt.imshow(cam)


register_gradient()
guided_model = modify_backprop(model, 'GuidedBackProp')
saliency_fn = compile_saliency_function(guided_model)
saliency = saliency_fn([img, 0])
gradcam = saliency[0] * heatmap[..., np.newaxis]
gradcam = deprocess_image(gradcam)

plt.imshow(gradcam)

#########################################################################
