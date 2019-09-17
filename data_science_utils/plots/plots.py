import numpy as np
import cv2
import sys
import requests
import os
import matplotlib.pyplot as 
import matplotlib.gridspec as gridspec

from io import BytesIO
from PIL import Image


#Function extracts .png images path from given directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
        #   img = image.load_img('images/'+filename, target_size=(224, 224))
        #   img= cv2.resize((224,224))
        #   img = cv2.imread(os.path.join(folder, filename))
        #   if img is not None:
                images.append(folder+'/'+filename)
    return images


#show the images as columns wise, by default columns is 1, required images is numpy array
def show_images(images, cols = 1, titles = None):
    """Display a list of images(numpy array) in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    
#images path 
def superimposed_images(img_path, model):
#     img_path = 'images/airplane.png'
    img = image.load_img(img_path, target_size=(32, 32))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer("activation_13")#change layer according to model
    
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(9):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    img = cv2.imread(img_path)
    # img = Image.open(BytesIO(response.content))
    # img = img.resize((224, 224))

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return superimposed_img
    

from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(2017)

import cv2
import random

import matplotlib.gridspec as gridspec

from keras.preprocessing import image
import numpy as np


def plot_loss_acc(history):
    plt.figure(figsize=(20,7))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'][1:])    
    plt.plot(history.history['val_loss'][1:])    
    plt.title('model loss')    
    plt.ylabel('val_loss')    
    plt.xlabel('epoch')    
    plt.legend(['Train','Validation'], loc='upper left')
    
    plt.subplot(1,2,2)
    plt.plot(history.history['acc'][1:])
    plt.plot(history.history['val_acc'][1:])
    plt.title('Model Accuracy')
    plt.ylabel('val_acc')
    plt.xlabel('epoch')
    plt.legend(['Train','Validation'], loc='upper left')
    plt.show()


def min_max_scale(X):
  return (X - np.min(X))/(np.max(X)-np.min(X))


def gradcam(model, layer, img, class_idx, preprocess_func=None, preprocess_img=min_max_scale,
            show=False):
    x = np.expand_dims(image.img_to_array(img), axis=0)
    img = np.copy(img)
    class_idx = np.argmax(class_idx, axis=0) if type(class_idx) == list or type(class_idx) == np.ndarray else class_idx
    if preprocess_func is not None:
        x = preprocess_func(x)
    if preprocess_img is not None:
        img = preprocess_img(img)
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)[0]

    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer(layer)
    layer_out_channels = last_conv_layer.output_shape[-1]

    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(layer_out_channels):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    heatmap = heatmap / 255
    for i in range(len(heatmap)):
        for j in range(len(heatmap[0])):
            if heatmap[i][j][1] <= 0.01 and heatmap[i][j][2] <= 0.01:
                heatmap[i][j] = 0

    superimposed_img = 0.6 * img + 0.4 * heatmap
    for i in range(len(heatmap)):
        for j in range(len(heatmap[0])):
            if np.sum(heatmap[i][j]) == 0:
                superimposed_img[i][j] = img[i][j]

    superimposed_img = np.clip(superimposed_img, 0, 1, )
    if show:
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        plt.imshow(heatmap)
        plt.axis("off")
        plt.show()
        plt.imshow(superimposed_img)
        plt.axis("off")
        plt.show()
    return img, heatmap, superimposed_img, preds


def show_examples_with_gradcam(model, layer, images, labels, classes=None, preprocess_func=None, preprocess_img=min_max_scale,
                               image_size_multiplier=3,
                               show_actual=True, show_heatmap=False, show_superimposed=True):
    columns = 5
    rows = int(np.ceil(len(images) / columns))
    num_inner_rows = int(show_actual + show_heatmap + show_superimposed)
    labels = np.argmax(labels, axis=1) if type(labels[0]) == list or type(labels[0]) == np.ndarray else labels
    fig_height = rows * image_size_multiplier * num_inner_rows
    fig_width = columns * image_size_multiplier
    fig = plt.figure(figsize=(fig_width, fig_height))
    outer = gridspec.GridSpec(rows, columns, wspace=0.0, hspace=0.2)
    for i in range(rows * columns):
        if i >= len(images):
            break
        x = images[i]
        y = labels[i]
        img, heatmap, superimposed_img, prediction = gradcam(model, layer, x, y, preprocess_func=preprocess_func,
                                                             preprocess_img=preprocess_img, show=False)
        inner = gridspec.GridSpecFromSubplotSpec(num_inner_rows, 1,
                                                 subplot_spec=outer[i], wspace=0.0, hspace=0.05)

        imgs = []
        if show_actual:
            imgs.append(img)
        if show_heatmap:
            imgs.append(heatmap)
        if show_superimposed:
            imgs.append(superimposed_img)
        label = classes[y] if classes is not None else ""
        label = label.split(' ', 1)[0]
        prediction = classes[prediction]
        titles = [("Actual:" + label + " Pred:" + prediction).replace(' ', '\n')]
        for j in range(num_inner_rows):
            ax = plt.Subplot(fig, inner[j])
            ax.imshow(imgs[j])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(titles.pop() if len(titles) > 0 else "")
            fig.add_subplot(ax)

    fig.show()


def show_layers_with_gradcam(model, layers, image, label, classes=None, preprocess_func=None, preprocess_img=min_max_scale,
                             image_size_multiplier=3,
                             show_actual=True, show_heatmap=False, show_superimposed=True):
    columns = 5
    rows = int(np.ceil(len(layers) / columns))
    num_inner_rows = int(show_actual + show_heatmap + show_superimposed)
    label = np.argmax(label, axis=0) if type(label) == list or type(label) == np.ndarray else label
    fig_height = rows * image_size_multiplier * num_inner_rows
    fig_width = columns * image_size_multiplier
    fig = plt.figure(figsize=(fig_width, fig_height))
    outer = gridspec.GridSpec(rows, columns, wspace=0.0, hspace=0.2)
    for i in range(rows * columns):
        if i >= len(layers):
            break
        layer = layers[i]
        img, heatmap, superimposed_img, prediction = gradcam(model, layer, image, label,
                                                             preprocess_func=preprocess_func,
                                                             preprocess_img=preprocess_img, show=False)
        inner = gridspec.GridSpecFromSubplotSpec(num_inner_rows, 1,
                                                 subplot_spec=outer[i], wspace=0.0, hspace=0.05)
        imgs = []
        if show_actual:
            imgs.append(img)
        if show_heatmap:
            imgs.append(heatmap)
        if show_superimposed:
            imgs.append(superimposed_img)
        lbl = classes[label] if classes is not None else ""
        lbl = lbl.split(' ', 1)[0]
        prediction = classes[prediction]
        titles = [("Actual:" + label + " Pred:" + prediction).replace(' ', '\n')]
        for j in range(num_inner_rows):
            ax = plt.Subplot(fig, inner[j])
            ax.imshow(imgs[j])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(titles.pop() if len(titles) > 0 else "")
            fig.add_subplot(ax)

    fig.show()


def find_misclassified(X, y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    misclassified = y_true != y_pred
    X = X[misclassified]
    y_true = y_true[misclassified]
    y_pred = y_pred[misclassified]
    return X, y_true, y_pred


def find_correct(X, y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    misclassified = y_true == y_pred
    X = X[misclassified]
    y_true = y_true[misclassified]
    y_pred = y_pred[misclassified]
    return X, y_true, y_pred


def show_misclassified_with_gradcam(model, layer, iterator, classes=None, preprocess_func=None, preprocess_img=min_max_scale,
                                    image_size_multiplier=3, examples=25,
                                    show_actual=True, show_heatmap=False, show_superimposed=True):
    columns = 5
    rows = int(np.ceil(examples / columns))
    images = []
    labels = []
    predictions = []
    while len(images) < examples:
        batchX, batchY = iterator.next()
        preds = model.predict(batchX)
        X, y_true, y_pred = find_misclassified(batchX, batchY, preds)
        images.extend(X)
        labels.extend(y_true)
        predictions.extend(y_pred)

    images = images[:examples]
    labels = labels[:examples]
    predictions = predictions[:examples]

    show_examples_with_gradcam(model, layer, images, labels, classes=classes, preprocess_func=preprocess_func,
                               preprocess_img=preprocess_img, image_size_multiplier=image_size_multiplier)


def show_correct_with_gradcam(model, layer, iterator, classes=None, preprocess_func=None, preprocess_img=min_max_scale,
                              image_size_multiplier=3, examples=25,
                              show_actual=True, show_heatmap=False, show_superimposed=True):
    columns = 5
    rows = int(np.ceil(examples / columns))
    images = []
    labels = []
    predictions = []
    while len(images) < examples:
        batchX, batchY = iterator.next()
        preds = model.predict(batchX)
        X, y_true, y_pred = find_correct(batchX, batchY, preds)
        images.extend(X)
        labels.extend(y_true)
        predictions.extend(y_pred)

    images = images[:examples]
    labels = labels[:examples]
    predictions = predictions[:examples]

    show_examples_with_gradcam(model, layer, images, labels, classes=classes, preprocess_func=preprocess_func,
                               preprocess_img=preprocess_img, image_size_multiplier=image_size_multiplier)
