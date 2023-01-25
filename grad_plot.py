import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model


class GradCAM:
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name
        if self.layer_name is None:
            self.layer_name = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name

    def compute_heatmap(self, image, class_idx, eps=1e-8):
        grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output,
                     self.model.output])
        grad_model.layers[-1].activation = tf.keras.activations.linear
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (conv_outputs, predictions) = grad_model(inputs)
            class_predictions = predictions[:, class_idx]

        grads = tape.gradient(class_predictions, conv_outputs)

        cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
        cast_grads = tf.cast(grads > 0, "float32")
        guided_grads = cast_conv_outputs * cast_grads * grads

        conv_outputs = conv_outputs[0]
        guided_grads = guided_grads[0]

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

        (w, h) = (image.shape[1], image.shape[2])
        # (w, h) = (150, 150)
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        # heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
        heatmap = cv2.resize(heatmap, (150, 150))
        heatmap = cv2.applyColorMap(heatmap, colormap)

        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        return heatmap, output


def plot_GradCam(cam, model, image, true_label=None, name=None, uncertainty=False, dataset='Xray'):
    image = image[np.newaxis, :, :, :]
    all_heatmap = []
    all_preds = []
    if uncertainty:
        mc_iter = 200
    else:
        mc_iter = 1

    if dataset == 'Xray':
        class_text = ['Covid19', 'Normal', 'Pneumonia']
    elif dataset == 'CT':
        class_text = ['nCT', 'NiCT', 'pCT']

    for _ in range(mc_iter):
        preds = model.predict(image)
        class_idx = np.argmax(preds[0])

        heatmap_p = cam.compute_heatmap(image, class_idx)
        all_heatmap.append(heatmap_p)
        all_preds.append(preds)

    preds_mean = np.mean(all_preds, axis=0)
    class_idx_mean = np.argmax(preds_mean[0])

    heatmap_mean = np.mean(all_heatmap, axis=0)

    heatmap_mean = heatmap_mean.astype("uint8")

    image = (image + 1) / 2
    image1 = (image * 255).astype("uint8")
    image1 = np.squeeze(image1, axis=0)
    image1 = np.squeeze(image1, axis=2)
    image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)

    (heatmap_mean, output_p) = cam.overlay_heatmap(heatmap_mean, image1, alpha=0.65,
                                                   colormap=cv2.COLORMAP_JET)

    print(preds_mean)
    fig = plt.figure(figsize=[15, 8])

    plt.subplot(3, 1, 1)
    plt.imshow(image1)
    title = 'True: {} \n Prediction: {} ({}%)'.format(class_text[true_label],
                                                      class_text[class_idx_mean],
                                                      int(preds_mean[0][class_idx_mean] * 100))
    plt.title(title, fontsize=16)
    plt.axis('off')

    plt.subplot(3, 1, 2)
    plt.imshow(np.flip(heatmap_mean, axis=2))
    plt.axis('off')

    plt.subplot(3, 1, 3)
    plt.imshow(np.flip(output_p, axis=2))
    plt.axis('off')

    fig.savefig('{}.pdf'.format(name), dpi=300)
    plt.show()
