import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.models import Model

def generate_gradcam(model, img_array, last_conv_layer_name='Conv_1', class_index=None, output_path='cam.jpg'):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Convert heatmap to image
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Load original image again
    img = img_array[0] * 255.0
    img = img.astype(np.uint8)

    # Superimpose
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Save and return path
    cv2.imwrite(output_path, superimposed_img)
    return output_path
