import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
model = VGG16(weights="imagenet")
def predict_image(image_path):
    """Predict the class and probability of an image using the pre-trained VGG16 model."""
    # Load and preprocess the image
    orig = cv2.imread(image_path)
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # Make predictions and decode the results
    preds = model.predict(image)
    i = np.argmax(preds[0])
    decoded = imagenet_utils.decode_predictions(preds)
    (imagenetID, label, prob) = decoded[0][0]
    label = "{}: {:.2f}%".format(label, prob * 100)
    print(f"[IMG-DATA] {label}")

    return orig, i, label


class GradCAM:
    """Compute the Grad-CAM heatmap for a given image and class."""
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName or self.find_target_layer()

    def find_target_layer(self):
        """Find the target layer to compute the Grad-CAM heatmap."""
        for layer in reversed(self.model.layers):
            if len(layer.output.shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        """Compute the Grad-CAM heatmap for the given image."""
        gradModel = Model(inputs=self.model.inputs,
                           outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        grads = tape.gradient(loss, convOutputs)

        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """Overlay the heatmap on the input image."""
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        return heatmap, output

def visualize_output(orig, label, output):
    """Display the original image and the Grad-CAM output."""
    # Draw the predicted label on the output image
    cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the images
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title("Output")
    plt.axis('off')
    plt.show()

# Example usage
image_path = "img/test_7.jpg"
orig, i, label = predict_image(image_path)

cam = GradCAM(model, i)
heatmap = cam.compute_heatmap(np.expand_dims(img_to_array(load_img(image_path, target_size=(224, 224))), axis=0))
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
_, output = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

visualize_output(orig, label, output)