import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf

# Load the model at startup
model_path = os.path.join(os.path.dirname(__file__), '../model/final_mobilenet_fault_model.keras')
model = load_model(model_path)
print("✅ Model loaded successfully")

# Class labels (update these to match your model's classes)
CLASS_LABELS = [
    "Bird_dropping",
    "Crack",
    "Dust",
    "Normal",
    "Shadow"
]

def get_recommendation(fault_type):
    recommendations = {
        "Normal": "Your solar panel appears to be in good condition. Regular cleaning is recommended to maintain optimal performance.",
        "Crack": "Micro-cracks detected. Contact a solar technician for inspection and potential panel replacement.",
        "Shadow": "Partial shading detected. Consider trimming nearby vegetation or relocating objects causing shadows.",
        "Dust": "Significant dirt accumulation detected. Cleaning the panels will improve energy production.",
        "Bird_dropping": "Bird droppings detected — install anti-bird mesh or clean panels to restore optimal performance."
    }
    return recommendations.get(fault_type, "No specific recommendation available.")

def preprocess_image(img_path):
    """Load and preprocess image for model prediction"""
    img = image.load_img(img_path, target_size=(160, 160))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def get_last_conv_layer(model):
    """Find the last convolutional layer in the model"""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No convolutional layer found in the model")

def generate_gradcam(model, img_array, class_index=None, layer_name=None):
    """
    Generate Grad-CAM heatmap for a given image and model
    Returns both the heatmap and superimposed image
    """
    if layer_name is None:
        layer_name = get_last_conv_layer(model)
    
    # Create gradient model
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]
    
    # Compute guided gradients
    grads = tape.gradient(loss, conv_outputs)[0]
    guided_grads = tf.cast(conv_outputs > 0, "float32") * tf.cast(grads > 0, "float32") * grads
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    
    # Generate heatmap
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
    cam = cv2.resize(cam.numpy(), (160, 160))
    cam = np.maximum(cam, 0)
    heatmap = cam / cam.max()
    
    # Convert to RGB heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Create superimposed image
    img = img_array[0] * 255
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    return heatmap, superimposed_img

def predict_image(img_path):
    """Make prediction and generate visualization for a single image"""
    try:
        # Preprocess and predict
        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array)[0]
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))
        fault_type = CLASS_LABELS[predicted_class]

        # Generate Grad-CAM visualization
        heatmap_filename = f"gradcam_{os.path.basename(img_path)}"
        heatmap_filepath = os.path.join("uploads", heatmap_filename)

        _, superimposed_img = generate_gradcam(
            model,
            img_array,
            class_index=predicted_class
        )

        # Save Grad-CAM image to 'uploads' folder
        cv2.imwrite(heatmap_filepath, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

        return {
            'fault_type': fault_type,
            'confidence': confidence,
            'recommendation': get_recommendation(fault_type),
            'heatmap_path': heatmap_filepath  # Return the full path for backend processing
        }
    except Exception as e:
        return {
            'error': str(e)
        }

def batch_predict(images_folder):
    """Process multiple images in a folder"""
    results = []
    for img_file in os.listdir(images_folder):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue
            
        img_path = os.path.join(images_folder, img_file)
        try:
            # Preprocess and predict
            img_array = preprocess_image(img_path)
            prediction = model.predict(img_array)[0]
            predicted_class = CLASS_LABELS[np.argmax(prediction)]
            confidence = float(np.max(prediction))
            
            # Generate Grad-CAM visualization
            heatmap_filename = f"gradcam_{img_file}"
            heatmap_path = os.path.join(images_folder, heatmap_filename)
            
            _, superimposed_img = generate_gradcam(
                model,
                img_array,
                class_index=np.argmax(prediction)
            )
            
            # Save visualization
            cv2.imwrite(heatmap_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
            
            results.append({
                'filename': img_file,
                'status': 'Normal' if predicted_class == "Normal" else 'Warning',
                'fault_type': predicted_class,
                'confidence': confidence,
                'recommendation': get_recommendation(predicted_class),
                'heatmap_path': heatmap_path
            })
            
        except Exception as e:
            results.append({
                'filename': img_file,
                'error': str(e)
            })
    
    return results