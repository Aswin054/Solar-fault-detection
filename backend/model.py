from tensorflow.keras.models import load_model

print("Starting model conversion...")

# 1. Load original model
model = load_model('model/final_mobilenet_fault_model.keras')
print("✅ Model loaded successfully")

# 2. Save in Keras 3 format (creates .keras file)
model.save('converted_model.keras')  # Keras 3 native format
print("✅ Model saved as converted_model.keras")

# Alternative: Save as SavedModel format (directory)
model.export('converted_model')  # Creates TF SavedModel format
print("✅ Model exported in TF SavedModel format to 'converted_model/'")