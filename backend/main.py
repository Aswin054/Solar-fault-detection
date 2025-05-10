from werkzeug.utils import secure_filename
import os
import io
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Image as RLImage, Table
from reportlab.lib.styles import getSampleStyleSheet
from typing import List, Optional
from pathlib import Path

app = FastAPI()

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.chmod(UPLOAD_FOLDER, 0o755)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path handling for different deployment scenarios
model_dir = os.path.join(os.path.dirname(__file__), 'model')

# Try loading in this order:
# 1. First attempt to load the converted SavedModel format
# 2. Fall back to .keras format
# 3. Fall back to .h5 format (if you have older models)

model = None
load_attempts = [
    os.path.join(model_dir, 'converted_model'),  # SavedModel directory
    os.path.join(model_dir, 'converted_model.keras'),  # Keras 3 format
    os.path.join(model_dir, 'final_mobilenet_fault_model.keras'),  # Original
    os.path.join(model_dir, 'final_mobilenet_fault_model.h5')  # Legacy format
]

for attempt in load_attempts:
    try:
        model = load_model(attempt)
        print(f"✅ Model loaded successfully from: {attempt}")
        break
    except Exception as e:
        print(f"⚠️ Failed to load from {attempt}: {str(e)}")

if model is None:
    raise FileNotFoundError("❌ Could not load model from any known location or format")

CLASS_LABELS = [
    "Bird_dropping",
    "Crack",
    "Dust",
    "Normal",
    "Shadow"
]

RECOMMENDATIONS = {
    "Normal": "Your solar panel appears to be in good condition. Regular cleaning is recommended to maintain optimal performance.",
    "Crack": "Micro-cracks detected. Contact a solar technician for inspection and potential panel replacement.",
    "Shadow": "Partial shading detected. Consider trimming nearby vegetation or relocating objects causing shadows.",
    "Dust": "Significant dirt accumulation detected. Cleaning the panels will improve energy production.",
    "Bird_dropping": "Bird droppings detected — install anti-bird mesh or clean panels to restore optimal performance."
}

def get_recommendation(fault_type: str) -> str:
    return RECOMMENDATIONS.get(fault_type, "No specific recommendation available.")

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path: str) -> np.ndarray:
    img = image.load_img(img_path, target_size=(160, 160))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No convolutional layer found in the model")

def generate_gradcam(model, img_array, class_index=None, layer_name=None):
    if layer_name is None:
        layer_name = get_last_conv_layer(model)
    
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]
    
    grads = tape.gradient(loss, conv_outputs)[0]
    guided_grads = tf.cast(conv_outputs > 0, "float32") * tf.cast(grads > 0, "float32") * grads
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    
    cam = tf.reduce_sum(weights * conv_outputs[0], axis=-1)
    cam = cv2.resize(cam.numpy(), (160, 160))
    cam = np.maximum(cam, 0)
    heatmap = cam / cam.max() if cam.max() != 0 else cam
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    img = (img_array[0] * 255).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return heatmap, superimposed_img

def generate_pdf_report(filename: str, fault_data: dict) -> io.BytesIO:
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = [
        Paragraph("Solar Panel Fault Detection Report", styles['Title']),
        Spacer(1, 12)
    ]
    
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    heatmap_path = os.path.join(UPLOAD_FOLDER, f"gradcam_{filename}")
    
    if os.path.exists(img_path) and os.path.exists(heatmap_path):
        img1 = RLImage(img_path, width=250, height=250)
        img2 = RLImage(heatmap_path, width=250, height=250)
        story.append(Table([[img1, img2]], colWidths=[250, 250]))
        story.append(Spacer(1, 20))
    
    story.append(Paragraph("Analysis Results", styles['Heading2']))
    text = (
        f"<b>Fault Type:</b> {fault_data['fault_type']}<br/>"
        f"<b>Confidence:</b> {fault_data['confidence']}%<br/>"
        f"<b>Recommendation:</b> {fault_data['recommendation']}"
    )
    story.append(Paragraph(text, styles['Normal']))
    
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer

def generate_batch_pdf_report(files_data: List[dict], summary: dict) -> io.BytesIO:
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = [
        Paragraph("Batch Solar Panel Analysis Report", styles['Title']),
        Spacer(1, 12),
        Paragraph("Summary", styles['Heading1']),
        Paragraph(
            f"<b>Total Images Analyzed:</b> {summary['total']}<br/>"
            f"<b>Normal Panels:</b> {summary['normal']}<br/>"
            f"<b>Faulty Panels:</b> {summary['faulty']}", 
            styles['Normal']
        ),
        Spacer(1, 24),
        Paragraph("Detailed Analysis", styles['Heading1'])
    ]
    
    for i, file_data in enumerate(files_data, 1):
        img_path = os.path.join(UPLOAD_FOLDER, file_data['filename'])
        heatmap_path = os.path.join(UPLOAD_FOLDER, f"gradcam_{file_data['filename']}")
        
        if os.path.exists(img_path) and os.path.exists(heatmap_path):
            story.append(Paragraph(f"Image {i}: {file_data['filename']}", styles['Heading2']))
            
            img1 = RLImage(img_path, width=200, height=200)
            img2 = RLImage(heatmap_path, width=200, height=200)
            story.append(Table([[img1, img2]], colWidths=[200, 200]))
            story.append(Spacer(1, 12))
            
            text = (
                f"<b>Fault Type:</b> {file_data['fault_type']}<br/>"
                f"<b>Confidence:</b> {file_data['confidence']}%<br/>"
                f"<b>Recommendation:</b> {file_data['recommendation']}"
            )
            story.append(Paragraph(text, styles['Normal']))
            story.append(Spacer(1, 24))
    
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())
        os.chmod(filepath, 0o644)
        
        img_array = preprocess_image(filepath)
        prediction = model.predict(img_array)[0]
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))
        fault_type = CLASS_LABELS[predicted_class]

        heatmap_filename = f"gradcam_{filename}"
        heatmap_filepath = os.path.join(UPLOAD_FOLDER, heatmap_filename)
        
        _, superimposed_img = generate_gradcam(model, img_array, class_index=predicted_class)
        cv2.imwrite(heatmap_filepath, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
        os.chmod(heatmap_filepath, 0o644)

        return {
            'status': 'success',
            'fault_type': fault_type,
            'confidence': round(confidence * 100, 2),
            'recommendation': get_recommendation(fault_type),
            'original_image': filename,
            'heatmap_image': heatmap_filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        if not allowed_file(file.filename):
            results.append({
                'filename': file.filename,
                'error': 'File type not allowed',
                'status': 'error'
            })
            continue

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        try:
            with open(filepath, "wb") as buffer:
                buffer.write(await file.read())
            os.chmod(filepath, 0o644)
            
            img_array = preprocess_image(filepath)
            prediction = model.predict(img_array)[0]
            predicted_class = np.argmax(prediction)
            confidence = float(np.max(prediction))
            fault_type = CLASS_LABELS[predicted_class]

            heatmap_filename = f"gradcam_{filename}"
            heatmap_filepath = os.path.join(UPLOAD_FOLDER, heatmap_filename)
            
            _, superimposed_img = generate_gradcam(model, img_array, class_index=predicted_class)
            cv2.imwrite(heatmap_filepath, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
            os.chmod(heatmap_filepath, 0o644)

            results.append({
                'status': 'success',
                'filename': filename,
                'fault_type': fault_type,
                'confidence': round(confidence * 100, 2),
                'recommendation': get_recommendation(fault_type),
                'original_image': filename,
                'heatmap_image': heatmap_filename,
                'is_normal': fault_type == "Normal"
            })
        except Exception as e:
            results.append({
                'filename': filename,
                'error': str(e),
                'status': 'error'
            })

    return {'results': results}

@app.post("/download_report")
async def download_report(filename: str = Form(...), fault_type: str = Form(...), 
                         confidence: float = Form(...), recommendation: str = Form(...)):
    try:
        pdf_buffer = generate_pdf_report(
            filename,
            {
                'fault_type': fault_type,
                'confidence': confidence,
                'recommendation': recommendation
            }
        )
        return StreamingResponse(
            io.BytesIO(pdf_buffer.getvalue()),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=solar_report_{filename}.pdf"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/download_batch_report")
async def download_batch_report(results: List[dict]):
    try:
        total = len(results)
        normal = sum(1 for r in results if r.get('is_normal'))
        faulty = total - normal
        summary = {'total': total, 'normal': normal, 'faulty': faulty}
        
        pdf_buffer = generate_batch_pdf_report(results, summary)
        return StreamingResponse(
            io.BytesIO(pdf_buffer.getvalue()),
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=batch_solar_report.pdf"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/uploads/{filename}")
async def serve_uploaded_file(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")