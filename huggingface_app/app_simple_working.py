"""
Simple Working Malaria Detection App
Uses your trained model from checkpoints/best.pt
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/best.pt"
CLASS_NAMES = ["Parasitized (Infected)", "Uninfected (Healthy)"]

# Build model - EfficientNet-B0
def build_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

# Load model
print(f"Loading model from {MODEL_PATH}...")
model = build_model().to(DEVICE)

if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print("✅ Model loaded successfully!")
else:
    print(f"⚠️ Warning: {MODEL_PATH} not found. Using untrained model.")

model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    """Make prediction on uploaded image"""
    if image is None:
        return "Please upload an image", None, None

    try:
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Preprocess
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = outputs.argmax(1).item()
            confidence = probabilities[predicted_class].item()

        # Format result
        diagnosis = CLASS_NAMES[predicted_class]
        prob_parasitized = probabilities[0].item() * 100
        prob_uninfected = probabilities[1].item() * 100

        # Color code
        if predicted_class == 0:  # Infected
            color = "#FF4444"
            status = "🦠 MALARIA DETECTED"
        else:
            color = "#44FF44"
            status = "✅ NO MALARIA DETECTED"

        result_html = f"""
        <div style='padding: 20px; background: {color}22; border-left: 5px solid {color}; border-radius: 5px;'>
            <h2 style='color: {color}; margin: 0 0 10px 0;'>{status}</h2>
            <p style='margin: 5px 0;'><strong>Prediction:</strong> {diagnosis}</p>
            <p style='margin: 5px 0;'><strong>Confidence:</strong> {confidence*100:.1f}%</p>
            <hr style='border: none; border-top: 1px solid #ccc; margin: 10px 0;'>
            <p style='margin: 5px 0;'><strong>Probabilities:</strong></p>
            <p style='margin: 5px 0;'>• Parasitized: {prob_parasitized:.1f}%</p>
            <p style='margin: 5px 0;'>• Uninfected: {prob_uninfected:.1f}%</p>
        </div>
        <div style='padding: 10px; background: #f0f0f0; margin-top: 10px; border-radius: 5px; font-size: 12px;'>
            <strong>⚠️ Disclaimer:</strong> This is a research tool. NOT for clinical diagnosis. Always confirm with certified lab tests.
        </div>
        """

        return result_html, image, f"Confidence: {confidence*100:.1f}%"

    except Exception as e:
        error_html = f"""
        <div style='padding: 20px; background: #ffeeee; border-left: 5px solid #ff0000; border-radius: 5px;'>
            <h3 style='color: #ff0000; margin: 0 0 10px 0;'>❌ Error</h3>
            <p>{str(e)}</p>
        </div>
        """
        return error_html, None, None

# Create Gradio interface
with gr.Blocks(title="Malaria Detection AI") as demo:

    gr.Markdown("""
    # 🔬 AI-Powered Malaria Detection
    ### Accuracy: 94.3% | Energy Savings: 85%

    Upload a blood cell microscopy image to detect malaria parasites.
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Blood Cell Image", type="pil")
            predict_btn = gr.Button("🔬 Analyze for Malaria", variant="primary")

        with gr.Column():
            result_output = gr.HTML(label="Diagnosis")
            image_output = gr.Image(label="Input Image")
            confidence_output = gr.Textbox(label="Confidence Score")

    predict_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[result_output, image_output, confidence_output]
    )

    gr.Markdown("""
    ---
    ### About This System
    - **Model:** EfficientNet-B0 trained with Adaptive Sparse Training
    - **Dataset:** NIH Malaria Cell Images (27,558 samples)
    - **Accuracy:** 94.3% validation accuracy
    - **Energy Savings:** 85% reduction in training cost

    **Developer:** Oluwafemi Idiakhoa | [GitHub](https://github.com/oluwafemidiakhoa/Malaria)
    """)

if __name__ == "__main__":
    demo.launch()
