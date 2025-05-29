import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.cm as cm

# Class labels
class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

# Medical suggestions
suggestions = {
    "COVID": "1. Initiate immediate isolation precautions to prevent transmission\n2. Obtain confirmatory RT-PCR testing for SARS-CoV-2\n3. Seek urgent evaluation by a healthcare provider for severity assessment\n4. Monitor oxygen saturation and respiratory status closely\n5. Consider follow-up imaging if clinical deterioration occurs",
    "Normal": "1. No immediate pulmonary intervention required based on imaging findings\n2. Continue routine health maintenance and preventive care\n3. Re-evaluate if respiratory symptoms persist or worsen\n4. Consider alternative diagnoses for ongoing symptoms\n5. Follow standard screening guidelines for future imaging",
    "Viral Pneumonia": "1. Initiate appropriate antiviral therapy per clinical guidelines\n2. Obtain viral PCR panel for pathogen identification\n3. Monitor for secondary bacterial infection\n4. Provide supportive care with oxygenation as needed\n5. Schedule follow-up imaging if no clinical improvement in 48-72 hours",
    "Lung_Opacity": "1. Obtain high-resolution CT for further characterization of findings\n2. Consult pulmonary medicine for comprehensive evaluation\n3. Consider bronchoscopy or biopsy if malignancy is suspected\n4. Initiate appropriate antimicrobial therapy if infectious etiology suspected\n5. Schedule follow-up imaging based on suspected etiology"
}

# Load model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load("resnet50_covid_classifier.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Grad-CAM storage
activations = []
gradients = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

model.layer4.register_forward_hook(forward_hook)
model.layer4.register_full_backward_hook(backward_hook)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Grad-CAM function
def generate_gradcam(model, image_tensor, class_idx):
    activations.clear()
    gradients.clear()

    output = model(image_tensor)
    model.zero_grad()
    output[0, class_idx].backward()

    act = activations[0].squeeze(0)
    grad = gradients[0].squeeze(0)
    pooled_grad = torch.mean(grad, dim=(1, 2))

    for i in range(act.shape[0]):
        act[i] *= pooled_grad[i]

    heatmap = torch.mean(act, dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap

# Model explanation
explanation = {
    "COVID": "The model detected diffuse lung abnormalities, suggesting ground-glass opacities typical of COVID-19 pneumonia. These often appear as hazy, bilateral patches with peripheral distribution. Findings may overlap with other viral pneumonias. Clinical correlation and testing are needed for confirmation.",
    "Lung_Opacity": "The model identified dense regions in lung fields, indicating possible fluid, infection, or masses. Opacities can vary from hazy to solid in appearance. They are nonspecific and require further evaluation. Differential includes pneumonia, edema, or fibrosis.",
    "Viral Pneumonia": "The model highlighted irregular, patchy opacities consistent with viral pneumonia patterns. These differ from bacterial pneumonia's dense consolidation. Bilateral involvement is common. Distinction from COVID-19 requires clinical context.",
    "Normal": "The model found no abnormal lung findings, showing clear fields and sharp anatomical borders. No consolidations or opacities were detected. Early disease may not be visible. Always correlate with patient symptoms."
}

# Streamlit UI
st.title("🩻🧠 Chest X-ray Classifier")
st.write("Upload a chest X-ray image to detect **COVID-19**, **Normal**, **Viral Pneumonia**, or **Lung Opacity**.")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        # Simple check: Ensure image has expected grayscale or X-ray-like pixel profile
        grayscale_ratio = np.mean(np.abs(np.array(image)[:, :, 0] - np.array(image)[:, :, 1]))
        if grayscale_ratio > 30:
            st.error("🚫 This doesn't appear to be a valid chest X-ray. Please upload a proper X-ray image.")
        else:
            st.image(image, caption="Uploaded Image", use_container_width=True)

            img_tensor = transform(image).unsqueeze(0)
            img_tensor.requires_grad_()

            outputs = model(img_tensor)
            probs = F.softmax(outputs[0], dim=0)
            pred_idx = torch.argmax(probs).item()
            pred_class = class_names[pred_idx]

            st.subheader("🔍 Prediction Results:")
            for i, prob in enumerate(probs):
                st.write(f"{class_names[i]}: {prob.item() * 100:.2f}%")

            st.success(f"🔬 Most likely: **{pred_class}**")

            st.subheader("📋 Suggested Medical Step:")
            st.info(suggestions[pred_class])

            st.subheader("🌡️ Grad-CAM Heatmap:")
            heatmap = generate_gradcam(model, img_tensor, pred_idx)

            # Blend heatmap with image safely
            heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(image.size)
            heatmap_resized = np.array(heatmap_resized)
            image_np = np.array(image).astype(np.float32) / 255.0
            heatmap_color = cm.jet(heatmap_resized / 255.0)[..., :3]
            blended = 0.6 * image_np + 0.4 * heatmap_color
            blended = np.clip(blended, 0, 1)

            st.image(blended, caption="Model Decision Heatmap", use_container_width=True)

            st.subheader("🧠 Model Explanation:")
            st.write(explanation.get(pred_class, "No explanation available."))

    except Exception as e:
        st.error(f"🚫 Unexpected error: {e}")
