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

# Updated medical suggestions
suggestions = {
    "COVID": "Seek immediate medical attention and isolate yourself. Follow up with a PCR test and consult with an infectious disease specialist.",
    "Normal": "Your chest X-ray appears normal. Maintain a healthy lifestyle and schedule regular check-ups.",
    "Viral Pneumonia": "Consult a physician for antiviral treatment and ensure adequate hydration and rest. Monitor symptoms closely.",
    "Lung_Opacity": "Schedule further diagnostic imaging such as a CT scan and consult a pulmonologist for possible interstitial lung disease or infection."
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

# Streamlit UI
st.title("\U0001FA7A\U0001FA7B Chest X-ray Classifier")
st.write("Upload a chest X-ray image to detect COVID-19, Normal, Viral Pneumonia, or Lung Opacity.")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)
    img_tensor.requires_grad_()

    outputs = model(img_tensor)
    probs = F.softmax(outputs[0], dim=0)
    pred_idx = torch.argmax(probs).item()
    pred_class = class_names[pred_idx]

    st.subheader("\U0001F50D Prediction Results:")
    for i, prob in enumerate(probs):
        st.write(f"{class_names[i]}: {prob.item() * 100:.2f}%")

    st.success(f"\U0001F52C Most likely: **{pred_class}**")

    st.subheader("\U0001FA7B Suggested Medical Step:")
    st.info(suggestions[pred_class])

    st.subheader("\U0001F321️ Grad-CAM Heatmap:")
    heatmap = generate_gradcam(model, img_tensor, pred_idx)

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(image.size)
    heatmap_color = cm.jet(np.array(heatmap_img) / 255.0)[..., :3]
    blended = np.array(image) / 255.0 * 0.6 + heatmap_color * 0.4
    blended = np.clip(blended, 0, 1)

    st.image(blended, caption="Model Decision Heatmap", use_container_width=True)

    st.subheader("🧠 Model Explanation:")
    explanation = {
        "COVID": "The model focused on diffuse areas in the lungs, which may indicate ground-glass opacities typical in COVID-related pneumonia.",
        "Lung_Opacity": "The model highlighted dense regions in the lung fields, possibly indicating fluid or mass consistent with lung opacity.",
        "Viral Pneumonia": "The model focused on irregular patches or localized opacities, common in viral infections like pneumonia.",
        "Normal": "The model did not detect abnormal regions, indicating clear lung fields."
    }
    st.write(explanation.get(pred_class, "No explanation available."))
