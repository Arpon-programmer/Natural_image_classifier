import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
model = torch.jit.load('Natual_Image_classifier_2_lightning_AI_99%.pt',map_location=device)
classes = ['Airplane', 'Car', 'Cat', 'Dog', 'Flower', 'Fruit', 'Motorbike', 'Person']
def pridiction(model,img):
    model.eval()
    with torch.no_grad():
        out = model(img)
        pred = torch.argmax(out, dim=1)
        return classes[pred.item()]
st.title('Predict objects via image')
st.text('Try our new AI model for image recognition!')
img = st.file_uploader('Upload an image', type=['jpg', 'png', 'jpeg'])
if img is not None:
    st.image(img, caption='Uploaded Image.')
    img = Image.open(img)
    trans_img = transform(img).unsqueeze(0).to(device)
    button = st.button('Recognize')
    if button:
        result = pridiction(model,trans_img)
        st.success(f'The object in the image is: {result}')