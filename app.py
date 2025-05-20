import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from deep_translator import GoogleTranslator
import torch

st.set_page_config(page_title="Auto Metadata Generator", layout="wide")

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_caption(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def translate_text(text, target_lang):
    return GoogleTranslator(source='auto', target=target_lang).translate(text)

# UI
st.title("🖼️ Auto Metadata Generator for Microstock")
st.markdown("Upload an image and get automatic caption, keywords, and translations.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating metadata..."):
        processor, model = load_model()
        caption_en = generate_caption(image, processor, model)
        caption_id = translate_text(caption_en, "id")
        keywords = [kw.strip() for kw in caption_en.replace(".", "").split()]

    st.subheader("📌 Captions")
    st.text_area("English", caption_en)
    st.text_area("Indonesian", caption_id)

    st.subheader("🔑 Keywords")
    st.write(", ".join(keywords))
