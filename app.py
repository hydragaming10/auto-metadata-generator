import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from keybert import KeyBERT
from deep_translator import GoogleTranslator
import io

st.set_page_config(page_title="Auto Metadata Generator", layout="centered")

st.title("📸 Auto Metadata Generator")
st.markdown("Upload gambar, dan aplikasi ini akan membantu membuat metadata seperti caption, title, description, dan keywords secara otomatis.")

uploaded_files = st.file_uploader(
    "Unggah gambar (bisa lebih dari satu)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    kw_model = KeyBERT()

    for uploaded_file in uploaded_files:
        st.image(uploaded_file, width=300)
        image = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')

        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs)
        caption_en = processor.decode(out[0], skip_special_tokens=True)
        caption_id = GoogleTranslator(source='en', target='id').translate(caption_en)

        keywords = kw_model.extract_keywords(
            caption_en,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=5
        )
        keywords_list = [kw[0] for kw in keywords]

        st.markdown("**Caption (EN):** " + caption_en)
        st.markdown("**Caption (ID):** " + caption_id)
        st.markdown("**Title:** " + caption_en.title())
        st.markdown("**Description:** " + caption_en + ". " + caption_id + ".")
        st.markdown("**Keywords:** " + ", ".join(keywords_list))
else:
    st.info("Silakan unggah gambar terlebih dahulu.")
