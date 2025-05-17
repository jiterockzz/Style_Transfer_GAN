import streamlit as st
from PIL import Image
import torch
from utils import load_image, save_image
from style_transfer import style_transfer
from upscaling import load_esrgan_model, upscale_image
import os

st.set_page_config(page_title="AI Style Transfer & Upscaler", layout="centered")
st.title("🎨 Neural Style Transfer & High-Res Upscaling")

st.markdown("""
Upload a **content image** and a **style reference image**, and the app will:
1. Apply the chosen artistic style to your content image.
2. Enhance the styled image using **super-resolution** (ESRGAN).
""")

content_file = st.file_uploader("Upload Content Image", type=["jpg", "png"], key="content")
style_file = st.file_uploader("Upload Style Image", type=["jpg", "png"], key="style")

if content_file and style_file:
    content_img = Image.open(content_file).convert("RGB")
    style_img = Image.open(style_file).convert("RGB")

    st.image(content_img, caption="Content Image", use_column_width=True)
    st.image(style_img, caption="Style Image", use_column_width=True)

    if st.button("🖌️ Stylize and Upscale"):
        with st.spinner("Applying style transfer..."):
            content_tensor = load_image(content_file, max_size=512)
            style_tensor = load_image(style_file, max_size=512)
            output_tensor = style_transfer(content_tensor, style_tensor)
            save_image(output_tensor, "stylized.jpg")
            styled_img = Image.open("stylized.jpg")

        st.image(styled_img, caption="Stylized Image", use_column_width=True)

        with st.spinner("Upscaling with ESRGAN..."):
            esrgan_model = load_esrgan_model("models/RRDB_ESRGAN_x4.pth")
            upscaled_img = upscale_image(styled_img, esrgan_model)
            upscaled_img.save("upscaled.jpg")

        st.image(upscaled_img, caption="Final Upscaled Image", use_column_width=True)
        st.success("Done! You can right-click and save the final image.")
else:
    st.info("Please upload both content and style images to proceed.")
