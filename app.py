"""
app.py
------
Multimodal Image Generation Studio — a Streamlit visual app that translates
natural language prompts into digital artwork using DALL-E 3 or Stable Diffusion.

Run with:
    streamlit run app.py
"""

import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from image_gen import generate_images, save_image, ImageGenerationError

load_dotenv()  # pulls OPENAI_API_KEY / STABILITY_API_KEY from a local .env file, if present

st.set_page_config(
    page_title="Multimodal Image Generation Studio",
    page_icon="🎨",
    layout="wide",
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {prompt, images, timestamp, provider}

# --------------------------------------------------------------------------
# Sidebar — provider + parameter payload controls
# --------------------------------------------------------------------------
st.sidebar.title("🎛️ Generation Parameters")

provider = st.sidebar.selectbox(
    "Provider",
    options=["openai", "stability"],
    format_func=lambda p: "DALL-E 3 (OpenAI)" if p == "openai" else "Stable Diffusion (Stability AI)",
)

api_key_env_name = "OPENAI_API_KEY" if provider == "openai" else "STABILITY_API_KEY"
api_key_input = st.sidebar.text_input(
    f"{api_key_env_name} (optional override)",
    type="password",
    help="Leave blank to use the key from your .env file / environment variable.",
)

st.sidebar.markdown("---")

if provider == "openai":
    model = st.sidebar.selectbox("Model", options=["dall-e-3", "dall-e-2"])
    size = st.sidebar.selectbox(
        "Resolution",
        options=["1024x1024", "1792x1024", "1024x1792"]
        if model == "dall-e-3"
        else ["256x256", "512x512", "1024x1024"],
    )
    quality = st.sidebar.selectbox("Quality", options=["standard", "hd"], disabled=(model != "dall-e-3"))
    style = st.sidebar.selectbox("Style", options=["vivid", "natural"], disabled=(model != "dall-e-3"))
    max_count = 1 if model == "dall-e-3" else 10
    count = st.sidebar.slider("Number of images", min_value=1, max_value=max_count, value=1)
    aspect_ratio = None
    negative_prompt = None
else:
    aspect_ratio = st.sidebar.selectbox(
        "Aspect Ratio", options=["1:1", "16:9", "9:16", "3:2", "2:3", "4:5", "5:4"]
    )
    count = st.sidebar.slider("Number of images", min_value=1, max_value=8, value=1)
    negative_prompt = st.sidebar.text_area(
        "Negative prompt (optional)", placeholder="e.g. blurry, low quality, watermark"
    )
    model, size, quality, style = None, None, None, None

save_locally = st.sidebar.checkbox("Save generated images to ./generated_images", value=True)

# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------
st.title("🎨 Multimodal Image Generation Studio")
st.caption("Turn natural language descriptions into digital artwork.")

prompt = st.text_area(
    "Describe the image you want to create",
    height=120,
    placeholder="A bioluminescent forest at night, painted in the style of a watercolor illustration...",
)

generate_clicked = st.button("✨ Generate", type="primary", use_container_width=False)

if generate_clicked:
    if not prompt.strip():
        st.warning("Please enter a description first.")
    else:
        with st.spinner("Generating your artwork..."):
            try:
                images = generate_images(
                    provider=provider,
                    prompt=prompt,
                    count=count,
                    size=size or "1024x1024",
                    aspect_ratio=aspect_ratio or "1:1",
                    quality=quality or "standard",
                    style=style or "vivid",
                    model=model or "dall-e-3",
                    negative_prompt=negative_prompt or None,
                    api_key=api_key_input or None,
                )

                saved_paths = []
                if save_locally:
                    for img in images:
                        saved_paths.append(save_image(img))

                st.session_state.history.insert(
                    0,
                    {
                        "prompt": prompt,
                        "provider": provider,
                        "images": images,
                        "paths": saved_paths,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                st.success(f"Generated {len(images)} image(s).")

            except ImageGenerationError as e:
                st.error(str(e))
            except Exception as e:  # noqa: BLE001
                st.error(f"Unexpected error: {e}")

# --------------------------------------------------------------------------
# Gallery — display results, most recent batch first
# --------------------------------------------------------------------------
st.markdown("---")
st.subheader("🖼️ Gallery")

if not st.session_state.history:
    st.info("Your generated images will appear here.")
else:
    for batch_idx, batch in enumerate(st.session_state.history):
        with st.container():
            st.markdown(f"**Prompt:** {batch['prompt']}")
            st.caption(f"{batch['provider']} · {batch['timestamp']}")

            cols = st.columns(min(len(batch["images"]), 4) or 1)
            for i, img in enumerate(batch["images"]):
                col = cols[i % len(cols)]
                with col:
                    st.image(img.image_bytes, use_container_width=True)
                    st.download_button(
                        label="Download",
                        data=img.image_bytes,
                        file_name=f"artwork_{batch_idx}_{i}.{img.format}",
                        mime=f"image/{img.format}",
                        key=f"dl_{batch_idx}_{i}",
                    )
            st.markdown("---")
