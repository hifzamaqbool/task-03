# Multimodal Image Generation Studio

A visual app that turns natural language descriptions into digital artwork,
built with Streamlit and pluggable text-to-image backends (DALL-E 3 via
OpenAI, or Stable Diffusion via Stability AI).

## Setup (in VS Code)

1. Open this folder in VS Code.
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add your API key(s):
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and paste in your `OPENAI_API_KEY` and/or `STABILITY_API_KEY`.

## Run

```bash
streamlit run app.py
```

This opens the app in your browser (usually `http://localhost:8501`). From
VS Code you can also just click "Run" with the integrated terminal, or use
the built-in `Run Python File` button pointed at `app.py` via the Streamlit CLI.

## How it's structured

- **`image_gen.py`** — the engine. Builds the provider-specific request
  payload (resolution/aspect ratio, generation count, quality/style), calls
  the API, and normalizes the response (URL or base64/binary) into raw image
  bytes via the `GeneratedImage` dataclass.
- **`app.py`** — the Streamlit UI. Lets you pick a provider, set parameters,
  enter a prompt, and view/download results in a gallery. Generated images
  are optionally saved to `./generated_images`.

## Notes on the two providers

| | DALL-E 3 (OpenAI) | Stable Diffusion (Stability AI) |
|---|---|---|
| Resolution control | Fixed sizes (`1024x1024`, `1792x1024`, `1024x1792`) | Aspect ratio (`1:1`, `16:9`, etc.) |
| Images per request | 1 (app loops for more) | Up to several per request |
| Extra controls | `quality` (standard/hd), `style` (vivid/natural) | `negative_prompt` |
| Response format | URL or base64 | Raw binary |

## Extending it

- Swap in a local Stable Diffusion pipeline (e.g. `diffusers` + `torch`) by
  adding a new branch in `generate_images()` in `image_gen.py` — the UI
  doesn't need to change.
- Add an image-to-image or inpainting tab by reusing the same
  `GeneratedImage` return type.
