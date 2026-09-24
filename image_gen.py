"""
image_gen.py
------------
Core engine for the Multimodal Image Generation Studio.

Handles:
  - Building provider-specific parameter payloads (resolution, aspect ratio, count)
  - Calling the text-to-image API (OpenAI DALL-E 3 or Stability AI Stable Diffusion)
  - Normalizing responses (URL-based vs binary-based) into raw PNG/JPEG bytes

Both providers are wrapped behind one common interface: generate_images(...)
so the UI layer (app.py) never has to care which backend is active.
"""

import base64
import io
import os
import time
from dataclasses import dataclass
from typing import List, Literal, Optional

import requests

Provider = Literal["openai", "stability"]


@dataclass
class GeneratedImage:
    """A single generated image, normalized across providers."""
    image_bytes: bytes
    format: str = "png"
    source_provider: str = ""
    prompt: str = ""
    seed: Optional[int] = None


class ImageGenerationError(RuntimeError):
    """Raised when the upstream API fails or returns something we can't parse."""


# --------------------------------------------------------------------------
# OpenAI DALL-E 3 / DALL-E 2 backend
# --------------------------------------------------------------------------

def _generate_openai(
    prompt: str,
    size: str,
    count: int,
    quality: str,
    style: str,
    model: str,
    api_key: str,
) -> List[GeneratedImage]:
    """
    Calls OpenAI's Images API.

    Payload notes:
      - DALL-E 3 only supports n=1 per request (loop client-side for more).
      - DALL-E 3 supports quality: "standard" | "hd" and style: "vivid" | "natural".
      - DALL-E 2 supports n up to 10 but ignores quality/style.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImageGenerationError(
            "The 'openai' package is not installed. Run: pip install openai"
        ) from e

    client = OpenAI(api_key=api_key)

    results: List[GeneratedImage] = []
    # DALL-E 3 caps n=1 per call, so we issue multiple calls if count > 1
    calls_needed = count if model == "dall-e-3" else 1
    n_per_call = 1 if model == "dall-e-3" else count

    for _ in range(calls_needed):
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": n_per_call,
            "response_format": "b64_json",  # ask for binary data directly
        }
        if model == "dall-e-3":
            payload["quality"] = quality
            payload["style"] = style

        try:
            response = client.images.generate(**payload)
        except Exception as e:  # noqa: BLE001 - surface upstream error cleanly
            raise ImageGenerationError(f"OpenAI API error: {e}") from e

        for item in response.data:
            if item.b64_json:
                image_bytes = base64.b64decode(item.b64_json)
            elif item.url:
                image_bytes = _download_binary(item.url)
            else:
                raise ImageGenerationError("OpenAI response had neither b64_json nor url.")
            results.append(
                GeneratedImage(
                    image_bytes=image_bytes,
                    format="png",
                    source_provider="openai",
                    prompt=prompt,
                )
            )

    return results


# --------------------------------------------------------------------------
# Stability AI (Stable Diffusion) backend
# --------------------------------------------------------------------------

_STABILITY_ENDPOINT = "https://api.stability.ai/v2beta/stable-image/generate/core"

def _generate_stability(
    prompt: str,
    aspect_ratio: str,
    count: int,
    api_key: str,
    negative_prompt: Optional[str] = None,
) -> List[GeneratedImage]:
    """
    Calls Stability AI's Stable Image (SD3/Core) endpoint.
    This endpoint returns raw image bytes directly (no URL), one image per call.
    """
    results: List[GeneratedImage] = []

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "image/*",
    }

    for _ in range(count):
        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        try:
            response = requests.post(
                _STABILITY_ENDPOINT,
                headers=headers,
                files={"none": ""},  # Stability's API requires multipart/form-data
                data=payload,
                timeout=60,
            )
        except requests.RequestException as e:
            raise ImageGenerationError(f"Network error calling Stability AI: {e}") from e

        if response.status_code != 200:
            raise ImageGenerationError(
                f"Stability AI API error ({response.status_code}): {response.text[:300]}"
            )

        results.append(
            GeneratedImage(
                image_bytes=response.content,
                format="png",
                source_provider="stability",
                prompt=prompt,
            )
        )

    return results


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _download_binary(url: str) -> bytes:
    """Downloads image bytes from a hosted URL (used for OpenAI's URL response mode)."""
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return r.content
    except requests.RequestException as e:
        raise ImageGenerationError(f"Failed to download image from URL: {e}") from e


def generate_images(
    provider: Provider,
    prompt: str,
    count: int = 1,
    size: str = "1024x1024",
    aspect_ratio: str = "1:1",
    quality: str = "standard",
    style: str = "vivid",
    model: str = "dall-e-3",
    negative_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[GeneratedImage]:
    """
    Single entry point used by the UI. Builds the right payload for the
    selected provider and returns a normalized list of GeneratedImage objects.
    """
    if not prompt or not prompt.strip():
        raise ImageGenerationError("Prompt cannot be empty.")

    count = max(1, min(count, 10))  # sane upper bound

    if provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ImageGenerationError(
                "Missing OpenAI API key. Set OPENAI_API_KEY or pass one in the UI."
            )
        return _generate_openai(
            prompt=prompt,
            size=size,
            count=count,
            quality=quality,
            style=style,
            model=model,
            api_key=key,
        )

    elif provider == "stability":
        key = api_key or os.getenv("STABILITY_API_KEY")
        if not key:
            raise ImageGenerationError(
                "Missing Stability AI API key. Set STABILITY_API_KEY or pass one in the UI."
            )
        return _generate_stability(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            count=count,
            api_key=key,
            negative_prompt=negative_prompt,
        )

    raise ImageGenerationError(f"Unknown provider: {provider}")


def save_image(image: GeneratedImage, output_dir: str = "generated_images") -> str:
    """Saves a GeneratedImage to disk and returns the file path."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time() * 1000)
    filename = f"{image.source_provider}_{timestamp}.{image.format}"
    path = os.path.join(output_dir, filename)
    with open(path, "wb") as f:
        f.write(image.image_bytes)
    return path
