"""Deterministic, non-persistent image transport views for E3.5."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

from PIL import Image

from agrinet.research.hcv.collector import image_url_content

TRANSPORT_IMAGE_VERSION = "agrinet.e35-transport-image/v1-rgb-jpeg-q88"


def transport_image(path: Path, *, max_side: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an API image content item and auditable public metadata.

    The bytes are generated in memory only.  The original source image remains
    the sample identity and the later Hermes image binding.
    """
    if max_side <= 0:
        return image_url_content(path), {"version": "original", "max_side": 0}
    raw = path.read_bytes()
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            encoded = io.BytesIO()
            image.save(encoded, format="JPEG", quality=88, optimize=True)
            payload = encoded.getvalue()
            size = list(image.size)
    except Exception as exc:
        raise ValueError(f"unable to create deterministic E3.5 transport image: {path}") from exc
    # Reuse the existing OpenAI-compatible data-URL encoder only after writing
    # no temporary output; its resize path would decode the JPEG again.
    import base64
    item = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")}}
    return item, {"version": TRANSPORT_IMAGE_VERSION, "max_side": max_side, "size": size,
                  "sha256": hashlib.sha256(payload).hexdigest()}
