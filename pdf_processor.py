"""
PDF processing — text extraction, image extraction, passage splitting, and metadata.
All expensive operations are cached so they only run once per unique file.
"""
import hashlib
import io
import base64
import logging
import streamlit as st
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner="Reading PDF...")
def extract_text(_file_bytes: bytes, filename: str) -> tuple[str, dict, list]:
    """
    Extract all text and images from a PDF.
    Returns (full_text, metadata, images_list).
    Each image in images_list is a dict with keys: bytes, caption, page, base64.
    """
    import tempfile, os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_file_bytes)
        tmp_path = tmp.name

    try:
        # --- Text extraction with pdfplumber ---
        pages_text = []
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            page_count = len(pdf.pages)

        full_text = "\n".join(pages_text)
        metadata = {
            "pages": page_count,
            "word_count": len(full_text.split()),
            "char_count": len(full_text),
        }

        # --- Image extraction with PyMuPDF ---
        images_list = []
        try:
            doc = fitz.open(tmp_path)
            img_count = 0
            for page_num, page in enumerate(doc):
                image_list = page.get_images(full=True)
                text_blocks = page.get_text("blocks")

                for img_info in image_list:
                    if img_count >= 15:
                        break
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        img_bytes = base_image["image"]
                        img_ext = base_image.get("ext", "png")

                        # Skip tiny images (icons, bullets, etc.)
                        pil_img = Image.open(io.BytesIO(img_bytes))
                        w, h = pil_img.size
                        if w < 50 or h < 50:
                            continue

                        # Find caption: look for text just below the image
                        caption = ""
                        try:
                            img_rect = page.get_image_bbox(img_info)
                            for block in text_blocks:
                                bx0, by0, bx1, by1 = block[:4]
                                if by0 > img_rect.y1 and by0 - img_rect.y1 < 60:
                                    block_text = block[4] if len(block) > 4 else ""
                                    if block_text.strip():
                                        caption = block_text.strip()[:200]
                                        break
                        except Exception:
                            pass

                        if not caption:
                            caption = f"Figure on page {page_num + 1}"

                        # Convert to base64 for display
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        img_b64 = base64.b64encode(buf.getvalue()).decode()

                        images_list.append({
                            "bytes": img_bytes,
                            "caption": caption,
                            "page": page_num + 1,
                            "base64": img_b64,
                            "width": w,
                            "height": h,
                        })
                        img_count += 1
                    except Exception as e:
                        logger.debug(f"Skipping image on page {page_num+1}: {e}")
                        continue

                if img_count >= 15:
                    break
            doc.close()
        except Exception as e:
            logger.warning(f"Image extraction failed for {filename}: {e}")

    finally:
        os.unlink(tmp_path)

    return full_text, metadata, images_list


def split_passages(text: str, max_length: int = 500) -> list[str]:
    """Split text into roughly equal passages for embedding."""
    words = text.split()
    passages, chunk = [], []
    for word in words:
        if len(" ".join(chunk)) + len(word) + 1 <= max_length:
            chunk.append(word)
        else:
            passages.append(" ".join(chunk))
            chunk = [word]
    if chunk:
        passages.append(" ".join(chunk))
    return passages
