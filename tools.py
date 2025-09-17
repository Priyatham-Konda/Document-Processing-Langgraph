from typing import Dict, Any, List
from pathlib import Path
import mimetypes
import tempfile
import os
import io

import fitz  # PyMuPDF for PDF handling
from PIL import Image

# Gemini API imports
from google import genai
from google.genai import types
from dotenv import load_dotenv
import sys
from pathlib import Path as _Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Load environment variables
load_dotenv()

# Thread-based timeout utilities (cross-platform, safe in background threads)
class TimeoutError(Exception):
    pass

def _run_with_timeout(func, timeout_seconds: int):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")

# --- Dynamic Prompt Storage Utilities ---
_PROMPTS_DIR = _Path("prompts")
_EXTRACTION_PROMPT_FILE = _PROMPTS_DIR / "extraction_prompt.txt"

_DEFAULT_EXTRACTION_PROMPT = (
    "You are an expert data extraction system.\n"
    "Given an accounting or business document image, extract all relevant fields with high precision.\n"
    "Guidelines:\n"
    "- Use domain knowledge for invoices, receipts, POs, tax forms, etc.\n"
    "- Normalize values (trim whitespace, keep original punctuation inside values).\n"
    "- Use dot-notation for nested keys (e.g., vendor.address.street).\n"
    "- Return ONLY valid JSON with no commentary.\n"
    "- Include as many clearly present fields as possible.\n"
)

def get_extraction_prompt() -> str:
    """Return the current extraction prompt, creating a default if missing."""
    try:
        _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        if not _EXTRACTION_PROMPT_FILE.exists():
            _EXTRACTION_PROMPT_FILE.write_text(_DEFAULT_EXTRACTION_PROMPT, encoding="utf-8")
        return _EXTRACTION_PROMPT_FILE.read_text(encoding="utf-8")
    except Exception:
        return _DEFAULT_EXTRACTION_PROMPT

def set_extraction_prompt(new_prompt: str) -> None:
    """Persist a new extraction prompt to disk."""
    _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    _EXTRACTION_PROMPT_FILE.write_text(new_prompt.strip(), encoding="utf-8")

#--Get the file metadata--
def get_file_metadata(filename: str, file_content: bytes) -> Dict[str, Any]:
    """Extract metadata from uploaded file bytes.

    Returns a dictionary including filename, file_size, mime_type, file_extension,
    and for PDFs: page_count and pdf_info when available.
    """
    mime_type, _ = mimetypes.guess_type(filename)

    metadata: Dict[str, Any] = {
        'filename': filename,
        'file_size': len(file_content) if file_content is not None else 0,
        'mime_type': mime_type,
        'file_extension': Path(filename).suffix.lower() if filename else ''
    }

    # Add specific metadata for PDFs
    if filename and filename.lower().endswith('.pdf') and file_content:
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            doc = fitz.open(temp_file_path)
            metadata['page_count'] = len(doc)
            metadata['pdf_info'] = doc.metadata
            doc.close()
        except Exception:
            metadata['page_count'] = 1
        finally:
            try:
                if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            except Exception:
                pass

    return metadata

#--Convert the segment to image--
def convert_segment_to_image(file_bytes: bytes, document_id: str, page_num: int, 
                           file_extension: str = ".pdf") -> str:
    """Convert a document segment (page) to an image and save it.
    
    Args:
        file_bytes: The original file bytes
        document_id: Document identifier for folder naming
        page_num: Page number (1-indexed)
        file_extension: File extension to determine conversion method
        
    Returns:
        Path to the saved image file
    """
    # Create document folder if it doesn't exist
    doc_folder = Path(f"document_images/{document_id}")
    doc_folder.mkdir(parents=True, exist_ok=True)
    
    # Generate image filename
    image_filename = f"{document_id}_page_{page_num}.png"
    image_path = doc_folder / image_filename
    
    try:
        if file_extension.lower() == ".pdf":
            # Convert PDF page to image
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(file_bytes)
                temp_file_path = temp_file.name
            
            doc = fitz.open(temp_file_path)
            if page_num <= len(doc):
                page = doc[page_num - 1]  # Convert to 0-indexed
                # Convert page to image with high DPI for better quality
                # mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap()
                # pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Save image
                with open(image_path, "wb") as img_file:
                    img_file.write(img_data)
            else:
                raise ValueError(f"Page {page_num} not found in PDF")
            
            doc.close()
            os.unlink(temp_file_path)
            
        else:
            # For image files, just copy/convert to PNG
            with Image.open(io.BytesIO(file_bytes)) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                img.save(image_path, "PNG")
        
        print(f"  Saved image: {image_path}")
        return str(image_path)
        
    except Exception as e:
        print(f"  Error converting page {page_num} to image: {str(e)}")
        # Return a placeholder path even if conversion fails
        return str(image_path)


#--Generic Gemini API function--
def ask_gemini_about_image(image_path: str, prompt: str, model: str = "gemini-2.5-flash") -> str:
    """Generic function to send image and prompt to Gemini API.
    
    Args:
        image_path: Path to the image file (local file)
        prompt: The text prompt/question to ask about the image
        model: Gemini model to use (default: gemini-2.5-flash)
        
    Returns:
        The text response from Gemini API
    """
    try:
        print(f"    [DEBUG] Starting Gemini API call...")
        print(f"    [DEBUG] API Key present: {bool(os.environ.get('GEMINI_API_KEY'))}")
        print(f"    [DEBUG] Model: {model}")
        print(f"    [DEBUG] Image path: {image_path}")
        print(f"    [DEBUG] Image exists: {os.path.exists(image_path)}")
        
        def _create_client():
            return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        client = _create_client()
        print(f"    [DEBUG] Client created successfully")

        parts = []

        # Load local image file
        print(f"    [DEBUG] Opening image file...")
        with open(image_path, "rb") as f:
            image_data = f.read()
            print(f"    [DEBUG] Image data size: {len(image_data)} bytes")
            
            # Determine MIME type based on file extension
            if image_path.lower().endswith('.png'):
                mime_type = "image/png"
            elif image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
                mime_type = "image/jpeg"
            else:
                mime_type = "image/png"  # Default fallback
                
            print(f"    [DEBUG] MIME type: {mime_type}")
            parts.append(
                types.Part.from_bytes(
                    data=image_data,
                    mime_type=mime_type
                )
            )
        print(f"    [DEBUG] Image part added to request")

        # Add the prompt
        print(f"    [DEBUG] Adding prompt (length: {len(prompt)})...")
        parts.append(types.Part.from_text(text=prompt))
        print(f"    [DEBUG] Prompt added to request")

        contents = [types.Content(role="user", parts=parts)]
        print(f"    [DEBUG] Content prepared with {len(parts)} parts")

        generate_content_config = types.GenerateContentConfig(
            response_modalities=["TEXT"],  # we only want text back
        )
        print(f"    [DEBUG] Config prepared")

        print(f"    [DEBUG] Calling Gemini API...")

        timeout_seconds = int(os.environ.get("GEMINI_IMAGE_TIMEOUT_SEC", "120"))

        def _do_call():
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )

        response = _run_with_timeout(_do_call, timeout_seconds)
        print(f"    [DEBUG] API call completed")

        result_text = response.candidates[0].content.parts[0].text
        print(f"    [DEBUG] Response text length: {len(result_text) if result_text else 0}")
        return result_text

    except Exception as e:
        print(f"    [DEBUG] Error calling Gemini API: {str(e)}")
        print(f"    [DEBUG] Error type: {type(e).__name__}")
        import traceback
        print(f"    [DEBUG] Traceback: {traceback.format_exc()}")
        return f"Error: {str(e)}"


def ask_gemini_text_only(prompt: str, model: str = "gemini-2.5-pro") -> str:
    """Generic function to send text-only prompt to Gemini API (no image).
    
    Args:
        prompt: The text prompt/question
        model: Gemini model to use (default: gemini-2.5-pro)
        
    Returns:
        The text response from Gemini API
    """
    try:
        print(f"    [DEBUG] Starting Gemini text-only API call...")
        print(f"    [DEBUG] API Key present: {bool(os.environ.get('GEMINI_API_KEY'))}")
        print(f"    [DEBUG] Model: {model}")
        
        def _create_client():
            return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        client = _create_client()
        print(f"    [DEBUG] Client created successfully")

        # Add the prompt as text only
        parts = [types.Part.from_text(text=prompt)]
        print(f"    [DEBUG] Prompt added to request (length: {len(prompt)})")

        contents = [types.Content(role="user", parts=parts)]
        print(f"    [DEBUG] Content prepared with {len(parts)} parts")

        generate_content_config = types.GenerateContentConfig(
            response_modalities=["TEXT"],  # we only want text back
        )
        print(f"    [DEBUG] Config prepared")

        print(f"    [DEBUG] Calling Gemini API...")

        timeout_seconds = int(os.environ.get("GEMINI_TEXT_TIMEOUT_SEC", "30"))

        def _do_call():
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )

        response = _run_with_timeout(_do_call, timeout_seconds)
        print(f"    [DEBUG] API call completed")

        result_text = response.candidates[0].content.parts[0].text
        print(f"    [DEBUG] Response text length: {len(result_text) if result_text else 0}")
        return result_text

    except Exception as e:
        print(f"    [DEBUG] Error calling Gemini API: {str(e)}")
        print(f"    [DEBUG] Error type: {type(e).__name__}")
        import traceback
        print(f"    [DEBUG] Traceback: {traceback.format_exc()}")
        return f"Error: {str(e)}"
