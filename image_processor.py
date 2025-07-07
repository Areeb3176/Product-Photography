import os
import shutil
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from rembg import remove
import time

# --- Gemini SDK Import ---
try:
    from google import genai as google_genai_sdk
    from google.genai import types as google_genai_types
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    print("[!] Google GenAI SDK not installed. Run: pip install google-genai")

# --- Gemini Model Configuration ---
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp-image-generation"

# --- Helper Functions ---
def setup_directories(*dirs):
    """Create directories if they don't exist"""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def reduce_noise_cv(pil_image):
    """Apply noise reduction to image"""
    try:
        if pil_image.mode in ['RGBA', 'P']:
            pil_image = pil_image.convert('RGB')
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)
        return Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"Noise reduction failed: {e}")
        return pil_image

def remove_background_rembg(pil_image, bg_color=(255, 255, 255, 0)):
    """Remove background from image"""
    try:
        result = remove(pil_image)
        if bg_color[3] == 0:
            return result
        else:
            background = Image.new("RGB", result.size, bg_color[:3])
            background.paste(result, mask=result.split()[3])
            return background
    except Exception as e:
        print(f"Background removal failed: {e}")
        return pil_image

def call_gemini_image_editing(api_key, model_name, prompt, input_pil_image):
    """Call Gemini API for image editing"""
    if not SDK_AVAILABLE or input_pil_image is None:
        return None
    try:
        client = google_genai_sdk.Client(api_key=api_key)
        if input_pil_image.mode == 'RGBA':
            rgb_image = Image.new("RGB", input_pil_image.size, (255, 255, 255))
            rgb_image.paste(input_pil_image, mask=input_pil_image.split()[3])
            image_to_send = rgb_image
        else:
            image_to_send = input_pil_image.convert("RGB")

        contents = [prompt, image_to_send]

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=google_genai_types.GenerateContentConfig(response_modalities=['Text', 'Image'])
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                image_bytes = part.inline_data.data
                return Image.open(BytesIO(image_bytes))

    except Exception as e:
        print(f"[Gemini Error] {e}")
        raise Exception(f"Gemini API error: {e}")
    return None

def process_image(image_path, prompt, api_key, preprocessed_dir, gemini_output_dir):
    """Process a single image through the enhancement workflow"""
    result = {
        'success': False,
        'preprocessed_path': None,
        'gemini_path': None,
        'error': None,
        'processing_time': 0
    }
    
    start_time = time.time()
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    try:
        # Open image
        img = Image.open(image_path)
        
        # Step 1: Noise Reduction
        denoised = reduce_noise_cv(img)
        
        # Step 2: Background Removal
        bg_removed = remove_background_rembg(denoised, bg_color=(255, 255, 255, 0))
        
        # Save preprocessed image
        pre_path = os.path.join(preprocessed_dir, f"{base_name}_pre.png")
        bg_removed.save(pre_path)
        result['preprocessed_path'] = os.path.join('preprocessed', f"{base_name}_pre.png")
        
        # Step 3: Gemini Editing
        if SDK_AVAILABLE and api_key:
            edited = call_gemini_image_editing(api_key, GEMINI_MODEL_NAME, prompt, bg_removed)
            if edited:
                final_path = os.path.join(gemini_output_dir, f"{base_name}_gemini.png")
                edited.save(final_path)
                result['gemini_path'] = os.path.join('gemini_edited', f"{base_name}_gemini.png")
                result['success'] = True
            else:
                result['error'] = "Gemini editing failed"
        else:
            result['error'] = "SDK or API key missing"
            
    except Exception as e:
        result['error'] = str(e)
    
    result['processing_time'] = round(time.time() - start_time, 2)
    return result