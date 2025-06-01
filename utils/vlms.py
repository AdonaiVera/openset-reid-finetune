import base64
import json
from pathlib import Path
import google.generativeai as genai
from PIL import Image
import io
from dotenv import load_dotenv
import os

class GeminiVLM:
    def __init__(self, api_key=None):
        """Initialize Gemini client with API key"""
        load_dotenv()

        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

        genai.configure(api_key=api_key)
        self.model = "gemini-1.5-flash-latest"
        self.client = genai.GenerativeModel(self.model)
        
    def image_to_base64(self, image_path):
        """Convert image to base64 string"""
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large (Gemini has limits)
            max_size = 1600
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _get_analysis_prompt(self):
        """Get the prompt template for image analysis"""
        return """
        Analyze the person in the image(s) and provide a JSON response with the following structure. 
        Always return valid JSON, with exactly these fields and possible values:
        {
            "gender": "Male" | "Female" | "Unknown",
            "age": "Infant" | "Child" | "Teenager" | "Adult" | "Elderly" | "Unknown",
            "ethnicity": "Caucasian" | "African" | "Asian" | "Unknown",
            "occupation": "Student" | "Professional" | "Unknown",
            "appearance": {
                "hair": {
                    "type": "Short" | "Long" | "Curly" | "Straight" | "Bald" | "Ponytail" | "Unknown",
                    "color": "Black" | "Brown" | "Blonde" | "White" | "Gray" | "Unknown",
                    "description": "string"
                },
                "beard": {
                    "type": "Full Beard" | "Goatee" | "Mustache" | "None" | "Unknown",
                    "color": "Black" | "Brown" | "White" | "Gray" | "Unknown",
                    "description": "string"
                },
                "expression": {
                    "type": "Neutral" | "Smiling" | "Serious" | "Unknown",
                    "description": "string"
                }
            },
            "posture": {
                "type": "Standing" | "Sitting" | "Walking" | "Unknown",
                "description": "string"
            },
            "actions": {
                "type": "Walking" | "Standing" | "Sitting" | "Unknown",
                "description": "string"
            },
            "clothing": {
                "upper": {
                    "type": "T-shirt" | "Shirt" | "Jacket" | "Sweater" | "Unknown",
                    "color": "Black" | "White" | "Red" | "Blue" | "Green" | "Yellow" | "Unknown",
                    "description": "string"
                },
                "lower": {
                    "type": "Pants" | "Shorts" | "Skirt" | "Unknown",
                    "color": "Black" | "White" | "Red" | "Blue" | "Green" | "Yellow" | "Unknown",
                    "description": "string"
                },
                "shoes": {
                    "type": "Sneakers" | "Boots" | "Sandals" | "Unknown",
                    "color": "Black" | "White" | "Red" | "Blue" | "Unknown",
                    "description": "string"
                }
            },
            "accessories": {
                "hat": {
                    "type": "None" | "Baseball Cap" | "Beanie" | "Unknown",
                    "color": "Black" | "White" | "Red" | "Blue" | "Unknown",
                    "description": "string"
                },
                "glasses": {
                    "type": "None" | "Sunglasses" | "Prescription" | "Unknown",
                    "color": "Black" | "White" | "Unknown",
                    "description": "string"
                }
            },
            "description": "Write a short, natural-language summary describing the person based ONLY on clearly observed attributes. Do NOT include information that is unclear or marked as 'Unknown'. Omit any fields with insufficient visual evidence. Keep the description concise, accurate, and based entirely on the visible data."
        }

        Important:
        1. Always return valid JSON
        2. Use only the specified values for each field
        3. For descriptions, provide brief but detailed strings
        4. If uncertain about any value, use "Unknown"
        5. Ensure all fields are present, even if set to "Unknown"
        """

    def generate_global_attributes_batch(self, image_paths):
        """Generate consolidated attributes from multiple images of the same person"""
        if not image_paths:
            return self._get_empty_attributes()

        try:
            contents = [self._get_analysis_prompt()]

            # Include up to 5 images
            for image_path in image_paths[:5]:
                image_base64 = self.image_to_base64(image_path)
                contents.append({
                    "mime_type": "image/jpeg",
                    "data": image_base64
                })

            response = self.client.generate_content(
                contents=contents,
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.1,
                    "top_k": 16,
                }
            )

            try:
                attributes = json.loads(response.text.strip().strip("```json").strip("```"))

            except json.JSONDecodeError as e:
                print(f"[Gemini JSON ERROR] Invalid output:\n{response.text}")
                return self._get_empty_attributes()

            description = attributes.get("description", "Unknown")

            return attributes, description

        except Exception as e:
            print(f"[Gemini ERROR] Failed to generate from batch: {e}")
            return self._get_empty_attributes()

    def _get_empty_attributes(self):
        """Return empty attribute structure"""
        return {
            "gender": "",
            "age": "",
            "ethnicity": "",
            "profession": "",
            "appearance": {
                "hair": {"type": "", "color": "", "description": ""},
                "beard": {"type": "", "color": "", "description": ""},
                "expression": {"type": "", "description": ""}
            },
            "posture": {"type": "", "description": ""},
            "actions": {"type": "", "description": ""},
            "clothing": {
                "upper": {"type": "", "color": "", "description": ""},
                "lower": {"type": "", "color": "", "description": ""},
                "shoes": {"type": "", "color": "", "description": ""}
            },
            "accessories": {
                "hat": {"type": "", "color": "", "description": ""},
                "glasses": {"type": "", "color": "", "description": ""}
            },
            "description": ""
        }, "Unknown"