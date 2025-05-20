import os
import base64
import time
import hashlib
import logging
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from groq import Groq

# Load environment variables from .env file
load_dotenv(find_dotenv())
groq_api_key = os.getenv("GROQ_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Groq client
if not groq_api_key:
    logger.error("GROQ_API_KEY not found in environment variables")
    st.error("API key not configured. Please contact the administrator.")
else:
    client = Groq(api_key=groq_api_key)

# Function to encode image to Base64
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode("utf-8")

# Function to validate file size
def validate_file(file):
    # Maximum file size (5MB)
    MAX_FILE_SIZE = 5 * 1024 * 1024
    
    if file.size > MAX_FILE_SIZE:
        return False, "File size exceeds the 5MB limit."
    
    # Check file type
    allowed_types = ["image/jpeg", "image/png"]
    if file.type not in allowed_types:
        return False, "Only JPEG and PNG files are allowed."
    
    return True, ""

# Function to send image and prompt for analysis by Groq API
def analyze_image(prompt, file, retry_attempts=2):
    # Validate the file first
    is_valid, message = validate_file(file)
    if not is_valid:
        st.error(message)
        return
    
    # Generate a unique request ID for logging
    request_id = hashlib.md5(f"{time.time()}-{file.name}".encode()).hexdigest()[:10]
    logger.info(f"Processing request {request_id} - file: {file.name}")
    
    try:
        base_64_img = encode_image(file)
        
        with st.spinner("Analyzing image..."):
            # Request to Groq API with system prompt
            for attempt in range(retry_attempts + 1):
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert assistant in image analysis. Be clear, technical, and provide detailed responses based on the visual content of the image. Always respond as if you are helping a non-specialist user."
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base_64_img}"},
                                    },
                                ],
                            }
                        ],
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                    )
                    
                    # Return analysis result
                    result = chat_completion.choices[0].message.content
                    logger.info(f"Request {request_id} processed successfully")
                    return st.write(result)
                    
                except Exception as e:
                    if attempt < retry_attempts:
                        logger.warning(f"Request {request_id} - Retry {attempt+1}/{retry_attempts} after error: {str(e)}")
                        time.sleep(1)  # Wait before retrying
                    else:
                        raise e

    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}")
        st.error(f"Error processing the image: {str(e)}")


# Main Streamlit application function
def main():
    # Header styling - simplified as in the original
    st.markdown(
        """
        <style>
        .centered-header {
            text-align: center;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<h1 class="centered-header">This is the Analyzer</h1>', unsafe_allow_html=True)

    st.markdown("### Type your instructions about what you want to check from your image provided")
    st.divider()

    # Simple layout as in the original - com a ordem invertida
    image_file = st.file_uploader("Add image file", type=["jpeg", "png"])
    prompt_input = st.text_area("Type your prompt", height=200, key="input_image")

    # Analyze button
    if prompt_input and image_file:
        analyze_button = st.button("Analyze")
        if analyze_button:
            analyze_image(prompt_input, image_file)
    else:
        st.button("Analyze", disabled=True)

# Session state management
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = 0

if __name__ == "__main__":
    main()
