import os
import base64
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from groq import Groq

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv(find_dotenv())
groq_api_key = os.getenv("GROQ_API_KEY")

# Inicializa o cliente Groq
client = Groq(api_key=groq_api_key)

# Função para codificar a imagem em Base64
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode("utf-8")

# Função para enviar imagem e prompt para análise pela API Groq
def analyze_image(prompt, file):
    base_64_img = encode_image(file)

    try:
        # Requisição à API Groq
        chat_completion = client.chat.completions.create(
            messages=[
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
            model="llama-3.2-11b-vision-preview",
        )

        # Retorna o resultado da análise
        result = chat_completion.choices[0].message.content
        return st.write(result)

    except Exception as e:
        st.error(f"Error processing the image: {str(e)}")

# Função principal do aplicativo Streamlit
def main():
    # Estilização do cabeçalho
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

    # Entrada para o prompt e upload de imagem
    prompt_input = st.text_area("Type your prompt", height=200, key="input_image")
    image_file = st.file_uploader("Add image file", type=["jpeg", "png"])

    # Botão para analisar
    if prompt_input and image_file:
        analyze_button = st.button("Analyze")
        if analyze_button:
            analyze_image(prompt_input, image_file)
    else:
        st.button("Analyze", disabled=True)

if __name__ == "__main__":
    main()
