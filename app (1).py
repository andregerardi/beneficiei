import fitz  # PyMuPDF
import streamlit as st
from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
API_KEY_ANDRE = st.secrets["auth_token"]

# Função para extrair texto de um arquivo PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Função para enviar a pergunta e obter resposta considerando histórico
def ask_question_from_pdf(pdf_text, question, history=[]):
    try:
        text_splitter = CharacterTextSplitter(chunk_size=500, separator="\n")
        chunks = text_splitter.split_text(pdf_text)
    except Exception as e:
        st.error(f"Erro ao dividir o texto: {str(e)}")
        return "", history
    
    # Calcular similaridade entre a pergunta e os chunks de texto
    vectorizer = TfidfVectorizer().fit_transform([question] + chunks)
    vectors = vectorizer.toarray()
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    similar_indices = cosine_similarities.argsort()[-4:][::-1]  # Pegar os 4 chunks mais similares

    client = OpenAI(
        api_key=API_KEY_ANDRE,
        base_url="https://fgv-pocs-genie.cloud.databricks.com/serving-endpoints"
    )

    # Criar mensagens incluindo o histórico da conversa
    messages = [{"role": "system", 
                 "content": "Você é um assistente técnico agrícola. Suas respostas devem ser fáceis de entender e voltadas para agricultores.Responda apenas com base o pdf fornecido. Não mencione por exemplo, 'Recomendo que você consulte um especialista em citricultura'."}]
    
    messages.extend(history)  # Adiciona histórico da conversa

    # Adicionar contexto relevante dos chunks mais similares do PDF
    relevant_chunks = "\n\n".join([chunks[i] for i in similar_indices])
    messages.append({"role": "user", "content": f"Baseado no seguinte conteúdo do PDF: {relevant_chunks}\n\nPergunta: {question}"})

    # Chamar a API
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="databricks-meta-llama-3-3-70b-instruct",
        max_tokens=1024
    )
    
    response = chat_completion.choices[0].message.content

    # Atualizar histórico corretamente
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})

    return response, history

# Interface do chatbot no Streamlit
def main():
    st.title("Agrônomo Virtual 🤖")

    uploaded_file = st.file_uploader("Escolha um arquivo PDF", type=["pdf"])

    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(uploaded_file)

        if not pdf_text.strip():
            st.error("O PDF não contém texto legível.")
            return

        if 'history' not in st.session_state:
            st.session_state.history = []

        st.subheader("Chat")

        # Exibir histórico corretamente
        for message in st.session_state.history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Entrada de chat (parecida com a do ChatGPT)
        question = st.chat_input("Digite sua pergunta sobre o PDF...")

        if question:
            with st.chat_message("user"):
                st.markdown(question)

            with st.spinner("Processando..."):
                response, history = ask_question_from_pdf(pdf_text, question, st.session_state.history)
                st.session_state.history = history

            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    main()
