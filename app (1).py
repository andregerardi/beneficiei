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
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")  # Especificar extração de texto puro
    return text

# Função para processar a pergunta e encontrar a melhor resposta
def ask_question_from_pdf(pdf_text, question, history=[]):
    text_splitter = CharacterTextSplitter(chunk_size=1000, separator="\n")
    chunks = text_splitter.split_text(pdf_text)

    # Verificação para evitar erro de chunk vazio
    if not chunks:
        return "O texto do PDF não pôde ser segmentado corretamente.", history

    # Calcular similaridade entre a pergunta e os chunks de texto
    vectorizer = TfidfVectorizer().fit_transform([question] + chunks)
    vectors = vectorizer.toarray()
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    similar_indices = cosine_similarities.argsort()[-4:][::-1]  # Pegar os 4 chunks mais relevantes

    client = OpenAI(
        api_key=API_KEY_ANDRE,
        base_url="https://fgv-pocs-genie.cloud.databricks.com/serving-endpoints"
    )

    # Criar mensagens incluindo o histórico da conversa
    messages = [
        {"role": "user", 
         "content": f"Analise o demonstrativo financeiro da empresa fornecido e identifique oportunidades de otimização tributária.\n\n"
                    "🔹 **Estimativa:**\n"
                    "- Qual foi o resultado financeiro da empresa? Procure no título *DFs Individuais / Demonstração do Resultado*, o dado está *Resultado Líquido das Operações Continuadas*.\n\n"
                    "- Quanto a empresa gastou Pesquisa e Desenvolvimento? Procure no texto por informações sobre *Despesas com Pesquisa e Desenvolvimento*. \n\n"
                    "- Qual é o setor produtivo da empresa?\n\n"
                    "- Quanto ela gastou com doações para organizações filantrópicas?\n\n"
                    "🔹 **Sugestões de Benefícios Fiscais:**\n"
                    "- Lista de incentivos estaduais e federais para o setor produtivo baseado no perfil da empresa. Localize no PDF e encontre o estado de origem da empresa.n\n"
                    "- Créditos tributários aplicáveis.\n\n"
                    f"{' '.join([chunks[i] for i in similar_indices])}\n\n"
                    f"Pergunta: {question}"
        }
    ]

    messages.extend(history)

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="databricks-meta-llama-3-3-70b-instruct",
            max_tokens=4096
        )
        response = chat_completion.choices[0].message.content
    except Exception as e:
        return f"Erro ao chamar a API: {str(e)}", history

    # Atualizar histórico corretamente
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})

    return response, history

# Interface do chatbot no Streamlit
def main():
    st.title("BeneficiEI 🤖")

    uploaded_file = st.file_uploader("Insira o demonstrativo", type=["pdf"])

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
