import fitz  # PyMuPDF
import streamlit as st
from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationChain  # Importando o ConversationChain
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langchain.vectorstores import Chroma

load_dotenv()
API_KEY_ANDRE = st.secrets["auth_token"]

# Inicializar o cliente OpenAI/Databricks
client = OpenAI(
    api_key=API_KEY_ANDRE,
    base_url="https://fgv-pocs-genie.cloud.databricks.com/serving-endpoints"
)

def get_embeddings(texts):
    if isinstance(texts, str):  # Se for um único texto, transforma em lista
        texts = [texts]
    
    embeddings = client.embeddings.create(
        model="databricks-gte-large-en",
        input=texts
    )
    
    return [emb.data.embedding for emb in embeddings.data]  # Retorna lista de vetores Acessa corretamente o vetor gerado


db = Chroma(
    collection_name="pdf_chunks", 
    embedding_function=get_embeddings  # Agora compatível com Chroma
)

# Função para extrair texto de um arquivo PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Função para dividir o PDF e armazenar embeddings no ChromaDB
def store_pdf_in_chromadb(pdf_text):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", 
        chunk_size=60, 
        chunk_overlap=30
    )
    chunks = text_splitter.split_text(pdf_text)

    for i, chunk in enumerate(chunks):
        embedding = get_embeddings(chunk)  # Gera embedding via Databricks
        db.add_texts([chunk], metadatas=[{"index": i}], embeddings=[embedding])

# Função para buscar trechos mais relevantes no ChromaDB
def retrieve_relevant_text(question):
    query_embedding = get_embeddings(question)  # Gera embedding da pergunta
    results = db.similarity_search_by_vector(query_embedding, k=4)  # Busca no ChromaDB
    return "\n\n".join([doc.page_content for doc in results])

# Função para responder perguntas com base no PDF
def ask_question(question, conversation_chain):
    relevant_text = retrieve_relevant_text(question)

    # Adicionar mensagem de sistema com o prompt
    system_message = {
        "role": "system",
        "content": "Você é um assistente técnico agrícola. Suas respostas devem ser fáceis de entender e voltadas para agricultores. Responda apenas com base no PDF fornecido. Não mencione, por exemplo, 'Recomendo que você consulte um especialista em citricultura'."
    }
    conversation_chain.append_message("system", system_message["content"])

    # Passando o texto relevante para a cadeia de conversação
    conversation_chain.append_message("user", question)
    conversation_chain.append_message("assistant", relevant_text)

    # Obter a resposta gerada pelo LLM com base no histórico da conversa
    response = conversation_chain.predict(input=question)

    return response

# Interface do chatbot no Streamlit
def main():
    st.title("Agrônomo Virtual 🤖")

    uploaded_file = st.file_uploader("Escolha um arquivo PDF", type=["pdf"])

    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(uploaded_file)

        if not pdf_text.strip():
            st.error("O PDF não contém texto legível.")
            return

        store_pdf_in_chromadb(pdf_text)  # Armazena os embeddings do PDF

        if 'history' not in st.session_state:
            st.session_state.history = []

        # Criar uma instância do ConversationChain
        conversation_chain = ConversationChain(llm=client, verbose=True)

        st.subheader("Chat")

        for message in st.session_state.history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Digite sua pergunta sobre o PDF...")

        if question:
            with st.chat_message("user"):
                st.markdown(question)

            with st.spinner("Processando..."):
                response = ask_question(question, conversation_chain)
                st.session_state.history.append({"role": "user", "content": question})
                st.session_state.history.append({"role": "assistant", "content": response})

            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    main()
