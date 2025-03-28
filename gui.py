# gui.py

import os
import shutil
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from rag_engine import RAGEngine

# Paths
TEMP_DOCS = "temp_docs"
PERSISTENT_DOCS = "storage/persistent_docs"

# Create folders
os.makedirs(TEMP_DOCS, exist_ok=True)
os.makedirs(PERSISTENT_DOCS, exist_ok=True)

# Load LLM
@st.cache_resource
def load_llm():
    model_name = "Qwen/Qwen1.5-1.8B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

# RAG Engine
rag = RAGEngine()
rag.load_index()

# Chat function
def chat_with_context(prompt, context, tokenizer, model, device):
    full_prompt = f"Context:\n{context}\n\nUser: {prompt}\nBot (responde en Markdown):"
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(full_prompt, "").strip()

# Streamlit GUI
def run():
    st.set_page_config(page_title="Local RAG Chat", layout="wide")
    st.title("🧠 Local RAG Chatbot")
    st.caption("Corre completamente en local. ¡Habla con tus PDFs con privacidad!")

    tokenizer, model, device = load_llm()

    # Estado inicial
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "generating" not in st.session_state:
        st.session_state.generating = False

    if "pending_input" not in st.session_state:
        st.session_state.pending_input = ""

    # Subida de PDF
    with st.sidebar:
        st.subheader("📄 Subir PDF")
        uploaded = st.file_uploader("Elige un PDF", type="pdf")
        keep_doc = st.checkbox("Guardar permanentemente", value=True)

        if uploaded:
            save_path = os.path.join(PERSISTENT_DOCS if keep_doc else TEMP_DOCS, uploaded.name)
            with open(save_path, "wb") as f:
                f.write(uploaded.read())
            rag.add_document(save_path, persistent=keep_doc)
            st.success(f"'{uploaded.name}' subido {'(permanente)' if keep_doc else '(temporal)'}.")

        if st.button("🗑️ Borrar documentos temporales"):
            shutil.rmtree(TEMP_DOCS)
            os.makedirs(TEMP_DOCS, exist_ok=True)
            st.success("Documentos temporales eliminados.")

    # Chat cancelable
    if st.session_state.generating:
        if st.button("❌ Cancelar generación"):
            st.session_state.generating = False
            st.session_state.chat_input = st.session_state.pending_input
            st.rerun()
    else:
        user_input = st.chat_input("Haz una pregunta sobre tus documentos...", key="chat_input")

        if user_input:
            st.session_state.generating = True
            st.session_state.pending_input = user_input
            st.session_state.chat_history.append(("You", user_input))

            # Mensaje de "Pensando..."
            with st.chat_message("Bot"):
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown("💭 Pensando...")

            try:
                context = rag.retrieve_context(user_input)
                response = chat_with_context(user_input, context, tokenizer, model, device)

                if st.session_state.generating:
                    thinking_placeholder.markdown(response, unsafe_allow_html=True)
                    st.session_state.chat_history.append(("Bot", response))

            except Exception as e:
                thinking_placeholder.markdown("⚠️ Error al generar respuesta.")
                st.session_state.chat_history.append(("Bot", "⚠️ Error en el modelo."))

            st.session_state.generating = False
            st.rerun()

    # Mostrar historial
    for sender, msg in st.session_state.chat_history:
        with st.chat_message(sender):
            if sender == "Bot":
                st.markdown(msg, unsafe_allow_html=True)
            else:
                st.markdown(f"**{msg}**")
