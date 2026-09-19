import os
import numpy as np
import pymupdf
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import torch.nn as nn

PDF_PATH = "Northgate_Solutions_HR_Policy_Manual.pdf"

@st.cache_resource(show_spinner="Loading the document and building the search index...")
def load_knowledge_base():
    client = Groq(api_key="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")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    doc = pymupdf.open(PDF_PATH)
    chunks = []
    for page in doc:
        text = page.get_text()
        for i in range(0, len(text), 450):
            chunk = text[i:i + 500].strip()
            if chunk:
                chunks.append(chunk)

    chunk_embeddings = embedder.encode(chunks, normalize_embeddings=True)
    return client, embedder, chunks, chunk_embeddings


client, embedder, chunks, chunk_embeddings = load_knowledge_base()

st.title("ChatBot")

system_prompt = (
    "You are a HR assistant. Answer only using the context given with each "
    "question. "
    "If the person greets you (e.g. 'hi', 'hello'), reply with a friendly "
    "greeting and ask how you can help. "
    "If the answer is not found in the context, do not say 'I don't know'. "
    "Instead, politely let them know this specific detail isn't covered in "
    "the document you have access to, and suggest they reach out to HR "
    "directly for more details - keep the tone warm and helpful, not "
    "dismissive."
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

prompt = st.chat_input("Ask me anything about the document")

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    question_embedding = embedder.encode([prompt], normalize_embeddings=True)[0]
    scores = chunk_embeddings @ question_embedding
    top_chunks = [chunks[i] for i in np.argsort(scores)[::-1][:5]]
    context = "\n\n".join(top_chunks)

    api_messages = st.session_state.messages[:-1] + [{
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {prompt}"
    }]

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=api_messages
    )

    response = completion.choices[0].message.content

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    with st.chat_message("assistant"):
        st.markdown(response)
