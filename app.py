import streamlit as st
import os

from utils.document_loader import load_and_split_documents
from utils.vectorstore import VectorStore
from utils.web_search import search_web
from models.llm import get_llm

st.set_page_config(page_title="AI Research Assistant")

st.title("AI Research Assistant")

llm = get_llm()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = VectorStore()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

mode = st.radio("Response Mode", ["Concise", "Detailed"])

uploaded_file = st.file_uploader("Upload Research PDF", type=["pdf"])

if uploaded_file:

    # Create folder automatically if it doesn't exist
    os.makedirs("data/documents", exist_ok=True)

    file_path = f"data/documents/{uploaded_file.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    documents = load_and_split_documents(file_path)

    st.session_state.vectorstore.add_documents(documents)

    st.success("Document processed successfully!")

question = st.text_input("Ask your question")

if question:

    docs = st.session_state.vectorstore.similarity_search(question)

    if docs:

        context = "\n".join([doc["text"] for doc in docs])
        source = "PDF Document"

    else:

        context = search_web(question)
        source = "Web Search"

    if mode == "Concise":

        prompt = f"""
        Answer briefly in 2-3 sentences.

        Context:
        {context}

        Question:
        {question}
        """

    else:

        prompt = f"""
        Provide a detailed explanation.

        Context:
        {context}

        Question:
        {question}
        """

    response = llm.invoke(prompt)

    st.write(response.content)

    st.write(f"Source: {source}")

    if docs:

        st.subheader("Document Sources")

        for doc in docs:

            page = doc["metadata"].get("page", "Unknown")

            st.write(f"Page {page + 1}")

    st.session_state.chat_history.append((question, response.content))

st.divider()

st.subheader("Chat History")

for q, a in st.session_state.chat_history:

    st.write("Q:", q)
    st.write("A:", a)