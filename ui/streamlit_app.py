import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import streamlit as st
from retriever import load_index, retrieve
from chain import build_prompt, answer_question


@st.cache_resource
def get_index():
    return load_index()


st.title("RAG Chatbot")
st.write("Ask a question and get answers grounded in your documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask a question...")

if question:
    with st.chat_message("user"):
        st.write(question)

    vectorstore, _ = get_index()
    prompt = build_prompt(st.session_state.messages)

    with st.spinner("Retrieving and generating answer..."):
        results = retrieve(question, vectorstore, k=3)
        context = "\n---\n".join(r.page_content for r in results)
        message = answer_question(question, context, prompt)
        answer = message.content[0].text

    with st.chat_message("assistant"):
        st.write(answer)
        with st.expander("Sources"):
            for i, chunk in enumerate(results, 1):
                st.markdown(f"**Chunk {i}**")
                st.write(chunk.page_content)
                st.divider()

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer})
