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

question = st.text_input("Your question", placeholder="e.g. What are the five pillars of the AWS Well-Architected Framework?")

if st.button("Ask") and question.strip():
    vectorstore, _ = get_index()
    prompt = build_prompt()

    with st.spinner("Retrieving and generating answer..."):
        results = retrieve(question, vectorstore, k=3)
        context = "\n---\n".join(r.page_content for r in results)
        message = answer_question(question, context, prompt)
        answer = message.content[0].text

    st.success(answer)

    with st.expander("Sources"):
        for i, chunk in enumerate(results, 1):
            st.markdown(f"**Chunk {i}**")
            st.write(chunk.page_content)
            st.divider()
