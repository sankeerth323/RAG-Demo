import streamlit as st
from response import answer_question

st.set_page_config(
    page_title="Business FAQ RAG Demo",
    page_icon="💬",
    layout="centered"
)

st.title("Business FAQ RAG Demo")
st.write("Retrieval-Augmented Generation using FAISS + Groq LLaMA-3")

# Input box
query = st.text_input("Ask a question")

# Button to trigger inference
if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = answer_question(query)
        st.success("Answer")
        st.write(answer)
