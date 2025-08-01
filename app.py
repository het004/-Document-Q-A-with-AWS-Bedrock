import streamlit as st
from main import (
    load_and_split,
    create_vector_store,
    create_qa_chain,
    create_bedrock_llm_mistral,
    create_bedrock_llm_llama,
)

st.title("📄 Document Q&A with AWS Bedrock")

model_name = st.selectbox("Select LLM model:", ["Mistral", "LLaMA"])

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
question = st.text_input("Ask a question:")

if uploaded_file and question:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    
    if model_name == "Mistral":
        llm = create_bedrock_llm_mistral()
    else:
        llm = create_bedrock_llm_llama()

    
    chunks = load_and_split("temp.pdf")
    vector_store = create_vector_store(chunks)
    qa_chain = create_qa_chain(vector_store, llm)
    answer = qa_chain.run(question)

    st.markdown("### 🧠 Answer:")
    st.write(answer)
