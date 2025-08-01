from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

from langchain_community.embeddings import BedrockEmbeddings
import boto3

def get_titan_embeddings():
    client = boto3.client("bedrock-runtime", region_name="ap-south-1")
    return BedrockEmbeddings(
        client=client,
        model_id="cohere.embed-english-v3"
    )

from langchain_community.vectorstores import FAISS

def create_vector_store(docs):
    embeddings = get_titan_embeddings()
    vector_faiss=FAISS.from_documents(docs, embeddings)
    vector_faiss.save_local("vector store data")
    return vector_faiss

from langchain_community.llms import Bedrock

def create_bedrock_llm_mistral():
    client = boto3.client("bedrock-runtime", region_name="ap-south-1")
    return Bedrock(
        client=client,
        model_id="mistral.mistral-7b-instruct-v0:2",
        model_kwargs={
            "temperature": 0.7,
            "max_tokens": 512,
        },
    )

def create_bedrock_llm_llama():
    client = boto3.client("bedrock-runtime", region_name="ap-south-1")
    return Bedrock(
        client=client,
        model_id="meta.llama3-8b-instruct-v1:0",
        model_kwargs={
            "temperature": 0.7,
            "top_p": 0.9,
            "max_gen_len": 512,
        },
    )

from langchain.chains import RetrievalQA

def create_qa_chain(vector_store,llm):
    llm = llm
    retriever = vector_store.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

if __name__ == "__main__":
    file_path = r"test_data\History of India.pdf"
    question = "What is the purpose of this document?"

    model_choice = input("Choose model (mistral / llama): ").strip().lower()

    if model_choice == "llama":
        llm = create_bedrock_llm_llama()
    elif model_choice == "mistral":
        llm = create_bedrock_llm_mistral()
    else:
        print("Invalid model choice. Defaulting to Mistral.")
        llm = create_bedrock_llm_mistral()

    chunks = load_and_split(file_path)
    vector_store = create_vector_store(chunks)
    qa = create_qa_chain(vector_store,llm)

    answer = qa.run(question)
    print("🧠 Answer:", answer)
