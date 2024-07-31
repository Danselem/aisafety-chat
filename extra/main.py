############################################
# Component #1 - Document Loader
############################################

import streamlit as st
import tempfile
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

dotenv_path = Path('./.env')
load_dotenv(dotenv_path=dotenv_path)

os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")


st.set_page_config(layout = "wide")

with st.sidebar:
    DOCS_DIR = os.path.abspath("./uploaded_docs")
    db_path = tempfile.mkdtemp()
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
    st.subheader("Add to the Knowledge Base")
    with st.form("my-form", clear_on_submit=True):
        # uploaded_files = st.file_uploader("Upload a file to the Knowledge Base:", accept_multiple_files = False, type="pdf")
        uploaded_file = st.file_uploader("Upload a file to the Knowledge Base:", accept_multiple_files = False, type="pdf")
        
        # pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
        # tmp_file_path = DOCS_DIR + '/' + uploaded_files.name
        # print(tmp_file_path)
        submitted = st.form_submit_button("Upload!")

    # if uploaded_files and submitted:
    #     for uploaded_file in uploaded_files:
    #         pdf_file = uploaded_file
    #         st.success(f"File {uploaded_file.name} uploaded successfully!")
            
    #         with open(os.path.join(DOCS_DIR, uploaded_file.name),"wb") as f:
    #             f.write(uploaded_file.read())
                
    if uploaded_file and submitted:
        
        pdf_file = uploaded_file
        st.success(f"File {uploaded_file.name} uploaded successfully!")
        # raw_documents = PyPDFLoader(uploaded_file.name).load()
            
        with open(os.path.join(DOCS_DIR, uploaded_file.name),"wb") as f:
            f.write(uploaded_file.read())

############################################
# Component #2 - Embedding Model and LLM
############################################

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# make sure to export your NVIDIA AI Playground key as NVIDIA_API_KEY!
# llm = ChatNVIDIA(model="ai-llama3-70b")
llm = ChatNVIDIA(model="nvidia/llama3-chatqa-1.5-70b")
document_embedder = NVIDIAEmbeddings(model="ai-embed-qa-4", model_type="passage")
query_embedder = NVIDIAEmbeddings(model="ai-embed-qa-4", model_type="query")

############################################
# Component #3 - Vector Database Store
############################################

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import FAISS
import pickle

with st.sidebar:
    # Option for using an existing vector store
    use_existing_vector_store = st.radio("Use existing vector store if available", ["Yes", "No"], horizontal=True)

# Path to the vector store file
vector_store_path = "vectorstore.pkl"

# Load raw documents from the directory
# raw_documents = DirectoryLoader(DOCS_DIR).load()
# raw_documents = ArxivLoader(DOCS_DIR).load()
raw_documents = PyPDFLoader(DOCS_DIR + "/"+ uploaded_file).load()


# Check for existing vector store file
vector_store_exists = os.path.exists(vector_store_path)
vectorstore = None
if use_existing_vector_store == "Yes" and vector_store_exists:
    with open(vector_store_path, "rb") as f:
        vectorstore = pickle.load(f)
    with st.sidebar:
        st.success("Existing vector store loaded successfully.")
else:
    with st.sidebar:
        if raw_documents:
            with st.spinner("Splitting documents into chunks..."):
                text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                documents = text_splitter.split_documents(raw_documents)

            with st.spinner("Adding document chunks to vector database..."):
                vectorstore = FAISS.from_documents(documents, document_embedder)

            with st.spinner("Saving vector store"):
                with open(vector_store_path, "wb") as f:
                    pickle.dump(vectorstore, f)
            st.success("Vector store created and saved.")
        else:
            st.warning("No documents available to process!", icon="⚠️")

############################################
# Component #4 - LLM Response Generation and Chat
############################################

st.subheader("Chat with your AI Assistant, Envie!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful AI assistant named Envie. You will reply to questions only based on the context that you are provided. If something is out of context, you will refrain from replying and politely decline to respond to the user."), ("user", "{input}")]
)
user_input = st.chat_input("Can you tell me what NVIDIA is known for?")
llm = ChatNVIDIA(model="ai-llama3-70b")

chain = prompt_template | llm | StrOutputParser()

if user_input and vectorstore!=None:
    st.session_state.messages.append({"role": "user", "content": user_input})
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    context = ""
    for doc in docs:
        context += doc.page_content + "\n\n"

    augmented_user_input = "Context: " + context + "\n\nQuestion: " + user_input + "\n"

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for response in chain.stream({"input": augmented_user_input}):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
