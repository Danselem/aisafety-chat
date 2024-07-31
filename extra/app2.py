import numpy as np
import pandas as pd
import configparser
import streamlit as st
import streamlit
import chromadb
import langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import tempfile
import shutil
import os
import openai

# you need your own API key and ini file
#config = configparser.ConfigParser()
#config.read('config.ini')
api_key = "your_openai_api_key_here" #api_key = config['DEFAULT']['OPENAI_API_KEY']

if not api_key:
    st.error("I can not find API key")
else:
    openai.api_key = api_key

    st.title('PDF Documents Q&A App')

    uploaded_file = st.file_uploader("Please upload your PDF file", type=["pdf"])
    if uploaded_file is not None:
        # save as tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            shutil.copyfileobj(uploaded_file, tmpfile)
            file_path = tmpfile.name

        # PyPDFLoader
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()

        # model and vectorstore
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".")
        vectorstore.persist()

        # Q&A chain
        pdf_qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

        # get question from user
        question = st.text_input("Please input your question:")
        if st.button('Get Answer'):
            result = pdf_qa({"question": question, "chat_history": []})
            answer = result["answer"]
            st.write("Answer:", answer)

        # delete tempfile
        if file_path:
            os.unlink(file_path)