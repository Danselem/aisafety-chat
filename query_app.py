import os

from fastapi import FastAPI
from langserve import add_routes
from langchain_pinecone import PineconeVectorStore


from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings


from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

from langchain.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_community.document_transformers import LongContextReorder


from operator import itemgetter
import streamlit as st


from pathlib import Path
from dotenv import load_dotenv


import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')

dotenv_path = Path('./.env')
load_dotenv(dotenv_path=dotenv_path)
# https://build.nvidia.com/explore/discover
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ['GRPC_VERBOSITY'] = 'ERROR'

st.set_page_config(layout = "wide")

index_name = 'langchain-nvidia'

llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")
query_embedder = NVIDIAEmbeddings(model="NV-Embed-QA", 
                                     truncate="END", 
                                     model_type="query")

vectorstore = PineconeVectorStore.from_existing_index(index_name, query_embedder)




############################################
# Component #4 - LLM Response Generation and Chat
############################################

st.subheader("Chat with your AI Assistant about AI Safety!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        
prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful AI assistant named Envie. You will reply to questions only based on the context that you are provided. If something is out of context, you will refrain from replying and politely decline to respond to the user."), ("user", "{input}")]
)

user_input = st.chat_input("What do you want to know about AI Safety?")
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

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