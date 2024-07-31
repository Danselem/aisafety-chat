import openai
import langchain
import pinecone 
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec, PodSpec  
import time 

import os
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

# index_name="llamaindex-rag-fs"


use_serverless = True  

# configure client  
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))  
if use_serverless:  
    spec = ServerlessSpec(cloud='aws', region='us-east-1')  
else:  
    # if not using a starter index, you should specify a pod_type too  
    spec = PodSpec()  
# check for and delete index if already exists  
index_name = 'langchain-nvidia'  
if index_name in pc.list_indexes().names():  
    pc.delete_index(index_name)  
# create a new index  
pc.create_index(  
    index_name,  
    dimension=1024,  # dimensionality of NVIDIA Embeddings  
    metric='cosine',  
    spec=spec  
)  
# wait for index to be initialized  
while not pc.describe_index(index_name).status['ready']:  
    time.sleep(1)  

index = pc.Index(index_name)  
print(index.describe_index_stats())

## Lets Read the document
def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents


raw_documents = read_doc('uploaded_docs/')
print(len(raw_documents))

# def chunk_data(docs,chunk_size=800,chunk_overlap=50):
#     text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
#     doc=text_splitter.split_documents(docs)
#     return docs

def chunk_data(docs,chunk_size=2000,chunk_overlap=200):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(docs)
    return documents

documents=chunk_data(docs=raw_documents)
print(len(documents))

# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
document_embedder = NVIDIAEmbeddings(model="NV-Embed-QA", 
                                     truncate="END", model_type="passage")



vectorstore = PineconeVectorStore.from_documents(
        documents,
        index_name=index_name,
        embedding=document_embedder
    )

query = "What is the title of the document?"
print(vectorstore.similarity_search(query))