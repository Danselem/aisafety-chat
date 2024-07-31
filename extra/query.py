import os
from langchain_pinecone import PineconeVectorStore

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

import google.generativeai as genai


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

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


index_name = 'langchain-retrieval'

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", task_type="retrieval_query")

# embeddings = NVIDIAEmbeddings(
#     mmodel="nvidia/embed-qa-4", 
#     truncate="END", model_type="query")



prompt_template = """
You're an AI Specialist specialised in answering questions based on \
    the context provided to you. Kindly answer the question appropriately.
\n\n
  Context:\n {context}?\n
  Question: \n{question}\n

  Answer:
"""

prompt = PromptTemplate(template = prompt_template, 
                        input_variables = ["context", "question"])

# Define the LLM Pinecone index and LLM model

def data_querying(question):
    vstore = PineconeVectorStore.from_existing_index(index_name, embeddings)
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        convert_system_message_to_human=True, 
        temperature=0.6)

    chain = load_qa_chain(llm, 
                        chain_type="stuff", 
                        prompt=prompt) # map_reduce - stuff

    docs = vstore.similarity_search(question, 3)

    response = chain({"input_documents":docs, "question": question}, return_only_outputs=True)
    response_text = response.get('output_text')

    return response_text
    
    
question = "What is world models?"
print(data_querying(question))