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
from langchain.document_transformers import LongContextReorder


from operator import itemgetter


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



app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain’s Runnable interfaces",
)


index_name = 'langchain-nvidia'

chat_prompt = ChatPromptTemplate.from_messages([("system",
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User messaged just asked you a question: {input}\n\n"
    " The following information may be useful for your response: "
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Make your response conversational)"
    ), ("user", "{input}")])

llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")
query_embedder = NVIDIAEmbeddings(model="NV-Embed-QA", 
                                     truncate="END", 
                                     model_type="query")

def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string. Optional, but useful"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name: out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

def output_puller(inputs):
    """"Output generator. Useful if your chain returns a dictionary with key 'output'"""
    for token in inputs:
        if token.get('output'):
            yield token.get('output')
            
docstore = PineconeVectorStore.from_existing_index(index_name, query_embedder)

long_reorder = RunnableLambda(LongContextReorder().transform_documents) ## GIVEN
context_getter = itemgetter("input") | docstore.as_retriever() | long_reorder | docs2str
retrieval_chain = {"input" : (lambda x: x)} | RunnableAssign({"context" : context_getter})

generator_chain = RunnableAssign({"output" : chat_prompt | llm }) ## TODO
generator_chain = generator_chain | output_puller ## GIVEN

rag_chain = retrieval_chain | generator_chain


# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001", task_type="retrieval_query")


add_routes(
    app,
    llm,
    path="/basic_chat",
)

add_routes(
    app,
    retrieval_chain,
    path="/retriever",
)

add_routes(
    app,
    generator_chain,
    path="/generator",
)

# add_routes(
#     app,
#     rag_chain,
#     path="/rag",
# )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9012)
