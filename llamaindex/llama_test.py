#My first attempt at RAG implementation
#following HuggingFace's Agents Course Unity 2.2
#on llamaindex.

import asyncio, chromadb, dotenv
from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.ollama import Ollama

#Load documents from directory
print("Loading documents...")
reader = SimpleDirectoryReader(input_dir="./example_docs")
documents = reader.load_data()
print("Ok!")

#Instantiate db to store vector embeddings
print("Instantiating ChromaDB...")
db = chromadb.PersistentClient(path="./chromadb/example_chroma_db")
chroma_collection = db.get_or_create_collection("example")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
print("Ok!")

#create nodes through
#1) breaking down documents into chunks
#2) convert chunks into vector representations
print("Setting up pipeline...")
pipeline = IngestionPipeline(transformations=[
    SentenceSplitter(chunk_overlap=0),
    HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ],
vector_store=vector_store,
)
print("OK!")

print("Feeding documents to pipeline...")
pipeline.run(documents=documents,)
print("OK!")

#Create index to embed queries in same space than nodes
print("Creating index...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
print("OK!")
print("Instantiating model...")

#You have credits at HF! =D
#dotenv.load_dotenv() #To load HF_TOKEN, or you could login()
#llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

#You have exceeded your monthly included credits :/
llm = Ollama(model="qwen3:1.7b", #works fine without gpu if...
    max_output_tokens=100,
    temperature=0.3,
    #context_window=12000,
    request_timeout=3600.0)     #...you set a high timeout

print("OK!")
print("Querying engine...")
query_engine = index.as_query_engine(
    llm=llm,
    response_mode="simple_summarize",
    similarity_top_k=1
)

print(query_engine.query("Explique quem é Anderson Soares Freixo em português"))
