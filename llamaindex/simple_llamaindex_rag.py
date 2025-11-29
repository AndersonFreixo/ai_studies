#My first attempt at RAG implementation using LlamaIndex
#It consists of a script that loads everything necessary
#to run the index as a query engine and then just loops
#getting queries from the user.

import asyncio, chromadb, dotenv
from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI #only if using model through HF API
from llama_index.llms.ollama import Ollama #only if using local model

CHROMA_PATH = "./chromadb/example_chroma_db"
CHROMA_COLLECTION_NAME = "example"
DOCS_PATH = "./example_docs"

def get_chroma_store(path, collection_name):
    """Instantiate chroma db client from 'path' and open collection 'collection_name'
    return a ChromaVectorStore object"""
    db = chromadb.PersistentClient(path=path)
    chroma_collection = db.get_or_create_collection(collection_name)
    return ChromaVectorStore(chroma_collection=chroma_collection)


def ingest_docs(docs_path, vector_store):
    """Retrieve documents in 'docs_path' and feed them to an IngestionPipeline.
    The embeddings are stores in 'vector_store'"""

    #create nodes through
    #1) breaking down documents into chunks
    #2) convert chunks into vector representations
    print("Setting up ingestion pipeline...")
    pipeline = IngestionPipeline(transformations=[
        SentenceSplitter(chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        ],
        vector_store=vector_store,
    )
    print("OK!")
    #Load documents from directory
    print("Loading documents...")
    reader = SimpleDirectoryReader(input_dir=docs_path)
    documents = reader.load_data()
    print("Ok!")
    print("Feeding documents to pipeline...")
    pipeline.run(documents=documents,)
    print("OK!")


if __name__ == "__main__":
    print("Getting vector store from ChromaDB...")
    vector_store = get_chroma_store(CHROMA_PATH, CHROMA_COLLECTION_NAME)
    print("Ok!")
    #You don't need to run the ingestion pipeline
    #after first run unless you add more docs
    ingest_docs(DOCS_PATH, vector_store)
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
        request_timeout=3600.0)     #...you set a high timeout and is willing to wait.
    print("OK!")

    print("Instantiating query engine...")
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="simple_summarize",
        similarity_top_k=1
        )
    print("OK!")

    while (query := input("Ask me something: ")) != "EXIT":
        print(query_engine.query(query))
        print()



