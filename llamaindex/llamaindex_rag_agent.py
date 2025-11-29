import simple_llamaindex_rag as slr
import asyncio
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context

MODEL_NAME = "qwen3:0.6b"

async def run_agent(agent):
    ctx = Context(agent)
    while (query := input(">>")) != "EXIT":
        print(await agent.run(query, ctx=ctx))

if __name__ == "__main__":
    #Load previously vector embedded documents
    vector_store = slr.get_chroma_store("./chromadb/example_chroma_db", "example")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    #If you want to add new documents to the vector store, use
    #slr.ingest_docs(PATH_TO_DOCS, vector_store, embed_model)
    index = slr.get_index(vector_store, embed_model)

    llm = Ollama(model=MODEL_NAME,
        max_output_tokens=100,
        temperature=0.3,
        request_timeout=3600.0)

    #Create query engine from index
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="simple_summarize",
        similarity_top_k=1
    )
    #Create a tool from the query engine
    query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="Curriculum retrieving tool",
        description="Retrieves information of candidates based on their curriculum.",
        return_direct=False,
    )
    #Create agent from the query engine tool
    query_engine_agent = AgentWorkflow.from_tools_or_functions(
        [query_engine_tool],
        llm=llm,
        system_prompt="You are a helpful assistant that has access to a database containing curriculums from many candidates to several different positions in a company.")

    print(f"Running model {MODEL_NAME} locally...")
    print("Type EXIT to quit the program.")
    asyncio.run(run_agent(query_engine_agent))


