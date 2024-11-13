from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
import os
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveJsonSplitter
from langchain_groq import ChatGroq
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

graph = Neo4jGraph()

llm=ChatGroq(temperature=0, model_name="llama3-8b-8192")

# Read the wikipedia article
class URLInfo(BaseModel):
    content: str = Field(description="information extracted from the url")

json_parser = JsonOutputParser(pydantic_object=URLInfo)
testPrompt = prompt = PromptTemplate(
    template="You are extracting information about the website given like the host, ports, services, credentials, metadata, kwargs, arguments and other information from the url the user has provided.\n{format_instructions}\n{url}\n",
    input_variables=["url"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()},
)

chain = testPrompt | llm | json_parser

raw_documents = chain.invoke({"url": "https://github.com/github"})
print('raw',raw_documents, type(raw_documents))
# print(json.loads(raw_documents.content))

# Define chunking strategy
splitter = RecursiveJsonSplitter(max_chunk_size=300)
documents = splitter.create_documents(texts=[raw_documents])
print('docs', documents)

llm_transformer = LLMGraphTransformer(llm=llm)

graph_documents = llm_transformer.convert_to_graph_documents(documents)

# model_name = "BAAI/bge-small-en"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)

default_cypher = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50"

def showGraph(cypher: str = default_cypher):
    # create a neo4j session to run queries
    driver = GraphDatabase.driver(
        uri = os.environ["NEO4J_URI"],
        auth = (os.environ["NEO4J_USERNAME"],
                os.environ["NEO4J_PASSWORD"]))
    session = driver.session()
    widget = GraphWidget(graph = session.run(cypher).graph())
    widget.node_label_mapping = 'id'
    #display(widget)
    return widget

showGraph()

vector_index = Neo4jVector.from_existing_graph(
    hf,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

# Retriever

graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All host, ports, services, credentials, metadata, kwargs, arguments and other information that "
        "appear in the text",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting all host, ports, services, credentials, metadata, kwargs, arguments and other information that from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)

entity_chain = prompt | llm.with_structured_output(Entities)

res = entity_chain.invoke({"question": "What are the ports?"}).names
print(res)