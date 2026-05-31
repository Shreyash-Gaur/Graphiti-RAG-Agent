# chainlit_app.py
from __future__ import annotations
from typing import List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import chainlit as cl
import nest_asyncio

# Pydantic AI and Graphiti Imports
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, RunContext
from graphiti_core import Graphiti

# Imports for Graphiti Ollama configuration
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

# Apply nest_asyncio to allow running asyncio in a running event loop
nest_asyncio.apply()
load_dotenv()

# ========== Define dependencies ==========
@dataclass
class GraphitiDependencies:
    """Dependencies for the Graphiti agent."""
    graphiti_client: Graphiti

# ========== Helper function to get model configuration ==========
def get_model():
    """Configure and return the LLM model object to use for the pydantic-ai agent."""
    model_choice = os.getenv('MODEL_CHOICE', 'qwen3:8b')
    provider = OpenAIProvider(api_key="ollama", base_url="http://localhost:11434/v1")
    return OpenAIModel(model_name=model_choice, provider=provider)

# ========== Create the Graphiti agent ==========
graphiti_agent = Agent(
    get_model(),
    system_prompt="""You are a specialized assistant whose only source of information is a knowledge graph.
    **You must not use your own pre-trained knowledge.**
    When the user asks a question, you **must** use the `search_graphiti` tool to find the answer.
    Base your answer *exclusively* on the facts retrieved from the tool.
    If the tool does not provide an answer, state that you could not find the information in the knowledge graph.""",
    deps_type=GraphitiDependencies
)

# ========== Define a result model for Graphiti search ==========
class GraphitiSearchResult(BaseModel):
    """Model representing a search result from Graphiti."""
    uuid: str = Field(description="The unique identifier for this fact")
    fact: str = Field(description="The factual statement retrieved from the knowledge graph")
    valid_at: Optional[str] = Field(None, description="When this fact became valid (if known)")
    invalid_at: Optional[str] = Field(None, description="When this fact became invalid (if known)")
    source_node_uuid: Optional[str] = Field(None, description="UUID of the source node")

# ========== Graphiti search tool ==========
@graphiti_agent.tool
async def search_graphiti(ctx: RunContext[GraphitiDependencies], query: str) -> List[GraphitiSearchResult]:
    """Search the Graphiti knowledge graph with the given query."""
    graphiti = ctx.deps.graphiti_client
    try:
        results = await graphiti.search(query)
        formatted_results = []
        for result in results:
            formatted_result = GraphitiSearchResult(
                uuid=result.uuid,
                fact=result.fact,
                source_node_uuid=result.source_node_uuid if hasattr(result, 'source_node_uuid') else None
            )
            if hasattr(result, 'valid_at') and result.valid_at:
                formatted_result.valid_at = str(result.valid_at)
            if hasattr(result, 'invalid_at') and result.invalid_at:
                formatted_result.invalid_at = str(result.invalid_at)
            formatted_results.append(formatted_result)
        return formatted_results
    except Exception as e:
        print(f"Error searching Graphiti: {str(e)}")
        raise

# ========== Chainlit Integration ==========

@cl.on_chat_start
async def on_chat_start():
    """Initializes the Graphiti agent when a new chat session starts."""
    cl.user_session.set("messages", [])
    
    # Neo4j connection parameters
    neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
    neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
    
    # --- GRAPHITI OLLAMA CONFIGURATION ---
    llm_config = LLMConfig(api_key="ollama", model="qwen3:8b", small_model="qwen3:8b", base_url="http://localhost:11434/v1")
    embedder_config = OpenAIEmbedderConfig(api_key="ollama", embedding_model="nomic-embed-text", embedding_dim=768, base_url="http://localhost:11434/v1")
    
    try:
        graphiti_client = Graphiti(
            neo4j_uri, neo4j_user, neo4j_password,
            llm_client=OpenAIClient(config=llm_config),
            embedder=OpenAIEmbedder(config=embedder_config),
            cross_encoder=OpenAIRerankerClient(client=OpenAIClient(config=llm_config), config=llm_config),
        )
        await graphiti_client.build_indices_and_constraints()
        cl.user_session.set("graphiti_client", graphiti_client)
        await cl.Message(content="Graphiti agent is ready! Ask me a question.").send()
    except Exception as e:
        await cl.Message(content=f"Error initializing Graphiti: {str(e)}").send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handles incoming messages from the user."""
    messages = cl.user_session.get("messages")
    graphiti_client = cl.user_session.get("graphiti_client")

    if not graphiti_client:
        await cl.Message(content="Graphiti client not initialized. Please restart the chat.").send()
        return

    deps = GraphitiDependencies(graphiti_client=graphiti_client)
    
    try:
        result = await graphiti_agent.run(
            message.content, message_history=messages, deps=deps
        )
        
        # Add new messages to chat history
        messages.extend(result.all_messages())
        cl.user_session.set("messages", messages)
        
        # Send the final output to the user
        await cl.Message(content=result.output).send()
        
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()

@cl.on_chat_end
async def on_chat_end():
    """Closes the Graphiti connection when the chat session ends."""
    graphiti_client = cl.user_session.get("graphiti_client")
    if graphiti_client:
        await graphiti_client.close()
        print("Graphiti connection closed.")