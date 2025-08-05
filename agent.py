from __future__ import annotations
from typing import List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
import asyncio
import os

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


load_dotenv()

# ========== Define dependencies ==========
@dataclass
class GraphitiDependencies:
    """Dependencies for the Graphiti agent."""
    graphiti_client: Graphiti

# ========== Helper function to get model configuration (CORRECTED) ==========
def get_model():
    """Configure and return the LLM model object to use for the pydantic-ai agent."""
    # Use the local model you have running in Ollama
    model_choice = os.getenv('MODEL_CHOICE', 'qwen3:8b')

    # Point the provider to your local Ollama endpoint
    provider = OpenAIProvider(
        api_key="ollama", 
        base_url="http://localhost:11434/v1"
    )

    # Create and return a single, configured model object
    return OpenAIModel(model_name=model_choice, provider=provider)

# ========== Create the Graphiti agent (CORRECTED) ==========
graphiti_agent = Agent(
    get_model(), # This now passes the single, correct model object
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
    """Search the Graphiti knowledge graph with the given query.
    
    Args:
        ctx: The run context containing dependencies
        query: The search query to find information in the knowledge graph
        
    Returns:
        A list of search results containing facts that match the query
    """
    # Access the Graphiti client from dependencies
    graphiti = ctx.deps.graphiti_client
    
    try:
        # Perform the search
        results = await graphiti.search(query)
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = GraphitiSearchResult(
                uuid=result.uuid,
                fact=result.fact,
                source_node_uuid=result.source_node_uuid if hasattr(result, 'source_node_uuid') else None
            )
            
            # Add temporal information if available
            if hasattr(result, 'valid_at') and result.valid_at:
                formatted_result.valid_at = str(result.valid_at)
            if hasattr(result, 'invalid_at') and result.invalid_at:
                formatted_result.invalid_at = str(result.invalid_at)
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    except Exception as e:
        # Log the error but don't close the connection since it's managed by the dependency
        print(f"Error searching Graphiti: {str(e)}")
        raise

# ========== Main execution function ==========
async def main():
    """Run the Graphiti agent with user queries."""
    print("Graphiti Agent - Powered by Pydantic AI, Graphiti, and Neo4j")
    print("Enter 'exit' to quit the program.")

    # Neo4j connection parameters
    neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
    neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
    
    # --- GRAPHITI OLLAMA CONFIGURATION START ---
    llm_config = LLMConfig(
        api_key="ollama", model="qwen3:8b", small_model="qwen3:8b", base_url="http://localhost:11434/v1"
    )
    embedder_config = OpenAIEmbedderConfig(
        api_key="ollama", embedding_model="nomic-embed-text", embedding_dim=768, base_url="http://localhost:11434/v1"
    )
    graphiti_client = Graphiti(
        neo4j_uri, neo4j_user, neo4j_password,
        llm_client=OpenAIClient(config=llm_config),
        embedder=OpenAIEmbedder(config=embedder_config),
        cross_encoder=OpenAIRerankerClient(client=OpenAIClient(config=llm_config), config=llm_config),
    )
    # --- GRAPHITI OLLAMA CONFIGURATION END ---
    
    # Initialize the graph database with graphiti's indices if needed
    try:
        await graphiti_client.build_indices_and_constraints()
        print("Graphiti indices built successfully.")
    except Exception as e:
        print(f"Note: Some indices might already exist, which is normal. Error: {str(e)}")

    console = Console()
    messages = []
    
    try:
        while True:
            # Get user input
            user_input = input("\n[You] ")
            
            # Check if user wants to exit
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("Goodbye!")
                break
            
            try:
                # Process the user input and output the response
                print("\n[Assistant]")
                
                # Pass the Graphiti client as a dependency
                deps = GraphitiDependencies(graphiti_client=graphiti_client)
                
                # Use the non-streaming 'run' method for more robust tool calling
                result = await graphiti_agent.run(
                    user_input, message_history=messages, deps=deps
                )
                
                # Print the final output using the rich console
                console.print(Markdown(result.output))
                
                # Add the new messages to the chat history
                messages.extend(result.all_messages())
                
            except Exception as e:
                print(f"\n[Error] An error occurred: {str(e)}")
    finally:
        # Close the Graphiti connection when done
        await graphiti_client.close()
        print("\nGraphiti connection closed.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        raise