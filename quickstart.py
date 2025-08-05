import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

# Import classes needed for Ollama configuration
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient


#################################################
# CONFIGURATION
#################################################
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

# Neo4j connection parameters (no changes here)
neo4j_uri = os.environ.get('NEO4J_URI', 'neo4j+s://9431dbf4.databases.neo4j.io')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')

if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')


async def main():
    #################################################
    # INITIALIZATION (MODIFIED FOR OLLAMA)
    #################################################

    # 1. Configure the LLM client to point to Ollama's OpenAI-compatible API
    #    Note: A dummy API key is required, but not used by Ollama.
    llm_config = LLMConfig(
        api_key="ollama",  # Ollama doesn't require a real API key
        model="qwen3:8b",
        small_model="qwen3:8b",
        base_url="http://localhost:11434/v1",  # Default Ollama API endpoint
    )
    llm_client = OpenAIClient(config=llm_config)

    # 2. Configure the Embedder to use a local model via Ollama
    embedder_config = OpenAIEmbedderConfig(
        api_key="ollama",
        embedding_model="nomic-embed-text",
        embedding_dim=768,  # Required for nomic-embed-text
        base_url="http://localhost:11434/v1",
    )
    embedder = OpenAIEmbedder(config=embedder_config)

    # 3. Configure the Cross-Encoder (reranker) to use the same Ollama LLM
    cross_encoder = OpenAIRerankerClient(client=llm_client, config=llm_config)

    # 4. Initialize Graphiti with the custom Ollama clients
    graphiti = Graphiti(
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )

    try:
        # The rest of the script remains the same
        await graphiti.build_indices_and_constraints()

        #################################################
        # ADDING EPISODES
        #################################################
        episodes = [
            {
                'content': 'Claude is the flagship AI assistant from Anthropic. It was previously '
                'known as Claude Instant in its earlier versions.',
                'type': EpisodeType.text,
                'description': 'AI podcast transcript',
            },
            {
                'content': 'As an AI assistant, Claude has been available since December 15, 2022 – Present',
                'type': EpisodeType.text,
                'description': 'AI podcast transcript',
            },
            {
                'content': {
                    'name': 'GPT-4',
                    'creator': 'OpenAI',
                    'capability': 'Multimodal Reasoning',
                    'previous_version': 'GPT-3.5',
                    'training_data_cutoff': 'April 2023',
                },
                'type': EpisodeType.json,
                'description': 'AI model metadata',
            },
            {
                'content': {
                    'name': 'GPT-4',
                    'release_date': 'March 14, 2023',
                    'context_window': '128,000 tokens',
                    'status': 'Active',
                },
                'type': EpisodeType.json,
                'description': 'AI model metadata',
            },
        ]

        for i, episode in enumerate(episodes):
            await graphiti.add_episode(
                name=f'AI Agents Unleashed {i}',
                episode_body=episode['content']
                if isinstance(episode['content'], str)
                else json.dumps(episode['content']),
                source=episode['type'],
                source_description=episode['description'],
                reference_time=datetime.now(timezone.utc),
            )
            print(f'Added episode: AI Agents Unleashed {i} ({episode["type"].value})')

        #################################################
        # BASIC SEARCH
        #################################################
        print("\nSearching for: 'Which AI assistant is from Anthropic?'")
        results = await graphiti.search('Which AI assistant is from Anthropic?')
        print('\nSearch Results:')
        for result in results:
            print(f'Fact: {result.fact}')
            print('---')

        #################################################
        # NODE SEARCH
        #################################################
        print(
            '\nPerforming node search using _search method with standard recipe NODE_HYBRID_SEARCH_RRF:'
        )
        node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_search_config.limit = 5
        node_search_results = await graphiti._search(
            query='Large Language Models',
            config=node_search_config,
        )
        print('\nNode Search Results:')
        for node in node_search_results.nodes:
            print(f'Node Name: {node.name}')
            print('---')

    finally:
        await graphiti.close()
        print('\nConnection closed')


if __name__ == '__main__':
    asyncio.run(main())