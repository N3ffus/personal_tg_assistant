import logging

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from src.application.ports.knowledge import KnowledgeMemoryError
from src.application.services.knowledge import KnowledgeService
from src.config import Settings
from src.infrastructure.knowledge.graphiti_memory import GraphitiKnowledgeMemory

logger = logging.getLogger(__name__)

# The driver reports "property key does not exist" for every not-yet-written
# property, which floods the log on an empty graph.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


def build_graphiti(settings: Settings) -> Graphiti:
    """Build the single per-process Graphiti client from application settings."""
    llm_config = LLMConfig(
        api_key=settings.resolved_graphiti_llm_api_key,
        model=settings.resolved_graphiti_llm_model,
        base_url=settings.resolved_graphiti_llm_base_url,
    )
    return Graphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
        # The generic client speaks plain OpenAI-compatible chat completions, so
        # any provider configured for this project can drive extraction.
        llm_client=OpenAIGenericClient(
            config=llm_config,
            structured_output_mode=settings.graphiti_llm_structured_output,
        ),
        embedder=OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                embedding_model=settings.graphiti_embedding_model,
                embedding_dim=settings.graphiti_embedding_dim,
                api_key=settings.resolved_graphiti_embedding_api_key,
                base_url=settings.resolved_graphiti_embedding_base_url,
            )
        ),
        # Passed explicitly so Graphiti never falls back to the OPENAI_API_KEY
        # environment variable. Hybrid search itself reranks with RRF.
        cross_encoder=OpenAIRerankerClient(config=llm_config),
        max_coroutines=settings.graphiti_max_coroutines,
    )


async def create_knowledge_service(settings: Settings) -> KnowledgeService | None:
    """Create the shared knowledge memory, or ``None`` when it is unavailable.

    Knowledge memory is an optional feature: a missing or unreachable Neo4j must
    never stop the bot from starting.
    """
    if not settings.knowledge_memory_enabled:
        logger.info("knowledge.disabled")
        return None
    memory = GraphitiKnowledgeMemory(graphiti=build_graphiti(settings))
    try:
        await memory.initialize()
    except KnowledgeMemoryError as error:
        logger.error(
            "knowledge.startup.failed uri=%s reason=%s — the assistant starts "
            "without long-term memory",
            settings.neo4j_uri,
            error,
        )
        await _release(memory)
        return None
    logger.info("knowledge.startup.completed uri=%s", settings.neo4j_uri)
    return KnowledgeService(memory=memory)


async def _release(memory: GraphitiKnowledgeMemory) -> None:
    try:
        await memory.close()
    except Exception as error:  # pragma: no cover - best-effort cleanup
        logger.warning("knowledge.close.failed reason=%s", error)
