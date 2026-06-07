# frontend/chainlit_app.py
"""
Enhanced Chainlit demo UI for Agentic-RAG with improved answer presentation.
Run:
  chainlit run frontend/chainlit_app.py --port 8001
"""

import os
import sys
import requests
import chainlit as cl
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from backend.core.config import settings

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def backend_query(
    query: str,
    mode: str = "concise",
    max_tokens: int = settings.MAX_TOKENS,
    temperature: float = 0.0,
    top_k: int = settings.TOP_K_RETRIEVAL,
    bypass_cache: bool = False,
) -> dict:
    """
    Send a query to the FastAPI backend.

    bypass_cache=True adds a suffix to the query that prevents a semantic
    cache hit while leaving the actual question intact for retrieval.
    The backend strips this marker before processing.
    """
    url = f"{BACKEND_URL}/query"

    # Cache bypass: the backend checks for this flag and skips semantic cache.
    # We send it as a top-level field; the backend reads it before cache check.
    payload = {
        "query":        query,
        "mode":         mode,           # was missing — always defaulted to "concise"
        "max_tokens":   max_tokens,
        "temperature":  temperature,
        "top_k":        top_k,
        "bypass_cache": bypass_cache,   # new field — backend honours this
    }
    try:
        response = requests.post(url, json=payload, timeout=1000)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def upload_file_to_backend(file_element) -> bool:
    url = f"{BACKEND_URL}/ingest/upload"
    try:
        with open(file_element.path, "rb") as f:
            files = {"file": (file_element.name, f, file_element.mime)}
            response = requests.post(url, files=files, timeout=1000)
            response.raise_for_status()
        return True
    except Exception as e:
        print(f"Upload failed: {e}")
        return False


@cl.on_chat_start
async def start():
    await cl.Message(content=(
        "# Agentic RAG ready\n"
        "Ask me anything about the uploaded documents.\n\n"
        "**Attach PDF or TXT files to add them to the knowledge base.**"
    )).send()


@cl.on_message
async def main(message: cl.Message):
    # --- File uploads ---
    if message.elements:
        uploaded = []
        for element in message.elements:
            if isinstance(element, cl.File):
                ok = await cl.make_async(upload_file_to_backend)(element)
                if ok:
                    uploaded.append(element.name)
        if uploaded:
            await cl.Message(
                content=f"Uploaded: {', '.join(uploaded)}\n"
                        f"*Files are being indexed and will be searchable shortly.*"
            ).send()
        if not message.content:
            return

    user_query = message.content.strip()

    # --- Regular concise query (uses semantic cache) ---
    msg = cl.Message(content="Thinking...")
    await msg.send()

    resp = await cl.make_async(backend_query)(
        user_query,
        mode="concise",
        max_tokens=settings.MAX_TOKENS,
        temperature=0.0,
    )

    if "error" in resp:
        msg.content = f"Error: {resp['error']}"
        await msg.update()
        return

    answer  = resp.get("answer", "No answer generated.")
    sources = resp.get("sources", [])

    source_text = ""
    if sources:
        source_text = "\n\n---\n**📚 Sources:**\n"
        for i, s in enumerate(sources, 1):
            meta    = s.get("metadata") or s.get("meta") or {}
            text    = s.get("text") or meta.get("text") or ""
            preview = text[:150].replace("\n", " ") + "..."
            source_text += f"{i}. {preview}\n"

    msg.content = f"{answer}{source_text}"
    await msg.update()

    # --- Action buttons ---
    long_tokens = settings.MAX_TOKENS * 2

    actions = [
        cl.Action(
            name="long_answer",
            payload={"query": user_query, "max_tokens": long_tokens},
            label=f"Long answer ({long_tokens} tokens)",
        ),
        cl.Action(
            name="creative_answer",
            payload={"query": user_query},
            label="Creative answer",
        ),
    ]
    await cl.Message(content="Options:", actions=actions).send()


@cl.action_callback("long_answer")
async def on_long_answer(action):
    """
    Detailed answer — bypasses semantic cache and uses mode=detailed.

    FIX: original sent no mode field and no cache bypass, so the cache
    returned the concise answer regardless of max_tokens.
    """
    query      = action.payload["query"]
    req_tokens = action.payload["max_tokens"]

    msg = cl.Message(content="Generating detailed answer...")
    await msg.send()

    resp = await cl.make_async(backend_query)(
        query,
        mode="detailed",        # ← was missing
        max_tokens=req_tokens,
        temperature=0.0,
        bypass_cache=True,      # ← was missing — caused cache hit
    )

    if "error" in resp:
        msg.content = f"Error: {resp['error']}"
    else:
        answer      = resp.get("answer", "No answer.")
        sources     = resp.get("sources", [])
        source_text = ""
        if sources:
            source_text = "\n\n---\n**📚 Sources:**\n"
            for i, s in enumerate(sources, 1):
                meta    = s.get("metadata") or s.get("meta") or {}
                text    = s.get("text") or meta.get("text") or ""
                preview = text[:150].replace("\n", " ") + "..."
                source_text += f"{i}. {preview}\n"
        msg.content = f"**Detailed answer:**\n\n{answer}{source_text}"

    await msg.update()


@cl.action_callback("creative_answer")
async def on_creative_answer(action):
    """
    Creative answer — bypasses semantic cache, raises temperature.

    FIX: original sent no mode field and no cache bypass, so the cache
    returned the same concise answer regardless of temperature=0.7.
    """
    query = action.payload["query"]

    msg = cl.Message(content="Thinking creatively...")
    await msg.send()

    resp = await cl.make_async(backend_query)(
        query,
        mode="detailed",        # more expansive generation
        max_tokens=settings.MAX_TOKENS,
        temperature=0.7,
        bypass_cache=True,      # ← was missing — caused cache hit
    )

    if "error" in resp:
        msg.content = f"Error: {resp['error']}"
    else:
        answer      = resp.get("answer", "No answer.")
        sources     = resp.get("sources", [])
        source_text = ""
        if sources:
            source_text = "\n\n---\n**📚 Sources:**\n"
            for i, s in enumerate(sources, 1):
                meta    = s.get("metadata") or s.get("meta") or {}
                text    = s.get("text") or meta.get("text") or ""
                preview = text[:150].replace("\n", " ") + "..."
                source_text += f"{i}. {preview}\n"
        msg.content = f"**Creative answer:**\n\n{answer}{source_text}"

    await msg.update()