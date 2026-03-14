"""
prompt_builder.py — pack retrieved chunks into an LLM-ready prompt

Takes a list of chunk dicts from retriever.py and a user question,
builds a structured system prompt with XML source tags, caps context
at MAX_CONTEXT_TOKENS using tiktoken, and returns a messages list
ready to pass directly to the Groq client.

Single public function:
    build_messages(chunks, code_snippet, question) -> list[dict]

Run directly to preview a prompt without calling the LLM:
    python -m src.prompt_builder
"""

import tiktoken
from rich.console import Console

from src.config import MAX_CONTEXT_TOKENS

console = Console()

# ── Tokeniser ──────────────────────────────────────────────────
# cl100k_base is close enough to llama token counts for budget purposes
_enc = tiktoken.get_encoding("cl100k_base")

def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


# ── System prompt template ─────────────────────────────────────
# The LLM is instructed to:
#   1. Explain the code using ONLY the provided sources
#   2. Cite sources inline using [SOURCE: url] markers
#   3. Never hallucinate — say "not covered in sources" if unsure

SYSTEM_PROMPT_HEADER = """\
You are an expert Python educator specialising in asyncio and async programming.
Your job is to explain code snippets clearly and accurately using ONLY the \
reference documentation provided below.

Rules:
- Base every explanation on the <source> blocks provided. Do not invent information.
- Cite sources inline like this: [SOURCE: url]
- If the answer is not covered in the sources, say exactly:
  "This specific behaviour is not covered in the provided documentation."
- Structure your explanation as:
  1. What the code does (1-2 sentences)
  2. Key concepts explained (with citations)
  3. Common pitfalls or notes (if covered in sources)
- Keep the explanation concise and practical.

Reference documentation:
"""

SYSTEM_PROMPT_FOOTER = """
Use only the sources above. Cite them inline with [SOURCE: url].
"""

# Token budget breakdown
# Total cap: MAX_CONTEXT_TOKENS
# Reserved for header + footer + user turn overhead: ~400 tokens
# Remaining budget allocated to source chunks
OVERHEAD_TOKENS = 400


# ── Source block formatting ────────────────────────────────────

def _format_source_block(chunk: dict, index: int) -> str:
    """
    Format a single chunk as an XML source block.

    Example output:
        <source index="1" url="https://..." title="Coroutines and Tasks">
        Tasks are used to schedule coroutines concurrently...
        </source>
    """
    return (
        f'<source index="{index}" '
        f'url="{chunk["url"]}" '
        f'title="{chunk["title"]}">\n'
        f'{chunk["text"].strip()}\n'
        f'</source>'
    )


def _pack_sources(chunks: list[dict], token_budget: int) -> tuple[str, int]:
    """
    Pack as many chunks as fit within the token budget.
    Chunks are already sorted by relevance score (highest first).

    Returns:
        (sources_text, chunks_included_count)
    """
    blocks: list[str] = []
    tokens_used = 0

    for i, chunk in enumerate(chunks, start=1):
        block = _format_source_block(chunk, i)
        block_tokens = _count_tokens(block)

        if tokens_used + block_tokens > token_budget:
            # This chunk would exceed budget — skip remaining
            break

        blocks.append(block)
        tokens_used += block_tokens

    sources_text = "\n\n".join(blocks)
    return sources_text, len(blocks)


# ── User turn formatting ───────────────────────────────────────

def _format_user_turn(code_snippet: str | None, question: str) -> str:
    """
    Build the user message content.
    Code snippet is optional — questions without code are valid.
    """
    parts: list[str] = []

    if code_snippet and code_snippet.strip():
        parts.append(
            f"Here is the code I want to understand:\n\n"
            f"```python\n{code_snippet.strip()}\n```"
        )

    parts.append(f"Question: {question.strip()}")
    return "\n\n".join(parts)


# ── Main builder ───────────────────────────────────────────────

def build_messages(
    chunks: list[dict],
    question: str,
    code_snippet: str | None = None,
) -> list[dict]:
    """
    Build the messages list for the Groq chat completion API.

    Args:
        chunks:       ranked chunks from retriever.py (highest score first)
        question:     the user's question about the code
        code_snippet: optional code the user wants explained

    Returns:
        list of message dicts:
        [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "..."},
        ]
    """
    if not chunks:
        # No context retrieved — still attempt but warn the LLM
        system_content = (
            SYSTEM_PROMPT_HEADER.strip()
            + "\n\n<source>No relevant documentation found.</source>\n"
            + SYSTEM_PROMPT_FOOTER.strip()
        )
        user_content = _format_user_turn(code_snippet, question)
        return [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content},
        ]

    # Calculate token budget for source chunks
    header_tokens    = _count_tokens(SYSTEM_PROMPT_HEADER + SYSTEM_PROMPT_FOOTER)
    user_tokens      = _count_tokens(_format_user_turn(code_snippet, question))
    available_budget = MAX_CONTEXT_TOKENS - header_tokens - user_tokens - OVERHEAD_TOKENS

    if available_budget <= 0:
        console.print(
            "[yellow]Warning:[/yellow] question + code snippet is very long, "
            "leaving little room for source context."
        )
        available_budget = MAX_CONTEXT_TOKENS // 2

    # Pack sources into the budget
    sources_text, n_included = _pack_sources(chunks, available_budget)

    if n_included < len(chunks):
        console.print(
            f"[dim]Context cap: included {n_included}/{len(chunks)} chunks "
            f"(budget={available_budget} tokens)[/dim]"
        )

    system_content = (
        SYSTEM_PROMPT_HEADER
        + sources_text
        + SYSTEM_PROMPT_FOOTER
    )
    user_content = _format_user_turn(code_snippet, question)

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]


def count_prompt_tokens(messages: list[dict]) -> int:
    """Return total token count across all messages. Used for logging."""
    return sum(_count_tokens(m["content"]) for m in messages)


# ── CLI preview ────────────────────────────────────────────────

if __name__ == "__main__":
    # Dummy chunks to preview prompt structure without needing Pinecone
    dummy_chunks = [
        {
            "chunk_id": "asyncio-task_001",
            "text": (
                "asyncio.gather(*aws, return_exceptions=False)\n\n"
                "Run awaitable objects in the aws sequence concurrently.\n"
                "If any awaitable in aws is a coroutine, it is automatically "
                "scheduled as a Task. If all awaitables are completed successfully, "
                "the result is an aggregate list of returned values."
            ),
            "url":      "https://docs.python.org/3/library/asyncio-task.html",
            "title":    "Coroutines and Tasks",
            "has_code": True,
            "score":    0.91,
        },
        {
            "chunk_id": "asyncio-task_002",
            "text": (
                "If return_exceptions is False (default), the first raised exception "
                "is immediately propagated to the task that awaits on gather(). "
                "Other awaitables in the aws sequence won't be cancelled and will "
                "continue to run."
            ),
            "url":      "https://docs.python.org/3/library/asyncio-task.html",
            "title":    "Coroutines and Tasks",
            "has_code": False,
            "score":    0.84,
        },
    ]

    dummy_code = """\
async def fetch_all(urls):
    results = await asyncio.gather(*[fetch(url) for url in urls])
    return results
"""
    dummy_question = "What happens if one of the fetches raises an exception?"

    messages = build_messages(
        chunks=dummy_chunks,
        question=dummy_question,
        code_snippet=dummy_code,
    )

    total_tokens = count_prompt_tokens(messages)

    console.print("\n[bold]Prompt preview[/bold]\n")
    for msg in messages:
        console.rule(f"[dim]{msg['role']}[/dim]")
        console.print(msg["content"])

    console.print()
    console.rule()
    console.print(f"[dim]Total prompt tokens: {total_tokens}[/dim]")
    console.print(f"[dim]Token cap: {MAX_CONTEXT_TOKENS}[/dim]")
    console.print(f"[dim]Headroom: {MAX_CONTEXT_TOKENS - total_tokens} tokens[/dim]\n")