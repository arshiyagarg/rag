"""
generator.py — call Groq LLM and parse the response

Takes a messages list from prompt_builder.py, calls the Groq API
with llama-3.3-70b-versatile, streams the response, extracts cited
source URLs, and returns a structured result dict.

Single public function:
    generate(messages) -> dict

Returned dict:
{
    "explanation":  "...",           # full LLM response text
    "sources":      ["url1", ...],   # unique URLs cited in the response
    "input_tokens":  412,
    "output_tokens": 284,
    "model":        "llama-3.3-70b-versatile"
}

Run directly to test generation with a dummy prompt:
    python -m src.generator
"""

import re
from groq import Groq
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from groq import RateLimitError, APIStatusError
from rich.console import Console
from rich.markdown import Markdown

from src.config import GROQ_API_KEY, GROQ_CHAT_MODEL, MAX_CONTEXT_TOKENS

console = Console()

# ── Client ─────────────────────────────────────────────────────
_client = Groq(api_key=GROQ_API_KEY)

# Max tokens to generate in the response
# Groq free tier context window for llama-3.3-70b is 128k
# We reserve 1024 for the explanation — enough for a detailed answer
MAX_OUTPUT_TOKENS = 1024

# Pattern to extract [SOURCE: url] citations from LLM output
SOURCE_CITATION_RE = re.compile(r"\[SOURCE:\s*(https?://[^\]]+)\]")


# ── Citation extraction ────────────────────────────────────────

def extract_sources(text: str) -> list[str]:
    """
    Extract unique cited URLs from the LLM response.
    Matches [SOURCE: https://...] patterns inserted per the system prompt.
    Preserves order of first appearance.
    """
    seen:    set[str]  = set()
    sources: list[str] = []
    for url in SOURCE_CITATION_RE.findall(text):
        url = url.strip()
        if url not in seen:
            seen.add(url)
            sources.append(url)
    return sources


# ── LLM call ───────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=5, max=60),
    retry=retry_if_exception_type((RateLimitError, APIStatusError)),
    reraise=True,
)
def _call_groq(messages: list[dict], stream: bool = True) -> tuple[str, int, int]:
    """
    Call the Groq chat completion API.

    Args:
        messages: list of {role, content} dicts from prompt_builder
        stream:   if True, streams tokens to console while collecting

    Returns:
        (full_response_text, input_tokens, output_tokens)
    """
    if stream:
        response_text = ""
        input_tokens  = 0
        output_tokens = 0

        with _client.chat.completions.create(
            model=GROQ_CHAT_MODEL,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.2,        # low temp = more factual, less creative
            stream=True,
        ) as stream_resp:
            for chunk in stream_resp:
                delta = chunk.choices[0].delta
                if delta.content:
                    response_text += delta.content
                    # Print token-by-token to console
                    console.print(delta.content, end="", highlight=False)

                # Capture usage from the final chunk
                if chunk.usage:
                    input_tokens  = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens

        console.print()  # newline after streamed output
        return response_text, input_tokens, output_tokens

    else:
        # Non-streaming path (used in eval mode)
        resp = _client.chat.completions.create(
            model=GROQ_CHAT_MODEL,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.2,
        )
        text          = resp.choices[0].message.content
        input_tokens  = resp.usage.prompt_tokens
        output_tokens = resp.usage.completion_tokens
        return text, input_tokens, output_tokens


# ── Main generate ──────────────────────────────────────────────

def generate(
    messages: list[dict],
    stream: bool = True,
) -> dict:
    """
    Generate an explanation from a packed prompt.

    Args:
        messages: output of prompt_builder.build_messages()
        stream:   stream tokens to console while generating (default True)
                  set False in eval/batch mode for cleaner output

    Returns:
        {
            "explanation":   str,
            "sources":       list[str],
            "input_tokens":  int,
            "output_tokens": int,
            "model":         str,
        }
    """
    if not messages:
        return {
            "explanation":   "",
            "sources":       [],
            "input_tokens":  0,
            "output_tokens": 0,
            "model":         GROQ_CHAT_MODEL,
        }

    try:
        explanation, input_tokens, output_tokens = _call_groq(messages, stream=stream)
    except RateLimitError as e:
        console.print(f"\n[red]Groq rate limit hit:[/red] {e}")
        raise
    except Exception as e:
        console.print(f"\n[red]Generation failed:[/red] {e}")
        raise

    sources = extract_sources(explanation)

    return {
        "explanation":   explanation,
        "sources":       sources,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "model":         GROQ_CHAT_MODEL,
    }


# ── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test generation with a dummy prompt — no Pinecone needed
    from src.prompt_builder import build_messages, count_prompt_tokens

    dummy_chunks = [
        {
            "chunk_id": "asyncio-task_001",
            "text": (
                "asyncio.gather(*aws, return_exceptions=False)\n\n"
                "Run awaitable objects in the aws sequence concurrently. "
                "If any awaitable in aws is a coroutine, it is automatically "
                "scheduled as a Task. If all awaitables are completed "
                "successfully, the result is an aggregate list of returned values.\n\n"
                "If return_exceptions is False (default), the first raised "
                "exception is immediately propagated to the task that awaits "
                "on gather(). Other awaitables in the aws sequence won't be "
                "cancelled and will continue to run."
            ),
            "url":      "https://docs.python.org/3/library/asyncio-task.html",
            "title":    "Coroutines and Tasks",
            "has_code": True,
            "score":    0.91,
        },
    ]

    dummy_code = """\
async def main():
    results = await asyncio.gather(
        fetch("https://example.com"),
        fetch("https://example.org"),
        return_exceptions=True,
    )
    return results
"""
    dummy_question = "What does return_exceptions=True do in asyncio.gather?"

    console.print("\n[bold]Generator test[/bold]\n")

    messages = build_messages(
        chunks=dummy_chunks,
        question=dummy_question,
        code_snippet=dummy_code,
    )

    console.print(f"[dim]Prompt tokens : {count_prompt_tokens(messages)}[/dim]")
    console.print(f"[dim]Model         : {GROQ_CHAT_MODEL}[/dim]")
    console.print("\n[bold]Response:[/bold]\n")

    result = generate(messages, stream=True)

    console.print("\n[bold]Result dict:[/bold]")
    console.print(f"  sources       : {result['sources']}")
    console.print(f"  input_tokens  : {result['input_tokens']}")
    console.print(f"  output_tokens : {result['output_tokens']}")
    console.print(f"  model         : {result['model']}\n")