"""
generator.py — call Groq LLM and parse the response

Three public functions:
    generate(messages, stream)         → dict        CLI + eval
    stream_generator(messages)         → Generator   Streamlit st.write_stream()
    extract_sources(text)              → list[str]   parse [SOURCE: url] citations

generate() returns:
{
    "explanation":  "...",
    "sources":      ["url1", ...],
    "input_tokens":  412,
    "output_tokens": 284,
    "model":        "llama-3.3-70b-versatile"
}

stream_generator() yields raw token strings — compatible with st.write_stream().
After the generator is exhausted, call get_stream_result() to get the full
result dict with sources and token counts.

Run directly to test:
    python -m src.generator
"""

import re
from typing import Generator
from groq import Groq, RateLimitError, APIStatusError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from rich.console import Console

from src.config import GROQ_API_KEY, GROQ_CHAT_MODEL, MAX_CONTEXT_TOKENS

console = Console()

# ── Client ─────────────────────────────────────────────────────
_client = Groq(api_key=GROQ_API_KEY)

MAX_OUTPUT_TOKENS  = 1024
# Handles [SOURCE: url] and [SOURCE: __url__] (LLM markdown artifacts)
SOURCE_CITATION_RE = re.compile(r"\[SOURCE:\s*[_*]*(https?://[^\]_*\s]+)[_*]*\]")


# ── Citation extraction ────────────────────────────────────────

def extract_sources(text: str) -> list[str]:
    """
    Extract unique cited URLs from LLM output.
    Matches [SOURCE: https://...] per the system prompt instruction.
    Preserves first-appearance order.
    """
    seen:    set[str]  = set()
    sources: list[str] = []
    for url in SOURCE_CITATION_RE.findall(text):
        url = url.strip()
        if url not in seen:
            seen.add(url)
            sources.append(url)
    return sources


# ── Core Groq call ─────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=5, max=60),
    retry=retry_if_exception_type((RateLimitError, APIStatusError)),
    reraise=True,
)
def _call_groq_blocking(messages: list[dict]) -> tuple[str, int, int]:
    """
    Non-streaming Groq call — used by generate() for CLI and eval.
    Returns (full_text, input_tokens, output_tokens).
    """
    resp = _client.chat.completions.create(
        model=GROQ_CHAT_MODEL,
        messages=messages,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.2,
        stream=False,
    )
    return (
        resp.choices[0].message.content,
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
    )


def _call_groq_streaming(messages: list[dict]):
    """
    Return a raw Groq streaming response object.
    Caller iterates over it to get chunks.
    Not decorated with @retry — streaming retries are handled by
    stream_generator() directly so partial output isn't lost.
    """
    return _client.chat.completions.create(
        model=GROQ_CHAT_MODEL,
        messages=messages,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.2,
        stream=True,
    )


# ── generate() — CLI + eval ────────────────────────────────────

def generate(
    messages: list[dict],
    stream:   bool = False,
) -> dict:
    """
    Generate a hint from a packed prompt.
    Default stream=False — cleaner for eval and CLI single-shot use.
    Set stream=True to print tokens to console as they arrive.

    Args:
        messages: output of prompt_builder.build_messages()
        stream:   print tokens to console while generating

    Returns:
        {explanation, sources, input_tokens, output_tokens, model}
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
        if stream:
            # Streaming path — collect tokens and print to console
            explanation   = ""
            input_tokens  = 0
            output_tokens = 0
            with _call_groq_streaming(messages) as stream_resp:
                for chunk in stream_resp:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        explanation += delta.content
                        console.print(delta.content, end="", highlight=False)
                    if chunk.usage:
                        input_tokens  = chunk.usage.prompt_tokens
                        output_tokens = chunk.usage.completion_tokens
            console.print()
        else:
            explanation, input_tokens, output_tokens = _call_groq_blocking(messages)

    except RateLimitError as e:
        console.print(f"\n[red]Groq rate limit:[/red] {e}")
        raise
    except Exception as e:
        console.print(f"\n[red]Generation failed:[/red] {e}")
        raise

    return {
        "explanation":   explanation,
        "sources":       extract_sources(explanation),
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "model":         GROQ_CHAT_MODEL,
    }


# ── stream_generator() — Streamlit ────────────────────────────

class StreamResult:
    """
    Holds the accumulated result after stream_generator() is exhausted.
    Access via: result = StreamResult(); ... yield from stream_generator(..., result)
    """
    def __init__(self):
        self.explanation:   str       = ""
        self.sources:       list[str] = []
        self.input_tokens:  int       = 0
        self.output_tokens: int       = 0
        self.model:         str       = GROQ_CHAT_MODEL

    def to_dict(self) -> dict:
        return {
            "explanation":   self.explanation,
            "sources":       self.sources,
            "input_tokens":  self.input_tokens,
            "output_tokens": self.output_tokens,
            "model":         self.model,
        }


def stream_generator(
    messages: list[dict],
    result:   StreamResult | None = None,
) -> Generator[str, None, None]:
    """
    Yield token strings from Groq — compatible with st.write_stream().

    Usage in Streamlit:
        result = StreamResult()
        st.write_stream(stream_generator(messages, result))
        # After stream is exhausted:
        sources = result.sources
        tokens  = result.input_tokens

    Args:
        messages: output of prompt_builder.build_messages()
        result:   optional StreamResult instance — populated with
                  full text, sources, and token counts after streaming.
                  If None, a local instance is used (sources/tokens lost).

    Yields:
        str — each token chunk from the LLM as it arrives
    """
    if not messages:
        return

    if result is None:
        result = StreamResult()

    accumulated = ""
    try:
        with _call_groq_streaming(messages) as stream_resp:
            for chunk in stream_resp:
                delta = chunk.choices[0].delta

                if delta.content:
                    accumulated           += delta.content
                    result.explanation    += delta.content
                    yield delta.content   # ← this is what st.write_stream() consumes

                # Token counts arrive on the final chunk
                if chunk.usage:
                    result.input_tokens  = chunk.usage.prompt_tokens
                    result.output_tokens = chunk.usage.completion_tokens

    except RateLimitError as e:
        error_msg = f"\n\n⚠️ Rate limit hit — please try again in a moment. ({e})"
        result.explanation += error_msg
        yield error_msg
        return
    except Exception as e:
        error_msg = f"\n\n⚠️ Generation error: {e}"
        result.explanation += error_msg
        yield error_msg
        return

    # Populate sources after full text is accumulated
    result.sources = extract_sources(result.explanation)


# ── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.prompt_builder import build_messages, count_prompt_tokens

    dummy_chunks = [
        {
            "chunk_id":    "so_12345_accepted_00",
            "text": (
                "For the two sum problem, use a hash map to store "
                "each number and its index. For each number, check if "
                "its complement (target - num) already exists in the map. "
                "This gives O(n) time complexity instead of O(n^2)."
            ),
            "url":         "https://stackoverflow.com/a/12345",
            "title":       "Two Sum - Hash Map approach",
            "source_type": "stackoverflow",
            "so_score":    412,
            "is_accepted": True,
            "topic":       None,
            "score":       0.91,
        },
    ]

    dummy_problem = (
        "Given an array of integers and a target, "
        "return indices of two numbers that add up to target."
    )
    dummy_code = """\
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
"""

    messages = build_messages(
        chunks=dummy_chunks,
        question=dummy_problem,
        code_snippet=dummy_code,
    )

    console.print(f"\n[bold]Generator test[/bold]")
    console.print(f"[dim]Prompt tokens : {count_prompt_tokens(messages)}[/dim]")
    console.print(f"[dim]Model         : {GROQ_CHAT_MODEL}[/dim]")

    # ── Test 1: generate() ────────────────────────────────────
    console.print("\n[bold]Test 1 — generate() non-streaming:[/bold]\n")
    result = generate(messages, stream=False)
    console.print(result["explanation"][:400] + "...")
    console.print(f"\n[dim]sources={result['sources']}[/dim]")
    console.print(f"[dim]tokens: in={result['input_tokens']} out={result['output_tokens']}[/dim]")

    # ── Test 2: stream_generator() ───────────────────────────
    console.print("\n[bold]Test 2 — stream_generator() (simulates st.write_stream):[/bold]\n")
    sr = StreamResult()
    for token in stream_generator(messages, sr):
        console.print(token, end="", highlight=False)
    console.print()
    console.print(f"\n[dim]sources={sr.sources}[/dim]")
    console.print(f"[dim]tokens: in={sr.input_tokens} out={sr.output_tokens}[/dim]\n")