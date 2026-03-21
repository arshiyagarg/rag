"""
prompt_builder.py — pack retrieved chunks into an LLM-ready prompt

Builds a DSA hint prompt — guides the user toward the solution
without revealing it. Cites SO answers and CP-Algorithms articles.

Single public function:
    build_messages(chunks, problem, code_snippet) -> list[dict]

Run directly to preview a prompt:
    python -m src.prompt_builder
"""

import tiktoken
from rich.console import Console

from src.config import MAX_CONTEXT_TOKENS

console = Console()

_enc = tiktoken.get_encoding("cl100k_base")

def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


# ── System prompts ────────────────────────────────────────────
# Two modes: hint (no code) and code-fix (corrected code on request)

HINT_PROMPT_HEADER = """You are an expert DSA tutor. Your job is to give HINTS only — never write code, never reveal the full solution.

Rules:
- Identify the algorithm pattern (Sliding Window, BFS, DP, Two Pointers, etc.)
- Explain WHY that pattern applies to this specific problem
- Point out conceptually what is wrong in the student's code — NO code whatsoever
- Do NOT write any code blocks, snippets, or corrected versions
- Base every explanation strictly on the <source> blocks provided below
- Cite sources inline: [SOURCE: url]
- If not covered in sources say exactly: "This pattern is not in the provided references."

Structure your response EXACTLY as:
1. Pattern identified: <name the pattern>
2. Why this pattern: <1-2 sentences with citation>
3. What's wrong in your code: <specific conceptual issue, absolutely no code>
4. Key concept to study: <with citation>
"""

CODE_FIX_PROMPT_HEADER = """You are an expert DSA tutor. The student is asking for corrected code.

CRITICAL LANGUAGE RULE: You MUST write the corrected code in the EXACT SAME
programming language as the student's code. If the student wrote C++, you write
C++. If Python, write Python. NEVER translate to a different language.

Rules:
- Read the student's code carefully and identify the SPECIFIC lines that are wrong
- Make MINIMAL changes — only fix the bug, do not rewrite the entire solution
- The corrected code must be in the SAME language as the student's input
- Do NOT translate C++ to Python, Python to Java, etc. under any circumstance
- Ground the fix in the <source> blocks below — do not invent logic
- If the sources don't cover this specific bug, say so explicitly
- Cite sources inline: [SOURCE: url]

Structure your response EXACTLY as:
1. Pattern identified: <name the pattern>
2. What was wrong: <the specific line(s) and why they are wrong>
3. Corrected code (same language as student):
```
<minimal corrected code in student's original language>
```
4. Why this fixes it: <explanation with citation>
"""

SYSTEM_PROMPT_FOOTER = """
Use only the sources above. Cite them with [SOURCE: url].
"""

OVERHEAD_TOKENS = 400


# ── Source block formatting ────────────────────────────────────

def _format_source_block(chunk: dict, index: int) -> str:
    """Format a chunk as an XML source block with rich metadata."""
    src_type  = chunk.get("source_type", "unknown")
    so_score  = chunk.get("so_score")
    is_acc    = chunk.get("is_accepted")
    topic     = chunk.get("topic", "")

    # Build attribute string based on source type
    if src_type == "stackoverflow":
        extra = f'score="{so_score}" accepted="{is_acc}"'
    elif src_type == "cp_algorithms":
        extra = f'topic="{topic}"'
    else:
        extra = ""

    return (
        f'<source index="{index}" type="{src_type}" '
        f'url="{chunk["url"]}" title="{chunk["title"]}" {extra}>\n'
        f'{chunk["text"].strip()}\n'
        f'</source>'
    )


def _pack_sources(chunks: list[dict], token_budget: int) -> tuple[str, int]:
    """Pack chunks into token budget, highest score first."""
    blocks: list[str] = []
    tokens_used = 0

    for i, chunk in enumerate(chunks, start=1):
        block        = _format_source_block(chunk, i)
        block_tokens = _count_tokens(block)
        if tokens_used + block_tokens > token_budget:
            break
        blocks.append(block)
        tokens_used += block_tokens

    return "\n\n".join(blocks), len(blocks)


# ── User turn ──────────────────────────────────────────────────

def _detect_language(code: str) -> str:
    """Heuristic language detection from code snippet."""
    code_lower = code.lower()
    if any(kw in code for kw in ["#include", "vector<", "int main", "::", "cout", "->", "auto "]):
        return "C++"
    if any(kw in code for kw in ["def ", "import ", "print(", "elif ", "None", "True", "False"]):
        return "Python"
    if any(kw in code for kw in ["public class", "System.out", "void ", "ArrayList", "import java"]):
        return "Java"
    if any(kw in code for kw in ["function ", "const ", "let ", "var ", "console.log", "=>"]):
        return "JavaScript"
    return "the same language as above"


def _format_user_turn(problem: str, code_snippet: str | None) -> str:
    """Build the user message with problem description and optional code."""
    parts: list[str] = [f"Problem:\n{problem.strip()}"]
    if code_snippet and code_snippet.strip():
        lang = _detect_language(code_snippet)
        parts.append(
            f"My current code (written in {lang} — fix must also be in {lang}):\n\n"
            f"```\n{code_snippet.strip()}\n```"
        )
    return "\n\n".join(parts)


# ── Main builder ───────────────────────────────────────────────

def build_messages(
    chunks:       list[dict],
    problem:      str,
    code_snippet: str | None = None,
    mode:         str        = "hint",
) -> list[dict]:
    """
    Build messages list for Groq chat completion.

    Args:
        chunks:       reranked chunks from retriever.py
        problem:      DSA problem description
        code_snippet: student's current code — optional
        mode:         "hint"     → no code in response, hints only
                      "code_fix" → corrected code grounded in sources

    Returns:
        [{"role": "system", ...}, {"role": "user", ...}]
    """
    header = CODE_FIX_PROMPT_HEADER if mode == "code_fix" else HINT_PROMPT_HEADER

    if not chunks:
        system_content = (
            header.strip()
            + "\n\n<source>No relevant references found.</source>\n"
            + SYSTEM_PROMPT_FOOTER.strip()
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": _format_user_turn(problem, code_snippet)},
        ]

    header_tokens    = _count_tokens(header + SYSTEM_PROMPT_FOOTER)
    user_tokens      = _count_tokens(_format_user_turn(problem, code_snippet))
    available_budget = MAX_CONTEXT_TOKENS - header_tokens - user_tokens - OVERHEAD_TOKENS
    available_budget = max(available_budget, MAX_CONTEXT_TOKENS // 2)

    sources_text, n_included = _pack_sources(chunks, available_budget)

    if n_included < len(chunks):
        console.print(
            f"[dim]Context cap: {n_included}/{len(chunks)} chunks included[/dim]"
        )

    system_content = header + sources_text + SYSTEM_PROMPT_FOOTER
    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": _format_user_turn(problem, code_snippet)},
    ]


def count_prompt_tokens(messages: list[dict]) -> int:
    return sum(_count_tokens(m["content"]) for m in messages)


# ── CLI preview ────────────────────────────────────────────────

if __name__ == "__main__":
    dummy_chunks = [
        {
            "chunk_id":    "so_12345_accepted_00",
            "text": (
                "For finding the longest substring without repeating characters, "
                "the sliding window technique is the most efficient approach. "
                "Maintain a window with two pointers and a set tracking current chars. "
                "When a duplicate is found, shrink the window from the left."
            ),
            "url":         "https://stackoverflow.com/a/12345",
            "title":       "Longest substring without repeating characters",
            "source_type": "stackoverflow",
            "so_score":    312,
            "is_accepted": True,
            "topic":       None,
        },
        {
            "chunk_id":    "cp_string_hashing_001",
            "text": (
                "## Two Pointers / Sliding Window\n\n"
                "The sliding window technique uses two pointers to represent "
                "the current window boundaries. The key insight is that we never "
                "need to re-examine characters that were already processed."
            ),
            "url":         "https://cp-algorithms.com/string/hashing.html",
            "title":       "String Hashing",
            "source_type": "cp_algorithms",
            "so_score":    None,
            "is_accepted": None,
            "topic":       "string",
        },
    ]

    dummy_problem = (
        "Given a string s, find the length of the longest substring "
        "without repeating characters."
    )
    dummy_code = """\
def length_of_longest_substring(s):
    result = 0
    for i in range(len(s)):
        for j in range(i, len(s)):
            if len(set(s[i:j])) == j - i:
                result = max(result, j - i)
    return result
"""

    messages = build_messages(
        chunks=dummy_chunks,
        problem=dummy_problem,
        code_snippet=dummy_code,
    )

    console.print("\n[bold]Prompt preview[/bold]\n")
    for msg in messages:
        console.rule(f"[dim]{msg['role']}[/dim]")
        console.print(msg["content"])

    console.print()
    console.rule()
    tokens = count_prompt_tokens(messages)
    console.print(f"[dim]Total tokens: {tokens} / {MAX_CONTEXT_TOKENS}[/dim]\n")