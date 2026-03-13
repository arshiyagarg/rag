import sys
from rich.console import Console

console = Console()


def check(label: str, fn):
    try:
        result = fn()
        console.print(f"  [green]OK[/green]  {label}" + (f" — {result}" if result else ""))
        return True
    except Exception as e:
        console.print(f"  [red]FAIL[/red] {label} — {e}")
        return False


def main():
    console.print("\n[bold]Code Explainer v0 — Day 1 Smoke Test[/bold]\n")
    results = []

    # ── Config & env ───────────────────────────────────────────
    console.print("[dim]Config & environment[/dim]")
    results.append(check("python-dotenv loads .env", lambda: (
        __import__("dotenv").load_dotenv() or "loaded"
    )))
    results.append(check("GEMINI_API_KEY present", lambda: (
        __import__("os").environ["GEMINI_API_KEY"][:8] + "..."
    )))
    results.append(check("GROQ_API_KEY present", lambda: (
        __import__("os").environ["GROQ_API_KEY"][:8] + "..."
    )))
    results.append(check("PINECONE_API_KEY present", lambda: (
        __import__("os").environ["PINECONE_API_KEY"][:8] + "..."
    )))

    # ── Library imports ────────────────────────────────────────
    console.print("\n[dim]Library imports[/dim]")
    for lib, import_str in [
        ("google-generativeai", "import google.generativeai as genai"),
        ("groq",                "from groq import Groq"),
        ("pinecone",            "from pinecone import Pinecone, ServerlessSpec"),
        ("chonkie",             "from chonkie import SemanticChunker"),
        ("requests",            "import requests"),
        ("beautifulsoup4",      "from bs4 import BeautifulSoup"),
        ("trafilatura",         "import trafilatura"),
        ("tiktoken",            "import tiktoken"),
        ("tenacity",            "import tenacity"),
        ("ragas",               "import ragas"),
        ("tqdm",                "from tqdm import tqdm"),
    ]:
        results.append(check(lib, lambda s=import_str: exec(s) or "imported"))

    # ── Live API checks ────────────────────────────────────────
    console.print("\n[dim]Live API checks[/dim]")

    def check_gemini_embed():
        import os
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        result = genai.embed_content(
            model="models/gemini-embedding-2-preview",
            content="hello world",
        )
        dim = len(result["embedding"])
        return f"dim={dim}, model=gemini-embedding-2-preview"

    def check_groq():
        import os
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=10,
        )
        return f"model={resp.model}, tokens={resp.usage.prompt_tokens}"

    def check_pinecone():
        import os
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        indexes = pc.list_indexes()
        return f"connected, indexes={len(indexes)}"

    results.append(check("Gemini embeddings (gemini-embedding-2-preview)", check_gemini_embed))
    results.append(check("Groq chat (llama-3.3-70b-versatile)", check_groq))
    results.append(check("Pinecone connection", check_pinecone))

    # ── Summary ────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    color = "green" if passed == total else "red"
    console.print(
        f"\n[bold {color}]"
        f"{'ALL CHECKS PASSED' if passed == total else 'SOME CHECKS FAILED'}"
        f" ({passed}/{total})[/bold {color}]\n"
    )
    if passed < total:
        console.print("[dim]Fix failing checks before starting Day 2.[/dim]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
    