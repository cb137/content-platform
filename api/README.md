# content-api

Tools for analysing LinkedIn posts.

Run every command from inside the `api/` folder.

## Install

```powershell
uv sync
```

## Test

```powershell
uv run pytest
uv run ruff check .
uv run mypy src
```

## Run

`post-stats` prints word, hook, character and hashtag counts for each post in a CSV
(columns: `id,niche,text,posted_at,likes,comments`).

```powershell
uv run post-stats                          # uses data/sample_posts.csv
uv run post-stats path\to\other_posts.csv
```

Example output:

```
id=1   words=27   hook_words=8   chars=141   hashtags=3
```

## AI layer

### Setup

Copy `.env.example` to `.env` (in `api/`) and fill in your key. `.env` is ignored by git.

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Needed for `LLM_PROVIDER=claude` |
| `OPENAI_API_KEY` | none | Needed for `LLM_PROVIDER=openai` |
| `LLM_PROVIDER` | `claude` | `claude` or `openai` |
| `CLAUDE_MODEL` | `claude-sonnet-5-5` | `claude-haiku-5-5` is about 20x cheaper |
| `OPENAI_MODEL` | `gpt-6.1-sol` | |

Prices live in `src/content_api/pricing.py`. Every call is appended to
`llm_calls.jsonl` (ignored by git).

### Commands

These **call the real AI API and cost money**:

```powershell
# The integration test (one real draft, a few cents). Normal runs skip it.
uv run pytest -m integration

# 10 drafts (the default) for a topic and niche; prints each hook and its cost
uv run python scripts/generate_drafts.py "rate limits" ai-engineering
uv run python scripts/generate_drafts.py "rate limits" ai-engineering --count 3
```

This one is free (it only reads `llm_calls.jsonl`):

```powershell
uv run python scripts/cost_report.py
```
