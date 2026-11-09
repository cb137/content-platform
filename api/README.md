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
