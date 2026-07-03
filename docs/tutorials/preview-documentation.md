# Preview Documentation

Use this when editing documentation pages.

## Serve Locally

```powershell
uv run --group docs mkdocs serve
```

Open the local URL printed by MkDocs, usually `http://127.0.0.1:8000/`.

## Build The Static Site

```powershell
uv run --group docs mkdocs build --strict
```

This writes the static site to `site/` and fails if MkDocs reports warnings or errors.

## Open The Built Site Directly

Open `site/index.html` in a browser. The generated links use explicit `.html` files so the static site can be browsed directly from disk.