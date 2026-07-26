# Preview Documentation

Use this when editing documentation pages.

## Serve Locally

```powershell
uv run --group docs mkdocs serve
```

Open the local URL printed by MkDocs, usually `http://127.0.0.1:8000/`.
This is the recommended workflow while editing documentation.

## Build The Static Site

```powershell
uv run --group docs mkdocs build --strict
```

This writes the static site to `site/` and fails if MkDocs reports warnings or errors.

## Open The Built Site Directly

Open `site/index.html` in a browser. The documentation build enables Material for
MkDocs offline support, and the generated links use explicit `.html` files so the
static site, including Mermaid diagrams, can be browsed directly from disk.