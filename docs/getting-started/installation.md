# Installation

## Prerequisites

- Python 3.12
- `uv`
- Git
- A shell such as PowerShell on Windows

The project uses `uv` for dependency and environment management.

## Install Dependencies

For full local development, including notebooks and documentation, run:

```powershell
uv sync --all-extras --dev --group notebook --group docs
```

This installs application extras, development tools, notebook tooling, and documentation tooling. Day-to-day documentation commands are listed in the [development workflow](development.md).

## Windows Hardlink Warning

On Windows, `uv` may report a hardlink fallback warning. To avoid that warning for the current PowerShell session:

```powershell
$env:UV_LINK_MODE = "copy"
```

To persist it for your Windows user account:

```powershell
[Environment]::SetEnvironmentVariable("UV_LINK_MODE", "copy", "User")
```

Open a new PowerShell terminal after setting it permanently.

## Notebook Tooling

Notebook dependencies include JupyterLab, ipykernel, ipywidgets, matplotlib, tqdm, nbstripout, and Jupytext.

Install the notebook output stripping filter once per clone:

```powershell
uv run nbstripout --install
```

Notebook code should be treated as exploratory or workflow-oriented. Stable reusable logic should move into `src/swingtrader`.

