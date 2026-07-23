# Core

Shared infrastructure and contracts used across the project. This package should stay lightweight and contain only stable code that can be safely imported by the data, modeling, and API packages, such as configuration, database connection helpers, shared schemas, domain-neutral numerical helpers, logging setup, and symbol metadata.

See also:

- [Architecture overview](../../../docs/architecture/overview.md)
- [Operations troubleshooting](../../../docs/operations/troubleshooting.md)

## Logging

Runnable entrypoints such as jobs, scripts, and notebooks should configure logging once before calling application code:

```python
from swingtrader.core.logging_config import configure_logging

configure_logging()
```

Library modules should not configure logging at import time. They should use the standard Python named logger pattern and let the entrypoint decide handlers, level, and output format:

```python
import logging

logger = logging.getLogger(__name__)
```
