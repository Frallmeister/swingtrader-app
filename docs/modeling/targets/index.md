# Modeling Targets

A target contract defines future-dependent outcomes for supervised learning. Target builders intentionally use future observations, expose their maximum horizon, preserve the canonical market index, and record deterministic manifests. They must remain separate from feature builders, which describe information available at prediction time.

| Version | Primary task | Primary output | Resolution semantics | Description |
| --- | --- | --- | --- | --- |
| [V1](v1-forward-return.md) | Binary classification | `target_significant_up_5d` | Fixed five-session horizon | Whether adjusted close exceeds an economic return threshold after five observed sessions |
| [V2](v2-atr-barrier.md) | Binary classification | `target_tp_before_sl_5d` | First barrier event or five-session timeout | Whether a next-open ATR-scaled take-profit is reached before the stop-loss |

Each target page begins with the same high-level taxonomy and then documents its formulas, temporal semantics, outputs, assumptions, and limitations. Shared model-reporting principles are documented under [Model Evaluation](../evaluation.md).
