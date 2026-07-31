# Modeling Targets

A target contract defines future-dependent outcomes for supervised learning. `TargetFamilySpec` passes its immutable parameter mapping to the builder, validates required inputs and index preservation, rejects output collisions, and selects only its declared target columns. `TargetSetSpec` composes families in declaration order and returns exactly the versioned target schema alongside the source columns. Every keyword-compatible builder parameter must be recorded explicitly, even when the builder defines a Python default, so manifests describe the exact execution. Target builders intentionally use future observations and expose their maximum horizon; they remain separate from feature builders, which describe information available at prediction time.

| Version | Primary task | Primary output | Resolution semantics | Description |
| --- | --- | --- | --- | --- |
| [V1](v1-forward-return.md) | Binary classification | `target_significant_up_5d` | Fixed five-session horizon | Whether adjusted close exceeds an economic return threshold after five observed sessions |
| [V3](v3-triple-barrier.md) | Three-class classification | `triple_barrier_label_5d` | First barrier event or five-session timeout | Whether take-profit, timeout, or stop-loss occurs first after next-open entry |

Each target page begins with the same high-level taxonomy and then documents its formulas, temporal semantics, outputs, assumptions, and limitations. Shared model-reporting principles are documented under [Model Evaluation](../evaluation.md).

V3 supersedes the experimental V2 ATR-barrier schema. Its direct `1`, `0`, and `-1` label removes parallel categorical and Boolean outcome columns.
