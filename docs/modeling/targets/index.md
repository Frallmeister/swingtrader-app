# Modeling Targets

A target contract defines future-dependent outcomes for supervised learning. `TargetFamilySpec` passes its immutable parameter mapping to the builder, validates required inputs and index preservation, rejects output collisions, and selects only its declared target columns. `TargetSetSpec` composes families in declaration order and returns exactly the declared target schema alongside the source columns. Every keyword-compatible builder parameter must be recorded explicitly, even when the builder defines a Python default, so manifests describe the exact execution. Target builders intentionally use future observations and expose their maximum horizon; they remain separate from feature builders, which describe information available at prediction time.

The target sets below are independent target variants rather than successive versions of one target. Each defines a distinct learning objective, is generated on its own from canonical prices, and none universally supersedes the others.

| Target set | Primary task | Primary output | Resolution semantics | Description |
| --- | --- | --- | --- | --- |
| [Forward return](forward-return.md) | Binary classification | `target_significant_up_5d` | Fixed five-session horizon | Whether adjusted close exceeds an economic return threshold after five observed sessions |
| [Triple barrier](triple-barrier.md) | Three-class classification | `triple_barrier_label_5d` | First barrier event or five-session timeout | Whether take-profit, timeout, or stop-loss occurs first after next-open entry |
| [Cross-sectional return](cross-sectional.md) | Regression | `forward_return_5d_cross_sectional_percentile` | Fixed five-session horizon | Market-relative returns, same-date future-return percentiles, and five ordinal relevance grades |

Each target page begins with the same high-level taxonomy and then documents its formulas, temporal semantics, outputs, assumptions, and limitations. Shared model-reporting principles are documented under [Model Evaluation](../evaluation.md).

Each target set is generated independently from canonical prices. Choosing one variant does not imply generating another, and their outputs do not overlap.
