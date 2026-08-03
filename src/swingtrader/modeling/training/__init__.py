"""Baseline models and reusable temporal evaluation harness."""

from swingtrader.modeling.training.baselines import (
    CONSTANT_PRIOR_MODEL_TYPE,
    LOGISTIC_REGRESSION_MODEL_TYPE,
    RANDOM_RANKING_MODEL_TYPE,
    BaselineModelArtifact,
    ConstantPriorClassifier,
    DateMatchedRandomRanker,
    MedianStandardizer,
    RegularizedLogisticRegression,
    deterministic_random_scores,
    fit_baseline_model,
    select_model_features,
)
from swingtrader.modeling.training.contracts import (
    PREDICTION_COLUMNS,
    EvaluationConfig,
    build_prediction_frame,
    validate_prediction_frame,
)
from swingtrader.modeling.training.evaluation import EvaluationReport, evaluate_predictions
from swingtrader.modeling.training.harness import (
    TEMPORAL_CV_RESULT_COLUMNS,
    BaselineExperimentResult,
    run_baseline_cross_validation,
    run_baseline_experiment,
    write_baseline_artifacts,
)
from swingtrader.modeling.training.ranking import (
    evaluate_cross_sectional_scores,
    prepare_xgboost_ranking_data,
)
from swingtrader.modeling.training.reporting import write_evaluation_artifacts

__all__ = [
    "CONSTANT_PRIOR_MODEL_TYPE",
    "LOGISTIC_REGRESSION_MODEL_TYPE",
    "PREDICTION_COLUMNS",
    "RANDOM_RANKING_MODEL_TYPE",
    "TEMPORAL_CV_RESULT_COLUMNS",
    "BaselineExperimentResult",
    "BaselineModelArtifact",
    "ConstantPriorClassifier",
    "DateMatchedRandomRanker",
    "EvaluationConfig",
    "EvaluationReport",
    "MedianStandardizer",
    "RegularizedLogisticRegression",
    "build_prediction_frame",
    "deterministic_random_scores",
    "evaluate_cross_sectional_scores",
    "evaluate_predictions",
    "fit_baseline_model",
    "prepare_xgboost_ranking_data",
    "select_model_features",
    "run_baseline_cross_validation",
    "run_baseline_experiment",
    "validate_prediction_frame",
    "write_baseline_artifacts",
    "write_evaluation_artifacts",
]
