from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from swingtrader.data.features.catalog import DEFAULT_FEATURE_SET
from swingtrader.modeling.datasets.catalog import (
    FORWARD_RETURN_PRIMARY_TASK,
    FORWARD_RETURN_TARGET_SET,
)
from swingtrader.modeling.experiments import (
    DatasetSplitSummary,
    DatasetSummary,
    ExperimentSpec,
    ModelSpec,
    TemporalSplitSpec,
    UniverseSpec,
    start_experiment_run,
    tracking,
)


def _experiment_spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="baseline",
        version="1",
        feature_set=DEFAULT_FEATURE_SET,
        target_set=FORWARD_RETURN_TARGET_SET,
        task=FORWARD_RETURN_PRIMARY_TASK,
        universe=UniverseSpec(
            name="se_large_mid_cap",
            version="2026-07-24",
            provider="yfinance",
            tickers=("ABB.ST", "VOLV-B.ST"),
        ),
        data_start=date(2010, 1, 1),
        data_end=date(2025, 12, 31),
        split=TemporalSplitSpec(
            name="holdout",
            version="1",
            train_start=date(2010, 1, 1),
            train_end=date(2021, 12, 31),
            validation_start=date(2022, 1, 1),
            validation_end=date(2023, 12, 31),
            test_start=date(2024, 1, 1),
            test_end=date(2025, 12, 31),
        ),
        model=ModelSpec(
            name="logistic_regression",
            version="1",
            model_type="sklearn.linear_model.LogisticRegression",
            hyperparameters={"C": 1.0},
        ),
        random_seeds={"model": 17},
    )


def _dataset_summary() -> DatasetSummary:
    return DatasetSummary(
        train=DatasetSplitSummary(
            rows=1_000,
            ticker_count=2,
            start_date=date(2010, 1, 4),
            end_date=date(2021, 12, 30),
            class_prevalence=0.18,
        ),
        validation=DatasetSplitSummary(
            rows=200,
            ticker_count=2,
            start_date=date(2022, 1, 3),
            end_date=date(2023, 12, 29),
            class_prevalence=0.17,
        ),
        test=DatasetSplitSummary(
            rows=250,
            ticker_count=2,
            start_date=date(2024, 1, 2),
            end_date=date(2025, 12, 30),
            class_prevalence=0.16,
        ),
    )


class FakeMlflow:
    def __init__(self) -> None:
        self.tracking_uri: str | None = None
        self.experiment_name: str | None = None
        self.params: dict[str, object] = {}
        self.metrics: list[tuple[dict[str, float], int | None]] = []
        self.tags: dict[str, str] = {}
        self.artifacts: list[tuple[str, str | None, str]] = []
        self.run_name: str | None = None

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment_name = name

    @contextmanager
    def start_run(self, *, run_name: str):
        self.run_name = run_name
        yield SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

    def log_params(self, params: dict[str, object]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metrics.append((dict(metrics), step))

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.update(tags)

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        artifact = Path(path)
        self.artifacts.append((artifact.name, artifact_path, artifact.read_text()))


def test_start_experiment_run_logs_configuration_summary_metrics_and_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setattr(tracking, "_import_mlflow", lambda: fake_mlflow)
    monkeypatch.setattr(tracking, "resolve_git_revision", lambda repository_root=None: "abc123")
    report = tmp_path / "report.txt"
    report.write_text("evaluation report", encoding="utf-8")

    with start_experiment_run(
        _experiment_spec(),
        experiment_name="tests",
        tracking_uri=tmp_path.joinpath("mlruns").as_uri(),
        dataset_summary=_dataset_summary(),
        tags={"purpose": "unit-test"},
    ) as run:
        assert run.run_id == "run-123"
        run.log_metrics({"validation.roc_auc": 0.71}, step=1)
        run.log_artifact(report, artifact_path="reports")

    assert fake_mlflow.experiment_name == "tests"
    assert fake_mlflow.params["git.commit"] == "abc123"
    assert fake_mlflow.params["task.target_column"] == "target_significant_up_5d"
    assert fake_mlflow.params["dataset.train.rows"] == 1_000
    assert fake_mlflow.params["dataset.test.ticker_count"] == 2
    prevalence_metrics = {
        "dataset.train.class_prevalence": 0.18,
        "dataset.validation.class_prevalence": 0.17,
        "dataset.test.class_prevalence": 0.16,
    }
    assert (prevalence_metrics, None) in fake_mlflow.metrics
    assert ({"validation.roc_auc": 0.71}, 1) in fake_mlflow.metrics
    assert fake_mlflow.tags["mlflow.source.git.commit"] == "abc123"
    assert fake_mlflow.tags["purpose"] == "unit-test"

    manifest_artifact = next(
        artifact for artifact in fake_mlflow.artifacts if artifact[0] == "experiment.json"
    )
    assert manifest_artifact[1] == "manifests"
    assert json.loads(manifest_artifact[2])["identifier"] == "baseline:1"
    assert ("report.txt", "reports", "evaluation report") in fake_mlflow.artifacts


def test_start_experiment_run_rejects_summary_outside_declared_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = FakeMlflow()
    valid = _dataset_summary()
    monkeypatch.setattr(tracking, "_import_mlflow", lambda: fake_mlflow)
    invalid = DatasetSummary(
        train=DatasetSplitSummary(
            rows=10,
            ticker_count=1,
            start_date=date(2009, 12, 31),
            end_date=date(2021, 12, 30),
        ),
        validation=valid.validation,
        test=valid.test,
    )

    with (
        pytest.raises(ValueError, match="train date range"),
        start_experiment_run(_experiment_spec(), dataset_summary=invalid),
    ):
        pass

    assert fake_mlflow.tracking_uri is None


def test_start_experiment_run_rejects_tickers_outside_universe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = FakeMlflow()
    valid = _dataset_summary()
    monkeypatch.setattr(tracking, "_import_mlflow", lambda: fake_mlflow)
    invalid = DatasetSummary(
        train=DatasetSplitSummary(
            rows=10,
            ticker_count=3,
            start_date=date(2010, 1, 4),
            end_date=date(2021, 12, 30),
        ),
        validation=valid.validation,
        test=valid.test,
    )

    with (
        pytest.raises(ValueError, match="universe size"),
        start_experiment_run(_experiment_spec(), dataset_summary=invalid),
    ):
        pass

    assert fake_mlflow.tracking_uri is None


def test_tracking_module_does_not_import_mlflow_until_a_run_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_mlflow(name: str) -> Any:
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(tracking.importlib, "import_module", missing_mlflow)

    with pytest.raises(ModuleNotFoundError, match="modeling extra"):
        tracking._import_mlflow()


def test_mlflow_transitive_import_errors_are_not_rewritten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dependency(name: str) -> Any:
        raise ModuleNotFoundError(name="mlflow_dependency")

    monkeypatch.setattr(tracking.importlib, "import_module", missing_dependency)

    with pytest.raises(ModuleNotFoundError) as error:
        tracking._import_mlflow()

    assert error.value.name == "mlflow_dependency"


@pytest.mark.parametrize(
    ("rows", "ticker_count", "start_date", "end_date", "message"),
    [
        (0, 1, None, None, "ticker count"),
        (1, 1, None, None, "date range"),
        (1, 1, date(2024, 1, 1), None, "provided together"),
    ],
)
def test_dataset_split_summary_rejects_inconsistent_metadata(
    rows: int,
    ticker_count: int,
    start_date: date | None,
    end_date: date | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DatasetSplitSummary(
            rows=rows,
            ticker_count=ticker_count,
            start_date=start_date,
            end_date=end_date,
        )


def test_experiment_run_rejects_boolean_metrics() -> None:
    fake_mlflow = FakeMlflow()
    run = tracking.ExperimentRun(fake_mlflow, "run-123")

    with pytest.raises(TypeError, match="real number"):
        run.log_metrics({"validation.flag": True})  # type: ignore[dict-item]


def test_dataset_split_summary_validates_prevalence() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        DatasetSplitSummary(
            rows=10,
            ticker_count=1,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            class_prevalence=1.1,
        )


def test_local_mlflow_run_can_be_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)
    tracking_uri = tracking.local_tracking_uri(tmp_path / "mlflow.db")

    with start_experiment_run(
        _experiment_spec(),
        experiment_name="integration-test",
        tracking_uri=tracking_uri,
        dataset_summary=_dataset_summary(),
    ) as run:
        run_id = run.run_id
        run.log_metrics({"validation.roc_auc": 0.71})

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    tracked_run = client.get_run(run_id)
    artifacts = client.list_artifacts(run_id, "manifests")

    assert tracked_run.data.params["experiment.identifier"] == "baseline:1"
    assert tracked_run.data.metrics["validation.roc_auc"] == pytest.approx(0.71)
    assert [artifact.path for artifact in artifacts] == ["manifests/experiment.json"]


def test_local_tracking_uri_uses_an_absolute_sqlite_database(tmp_path: Path) -> None:
    assert tracking.local_tracking_uri(tmp_path / "runs.db") == (
        f"sqlite:///{tmp_path.joinpath('runs.db').resolve().as_posix()}"
    )


@pytest.mark.parametrize(("rows", "ticker_count"), [(True, 1), (1, 1.5)])
def test_dataset_split_summary_rejects_non_integer_counts(
    rows: object,
    ticker_count: object,
) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        DatasetSplitSummary(
            rows=rows,  # type: ignore[arg-type]
            ticker_count=ticker_count,  # type: ignore[arg-type]
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )


def test_experiment_run_validates_metric_step() -> None:
    run = tracking.ExperimentRun(FakeMlflow(), "run-123")

    with pytest.raises(ValueError, match="must not be negative"):
        run.log_metrics({"validation.roc_auc": 0.71}, step=-1)


def test_start_experiment_run_rejects_blank_run_name() -> None:
    with (
        pytest.raises(ValueError, match="run name"),
        start_experiment_run(_experiment_spec(), run_name="   "),
    ):
        pass


def test_dataset_split_summary_rejects_non_numeric_prevalence() -> None:
    with pytest.raises(TypeError, match="real number"):
        DatasetSplitSummary(
            rows=10,
            ticker_count=1,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            class_prevalence="0.5",  # type: ignore[arg-type]
        )


def test_dataset_summary_requires_split_summary_instances() -> None:
    split = DatasetSplitSummary(
        rows=10,
        ticker_count=1,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
    )

    with pytest.raises(TypeError, match="validation summary"):
        DatasetSummary(
            train=split,
            validation={"rows": 10},  # type: ignore[arg-type]
            test=split,
        )


def test_experiment_run_rejects_numeric_strings() -> None:
    run = tracking.ExperimentRun(FakeMlflow(), "run-123")

    with pytest.raises(TypeError, match="real number"):
        run.log_metrics({"validation.roc_auc": "0.71"})  # type: ignore[dict-item]


def test_reserved_tracking_tags_use_authoritative_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = FakeMlflow()
    spec = _experiment_spec()
    monkeypatch.setattr(tracking, "_import_mlflow", lambda: fake_mlflow)
    monkeypatch.setattr(tracking, "resolve_git_revision", lambda repository_root=None: "abc123")

    with start_experiment_run(
        spec,
        tracking_uri="file:///tmp/mlruns",
        tags={
            "experiment.digest": "incorrect",
            "mlflow.source.git.commit": "incorrect",
        },
    ):
        pass

    assert fake_mlflow.tags["experiment.digest"] == spec.digest
    assert fake_mlflow.tags["mlflow.source.git.commit"] == "abc123"


def test_empty_dataset_split_rejects_class_prevalence() -> None:
    with pytest.raises(ValueError, match="prevalence"):
        DatasetSplitSummary(
            rows=0,
            ticker_count=0,
            start_date=None,
            end_date=None,
            class_prevalence=0.0,
        )


def test_non_empty_dataset_split_requires_a_ticker() -> None:
    with pytest.raises(ValueError, match="at least one ticker"):
        DatasetSplitSummary(
            rows=1,
            ticker_count=0,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        )
