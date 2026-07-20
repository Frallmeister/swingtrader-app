import pytest

from swingtrader.indicators._validation import (
    validate_fast_slow_signal_lengths,
    validate_length,
    validate_multiplier,
    validate_num_std,
)


def test_validate_length_accepts_positive_integers() -> None:
    validate_length(1)
    validate_length(14)


@pytest.mark.parametrize("length", [0, -1, True, 1.5, "2"])
def test_validate_length_rejects_invalid_values(length: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        validate_length(length)  # type: ignore[arg-type]


def test_validate_multiplier_accepts_positive_numbers() -> None:
    validate_multiplier(1)
    validate_multiplier(2.5)


@pytest.mark.parametrize("multiplier", [0, -1, True, "2"])
def test_validate_multiplier_rejects_invalid_values(multiplier: object) -> None:
    with pytest.raises(ValueError, match="positive number"):
        validate_multiplier(multiplier)  # type: ignore[arg-type]


def test_validate_num_std_accepts_positive_numbers() -> None:
    validate_num_std(1)
    validate_num_std(2.0)


@pytest.mark.parametrize("num_std", [0, -1, True, "2"])
def test_validate_num_std_rejects_invalid_values(num_std: object) -> None:
    with pytest.raises(ValueError, match="positive number"):
        validate_num_std(num_std)  # type: ignore[arg-type]


def test_validate_fast_slow_signal_lengths_returns_the_tuple_on_success() -> None:
    result = validate_fast_slow_signal_lengths((12, 26, 9))

    assert result == (12, 26, 9)


def test_validate_fast_slow_signal_lengths_rejects_wrong_tuple_length() -> None:
    with pytest.raises(ValueError, match="fast, slow, and signal"):
        validate_fast_slow_signal_lengths((12, 26))  # type: ignore[arg-type]


@pytest.mark.parametrize("lengths", [(2, 2, 1), (3, 2, 1)])
def test_validate_fast_slow_signal_lengths_rejects_fast_not_below_slow(
    lengths: tuple[int, int, int],
) -> None:
    with pytest.raises(ValueError, match="fast length"):
        validate_fast_slow_signal_lengths(lengths)


@pytest.mark.parametrize("lengths", [(0, 26, 9), (12, 0, 9), (12, 26, 0), (True, 26, 9)])
def test_validate_fast_slow_signal_lengths_rejects_invalid_lengths(
    lengths: tuple[int, int, int],
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        validate_fast_slow_signal_lengths(lengths)  # type: ignore[arg-type]
