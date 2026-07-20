"""Private parameter validation for indicator calculations."""


def validate_length(length: int) -> None:
    """Validate that a window length is a positive integer."""
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError(f"Length must be a positive integer; got {length!r}")


def validate_multiplier(multiplier: float) -> None:
    """Validate that a band multiplier is a positive number."""
    if isinstance(multiplier, bool) or not isinstance(multiplier, int | float) or multiplier <= 0:
        raise ValueError(f"Multiplier must be a positive number; got {multiplier!r}")


def validate_num_std(num_std: float) -> None:
    """Validate that a standard-deviation multiplier is a positive number."""
    if isinstance(num_std, bool) or not isinstance(num_std, int | float) or num_std <= 0:
        raise ValueError(f"num_std must be a positive number; got {num_std!r}")


def validate_fast_slow_signal_lengths(lengths: tuple[int, int, int]) -> tuple[int, int, int]:
    """Validate the fast, slow, and signal lengths shared by MACD and PPO."""
    if len(lengths) != 3:
        raise ValueError("Lengths must contain fast, slow, and signal lengths.")

    fast_length, slow_length, signal_length = lengths
    validate_length(fast_length)
    validate_length(slow_length)
    validate_length(signal_length)
    if fast_length >= slow_length:
        raise ValueError(
            "The fast length must be lower than the slow length; "
            f"got fast={fast_length!r}, slow={slow_length!r}"
        )
    return fast_length, slow_length, signal_length
