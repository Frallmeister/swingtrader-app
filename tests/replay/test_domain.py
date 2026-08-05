import pytest

from swingtrader.replay.domain import AVANZA_COURTAGE_PROFILES, CourtageProfileName


@pytest.mark.parametrize(
    ("name", "gross", "expected"),
    [
        (CourtageProfileName.MINI, 100.0, 1.0),
        (CourtageProfileName.MINI, 10_000.0, 25.0),
        (CourtageProfileName.SMALL, 10_000.0, 39.0),
        (CourtageProfileName.MEDIUM, 200_000.0, 138.0),
        (CourtageProfileName.FIXED, 10_000.0, 99.0),
    ],
)
def test_avanza_courtage_profiles(name, gross, expected):
    assert AVANZA_COURTAGE_PROFILES[name].calculate(gross) == pytest.approx(expected)
