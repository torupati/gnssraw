import csv
from pathlib import Path

import pytest

from app import rnxproc


@pytest.fixture
def sample_rinex_file():
    """Return the path to a sample RINEX file."""
    sample_file = (
        Path(__file__).parent.parent
        / "sample_data"
        / "static_baseline"
        / "3075358x.25o"
    )
    if not sample_file.exists():
        pytest.skip(f"Sample RINEX file not found: {sample_file}")
    return sample_file


@pytest.fixture
def signal_code_map_file():
    """Return the path to the signal code map file."""
    map_file = Path(__file__).parent.parent / "app" / ".signal_code_map.json"
    if not map_file.exists():
        pytest.skip(f"Signal code map file not found: {map_file}")
    return map_file


def test_main_missing_signal_code_map_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_rinex_file: Path
):
    missing_map = tmp_path / "missing_signal_code_map.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "rnxproc",
            str(sample_rinex_file),
            "--outdir",
            str(tmp_path / "out"),
            "--signal-code-map",
            str(missing_map),
        ],
    )

    assert rnxproc.main() == 1


def test_main_invalid_signal_code_map_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_rinex_file: Path
):
    invalid_map = tmp_path / "invalid_signal_code_map.json"
    invalid_map.write_text("{", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "rnxproc",
            str(sample_rinex_file),
            "--outdir",
            str(tmp_path / "out"),
            "--signal-code-map",
            str(invalid_map),
        ],
    )

    assert rnxproc.main() == 1


def test_csv_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_rinex_file: Path,
    signal_code_map_file: Path,
):
    """Test CSV export of ambiguity statistics."""
    csv_output = tmp_path / "test_statistics.csv"

    monkeypatch.setattr(
        "sys.argv",
        [
            "rnxproc",
            str(sample_rinex_file),
            "--outdir",
            str(tmp_path / "out"),
            "--signal-code-map",
            str(signal_code_map_file),
            "--csv",
            str(csv_output),
            "--skip-plot",
        ],
    )

    result = rnxproc.main()
    assert result == 0, "rnxproc.main() should return 0 on success"

    # Verify CSV file exists
    assert csv_output.exists(), f"CSV file should exist at {csv_output}"

    # Read and validate CSV content
    with open(csv_output, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        # Check that CSV has data
        assert len(rows) > 0, "CSV should contain at least one row of data"

        # Check that all required columns exist
        expected_columns = [
            "start_time",
            "end_time",
            "satellite",
            "band",
            "num_epochs",
            "widelane_ambiguity_mean",
            "widelane_ambiguity_std",
            "widelane_ambiguity_max_min",
            "snr_mean",
            "snr_max_min",
        ]
        assert reader.fieldnames is not None, "CSV file should have headers"
        for col in expected_columns:
            assert col in reader.fieldnames, f"Column '{col}' should exist in CSV"

        # Validate first row has reasonable values
        first_row = rows[0]
        assert first_row["satellite"].startswith(("G", "E", "J", "R")), (
            "Satellite ID should start with G, E, J, or R"
        )
        assert "_" in first_row["band"], (
            "Band should be a frequency combination like L1_L2"
        )
        assert int(first_row["num_epochs"]) > 0, "num_epochs should be greater than 0"
