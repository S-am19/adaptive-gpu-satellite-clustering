import csv
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from benchmark_adaptive_v2 import write_measurements


def test_measurement_writer_unions_method_specific_timing_columns(tmp_path):
    path = tmp_path / "measurements.csv"
    records = [
        {"method": "exact", "timing_assignment_s": 1.0},
        {"method": "decayed", "timing_assignment_s": 2.0,
         "timing_sampling_provenance_s": 0.1},
    ]

    write_measurements(path, records)

    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["timing_sampling_provenance_s"] == ""
    assert rows[1]["timing_sampling_provenance_s"] == "0.1"
    assert rows[0]["timing_assignment_s"] == "1.0"
    assert rows[1]["timing_assignment_s"] == "2.0"
