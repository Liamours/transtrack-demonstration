import csv

from src.transtrack_demo.logger import CsvLogger


def test_csv_logger_creates_header_and_appends_result(tmp_path):
    log_path = tmp_path / "logs" / "fatigue.csv"
    logger = CsvLogger(log_path)

    logger.write({"label": "normal", "confidence": 0.81234, "class_id": 1})

    with log_path.open(newline="") as file:
        rows = list(csv.reader(file))

    assert rows[0] == ["timestamp", "label", "confidence", "class_id"]
    assert rows[1][1:] == ["normal", "0.8123", "1"]
