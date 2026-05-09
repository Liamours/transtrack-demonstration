from src.transtrack_demo import ui


def test_inference_line_prints_all_scores(capsys):
    ui.inference_line(
        {
            "label": "normal",
            "confidence": 0.8,
            "scores": {
                "eyes_closed": 0.1,
                "normal": 0.8,
                "yawning": 0.1,
            },
        }
    )

    output = capsys.readouterr().out
    assert "eyes_closed" in output
    assert "normal" in output
    assert "yawning" in output
