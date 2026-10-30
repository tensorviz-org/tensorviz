import pytest

from scripts.assert_sweep_report import main


@pytest.mark.parametrize("single,combinations,expected", [
    ("448/448", "2/2", 0),
    ("448/448", "0/2", 1),
    ("448/448", "1/2", 1),
    ("448/448", "0/0", 1),
    ("447/448", "2/2", 1),
    ("0/0", "2/2", 1),
])
def test_gate_requires_complete_single_and_combination_validation(tmp_path, single, combinations, expected):
    report = tmp_path / "sweep.md"
    report.write_text(f"""# Template Validation Sweep Report
- Total templates discovered: **224**
- Single-template checks: **{single} passed**
- Combination checks: **{combinations} passed**
""")
    assert main(str(report)) == expected
