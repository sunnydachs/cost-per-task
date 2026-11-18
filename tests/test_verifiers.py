"""Tests for the verifiers and the aggregation math.

The verifiers decide PASS/FAIL for every attempt, so a wrong verifier turns
the whole measurement into fiction. These tests pin the accept/reject
behaviour. No network access: the LLM is never called.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cost_per_task as cpt  # noqa: E402

GOOD_CSV = (
    "id,name,email,country\n"
    "1,山田太郎,taro@example.com,Japan\n"
    "2,鈴木花子,hanako@example.com,Japan\n"
    "4,鈴木大輔,dai@example.com,USA\n"
)
GOOD_PY = (
    "```python\n"
    "def chunk(items, size):\n"
    "    out = []\n"
    "    for i in range(0, len(items), size):\n"
    "        out.append(items[i:i + size])\n"
    "    return out\n"
    "```"
)
BAD_PY = (
    "```python\n"
    "def chunk(items, size):\n"
    "    out = []\n"
    "    for i in range(0, len(items), size):\n"
    "        out.append(items[i:i + size - 1])\n"
    "    return out\n"
    "```"
)
GOOD_JSON = (
    '{"customer":"山田","items":[{"name":"A","qty":2,"price":500},'
    '{"name":"B","qty":1,"price":1200}],"shipping":"ヤマト",'
    '"ordered_date":"2026-11-10","total":2200}'
)
GOOD_DOC = (
    '{"invoice_no":"INV-2026-114","issue_date":"2026-11-05","total_amount":530000,'
    '"currency":"JPY","vendor_name":"株式会社サンプル商事"}'
)


def test_csv_verifier_accepts_correct_output():
    ok, detail = cpt.verify_csv_convert(GOOD_CSV)
    assert ok, detail


def test_csv_verifier_rejects_wrong_country_and_extra_row():
    ok, _ = cpt.verify_csv_convert(GOOD_CSV.replace("Japan", "JP"))
    assert not ok
    ok, _ = cpt.verify_csv_convert(GOOD_CSV + "3,佐藤次郎,,Japan\n")
    assert not ok


def test_py_verifier_accepts_fixed_and_rejects_buggy():
    ok, detail = cpt.verify_py_bugfix(GOOD_PY)
    assert ok, detail
    ok, _ = cpt.verify_py_bugfix(BAD_PY)
    assert not ok


def test_json_verifier_accepts_correct_and_rejects_wrong_total():
    ok, detail = cpt.verify_json_build(GOOD_JSON)
    assert ok, detail
    ok, _ = cpt.verify_json_build(GOOD_JSON.replace("2200", "3200"))
    assert not ok


def test_doc_verifier_rejects_total_including_prior_balance():
    ok, detail = cpt.verify_doc_extract(GOOD_DOC)
    assert ok, detail
    ok, _ = cpt.verify_doc_extract(GOOD_DOC.replace("530000", "542000"))
    assert not ok
    ok, _ = cpt.verify_doc_extract(GOOD_DOC.replace("2026-11-05", "2026-12-05"))
    assert not ok


def test_aggregate_math():
    """Two tasks: one passes on attempt 1, one on attempt 3 -> retry tax 4/2 = 2.0."""
    results = [
        {"task_id": "csv_convert", "arm": "A", "attempt": 1, "ok": True,
         "detail": "match", "usage": {"total_tokens": 100}, "latency_s": 5.0,
         "transport_errors": [], "response": ""},
        {"task_id": "json_build", "arm": "A", "attempt": 1, "ok": False,
         "detail": "mismatch", "usage": {"total_tokens": 100}, "latency_s": 5.0,
         "transport_errors": [], "response": ""},
        {"task_id": "json_build", "arm": "A", "attempt": 2, "ok": False,
         "detail": "mismatch", "usage": {"total_tokens": 100}, "latency_s": 5.0,
         "transport_errors": [], "response": ""},
        {"task_id": "json_build", "arm": "A", "attempt": 3, "ok": True,
         "detail": "match", "usage": {"total_tokens": 100}, "latency_s": 5.0,
         "transport_errors": [], "response": ""},
    ]
    agg = cpt.aggregate(results, [("A", "vendor/model-a")])["A"]
    assert agg["solved"] == 2
    assert agg["retry_tax"] == 2.0
    assert agg["first_attempt_success_rate"] == 0.5
    assert agg["tokens_per_solved_task"] == 200.0
    assert agg["tokens_per_success_incl_failed_waste"] == 200.0
    assert agg["attempts_per_solved_task"] == 2.0
