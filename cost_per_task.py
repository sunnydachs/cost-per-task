"""Measure what a task actually costs to complete with an LLM.

Instead of "tokens per request", this measures the cost of ONE COMPLETED TASK:
how many attempts it took, how many tokens were burned in total, and how long
the machine was busy. Verification is deterministic (no human in the loop), so
the numbers are reproducible and the "retry tax" is observable.

Everything is vendor-neutral: point it at any OpenAI-compatible endpoint.

    export NVIDIA_API_KEY=...            # or --api-key-env MY_KEY
    python cost_per_task.py --model A=vendor/model-one --model B=vendor/model-two

Outputs: results.json (every attempt) and report.md (aggregates).
Stdlib only — no runtime dependencies.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_KEY_ENV = "NVIDIA_API_KEY"
DEFAULT_TIMEOUT_S = 420
MAX_TOKENS = 4096


# ─────────────────────────────── verifiers ───────────────────────────────
def _strip_fence(text: str) -> str:
    """Remove code fences and surrounding whitespace."""
    blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", text, re.S)
    if blocks:
        return "\n".join(blocks).strip()
    return text.strip()


def _parse_csv(text: str):
    import csv
    import io
    rows = [r for r in csv.reader(io.StringIO(_strip_fence(text)))
            if any(c.strip() for c in r)]
    if not rows:
        return [], []
    return [c.strip() for c in rows[0]], [[c.strip() for c in r] for r in rows[1:]]


def verify_csv_convert(out: str):
    """Header, values and the dropped row must all match."""
    header, rows = _parse_csv(out)
    want_header = ["id", "name", "email", "country"]
    if header != want_header:
        return False, f"header mismatch: {header}"
    want = {
        ("1", "山田太郎", "taro@example.com", "Japan"),
        ("2", "鈴木花子", "hanako@example.com", "Japan"),
        ("4", "鈴木大輔", "dai@example.com", "USA"),
    }
    got = {tuple(r) for r in rows}
    if got != want:
        return False, f"rows mismatch: got={sorted(got)} want={sorted(want)}"
    return True, "match"


CHUNK_TEST = """
from chunk import chunk
assert chunk([1, 2, 3, 4, 5, 6, 7], 3) == [[1, 2, 3], [4, 5, 6], [7]], chunk([1,2,3,4,5,6,7], 3)
assert chunk([], 3) == []
assert chunk([1], 5) == [[1]]
print("OK")
"""


def verify_py_bugfix(out: str):
    """Run the returned code against a test file."""
    code = _strip_fence(out)
    if "def chunk" not in code:
        return False, "no chunk() function found"
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "chunk.py").write_text(code, encoding="utf-8")
        (Path(td) / "test_chunk.py").write_text(CHUNK_TEST, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "test_chunk.py"], cwd=td,
                               capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return False, "test timed out"
    if r.returncode == 0:
        return True, "tests passed"
    tail = (r.stderr or r.stdout).strip().splitlines()
    return False, "tests failed: " + (" / ".join(tail[-2:])[:200] if tail else "unknown")


def _extract_json(text: str):
    """Parse a JSON object/array out of a model response."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    if text.startswith("["):
        end = text.rfind("]")
        if end != -1:
            try:
                return json.loads(text[: end + 1])
            except json.JSONDecodeError:
                fixed = re.sub(r",\s*([}\]])", r"\1", text[: end + 1])
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None
    candidate = text[start: end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


def verify_json_build(out: str):
    """Every key and value of the order JSON must be right."""
    data = _extract_json(out)
    if not isinstance(data, dict):
        return False, "not a JSON object"
    want_keys = {"customer", "items", "shipping", "ordered_date", "total"}
    if set(data.keys()) != want_keys:
        return False, f"key mismatch: {sorted(data.keys())}"
    if data.get("customer") != "山田":
        return False, f"customer mismatch: {data.get('customer')!r}"
    if data.get("shipping") != "ヤマト":
        return False, f"shipping mismatch: {data.get('shipping')!r}"
    if data.get("ordered_date") != "2026-11-10":
        return False, f"ordered_date mismatch: {data.get('ordered_date')!r}"
    items = data.get("items")
    if not isinstance(items, list):
        return False, "items is not a list"
    got_items = {(i.get("name"), i.get("qty"), i.get("price")) for i in items
                 if isinstance(i, dict)}
    want_items = {("A", 2, 500), ("B", 1, 1200)}
    if got_items != want_items:
        return False, f"items mismatch: {got_items}"
    if data.get("total") != 2200:
        return False, f"total mismatch: {data.get('total')!r}"
    return True, "match"


def _digits(value) -> str:
    if isinstance(value, (int, float)):
        return str(int(value))
    return re.sub(r"[^\d]", "", str(value or ""))


def verify_doc_extract(out: str):
    """The total must be the invoice total, not the total plus the prior balance."""
    data = _extract_json(out)
    if not isinstance(data, dict):
        return False, "not a JSON object"
    checks = {
        "invoice_no": (lambda v: str(v).strip() == "INV-2026-114"),
        "issue_date": (lambda v: str(v).strip() == "2026-11-05"),
        "total_amount": (lambda v: _digits(v) == "530000"),
        "currency": (lambda v: str(v).strip().upper() == "JPY"),
        "vendor_name": (lambda v: str(v).strip() == "株式会社サンプル商事"),
    }
    for key, ok in checks.items():
        if key not in data:
            return False, f"missing key: {key}"
        if not ok(data[key]):
            return False, f"{key} mismatch: {data[key]!r}"
    return True, "match"


# ─────────────────────────────── tasks ───────────────────────────────
TASKS = [
    {
        "id": "csv_convert",
        "title": "Normalize a messy CSV (column mapping, row filtering)",
        "prompt": """次のCSVを、指定のルールで正規化してください。

```
ID, 名前 ,e-mail,country
1, 山田太郎, TARO@Example.COM ,jp
2, 鈴木花子, hanako@example.com, JP
3, 佐藤次郎,, jp
4, 鈴木大輔, DAI@Example.com,us
```

ルール:
- 列は id,name,email,country の順（ヘッダーも出力する）
- 各セルの前後の空白は取り除き、email は小文字にする
- country は jp を Japan、us を USA にする
- email が空の行は捨てる
- 出力はCSVのテキストだけ。説明文やコードフェンスは不要

出力:""",
        "verify": verify_csv_convert,
    },
    {
        "id": "py_bugfix",
        "title": "Fix a Python bug (judged by running a test)",
        "prompt": """下の関数はバグがあります。テストが通るように直してください。

```python
# chunk.py
def chunk(items, size):
    out = []
    for i in range(0, len(items), size):
        out.append(items[i:i + size - 1])
    return out
```

テスト:
```python
from chunk import chunk
assert chunk([1, 2, 3, 4, 5, 6, 7], 3) == [[1, 2, 3], [4, 5, 6], [7]]
assert chunk([], 3) == []
assert chunk([1], 5) == [[1]]
```

回答は修正後の chunk.py の全文を、コードブロック1つで返してください。説明は不要です。""",
        "verify": verify_py_bugfix,
    },
    {
        "id": "json_build",
        "title": "Build a JSON document from order details",
        "prompt": """次の注文情報からJSONを作ってください。

顧客: 山田
商品: A を2個 × 500円, B を1個 × 1200円
配送: ヤマト
注文日: 2026-11-10

出力JSONのキー:
- customer: 顧客名
- items: 配列。各要素は name, qty, price
- shipping: 配送先
- ordered_date: 注文日（YYYY-MM-DD）
- total: items の合計金額

説明文は不要です。JSONだけ出力してください。""",
        "verify": verify_json_build,
    },
    {
        "id": "doc_extract",
        "title": "Extract fields from a noisy invoice",
        "prompt": """次の請求書から、次の5項目をJSONで抽出してください。
- invoice_no
- issue_date（YYYY-MM-DD）
- total_amount（数字のみ）
- currency
- vendor_name

```
INVOICE
株式会社サンプル商事
Invoice No: INV-2026-114
発行日: 2026-11-05
前月未払: 12,000
商品: CONSULTING 10 hours x 50,000 = 500,000
商品: SUPPORT 1 month = 30,000
Total: 530,000 JPY
お支払期限: 2026-12-05
お問い合わせ: billing@example.com
-- Page 1/1 --
```

注意: 前月未払・支払期限・問い合わせ先・ページ番号は使いません。説明文は不要です。JSONだけ出力してください。""",
        "verify": verify_doc_extract,
    },
]


# ─────────────────────────────── API call ───────────────────────────────
def call_model(base_url: str, key: str, model: str, prompt: str,
               timeout: int = DEFAULT_TIMEOUT_S,
               transport_retries: int = 2) -> dict:
    """One task attempt. Transport errors are retried inside the attempt and
    are NOT counted as attempts; only verifier failures count."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": model,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    errors = []
    t0 = time.time()
    for attempt in range(transport_retries + 1):
        req = Request(url, data=body, headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        })
        try:
            with urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:200]
            errors.append(f"HTTP {e.code}: {detail}")
            if e.code not in (429, 503):
                break
        except (URLError, TimeoutError, OSError, ValueError) as e:
            errors.append(f"exception: {e}")
        else:
            choices = payload.get("choices") or []
            content = ""
            if choices:
                content = (choices[0].get("message") or {}).get("content") or ""
            if content.strip():
                usage = payload.get("usage") or {}
                return {
                    "text": content.strip(),
                    "usage": {
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                    },
                    "latency_s": round(time.time() - t0, 2),
                    "transport_errors": errors,
                }
            errors.append("empty response")
        time.sleep(min(5 * (attempt + 1), 30))
    return {
        "text": "",
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        "latency_s": round(time.time() - t0, 2),
        "transport_errors": errors,
    }


# ─────────────────────────────── runner ───────────────────────────────
def load_results(path: Path) -> list:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []
    return []


def save_results(results: list, path: Path) -> None:
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def run(tasks, arms, base_url, key, results_path, max_attempts=3, delay_s=2.0):
    """Run every (task, arm) pair. Resumable: already-solved pairs are skipped
    and already-recorded attempts are not repeated."""
    results = load_results(results_path)
    solved = {(r["task_id"], r["arm"]) for r in results if r["ok"]}
    done = {(r["task_id"], r["arm"], r["attempt"]) for r in results}
    for task in tasks:
        for label, model in arms:
            if (task["id"], label) in solved:
                continue
            for attempt in range(1, max_attempts + 1):
                if (task["id"], label, attempt) in done:
                    continue
                print(f"[run] {task['id']} {label} attempt {attempt}", flush=True)
                call = call_model(base_url, key, model, task["prompt"])
                text = call["text"]
                ok, detail = (False, "no response") if not text else task["verify"](text)
                results.append({
                    "task_id": task["id"],
                    "arm": label,
                    "model": model,
                    "attempt": attempt,
                    "at": datetime.now(timezone.utc).isoformat(),
                    "ok": ok,
                    "detail": detail,
                    "usage": call["usage"],
                    "latency_s": call["latency_s"],
                    "transport_errors": call["transport_errors"],
                    "response": text,
                })
                save_results(results, results_path)
                print(f"       -> ok={ok} detail={detail[:80]} "
                      f"tokens={call['usage'].get('total_tokens')} "
                      f"{call['latency_s']}s", flush=True)
                if ok:
                    break
                time.sleep(delay_s)
    return results


# ─────────────────────────────── aggregation ───────────────────────────────
def aggregate(results, arms, tasks=TASKS):
    """Per-arm aggregates. Cost is reported in TOKENS (the cash cost depends on
    the provider and plan, so it is never invented here)."""
    out = {}
    for label, _model in arms:
        rows = [r for r in results if r["arm"] == label]
        solved = 0
        tokens_solved = 0
        attempts_solved = 0
        all_tokens = 0
        all_attempts = 0
        all_seconds = 0.0
        first_attempt_ok = 0
        per_task = []
        for task in tasks:
            trs = sorted([r for r in rows if r["task_id"] == task["id"]],
                         key=lambda r: r["attempt"])
            if not trs:
                continue
            success = next((r for r in trs if r["ok"]), None)
            if success:
                trs = [r for r in trs if r["attempt"] <= success["attempt"]]
            attempts_used = success["attempt"] if success else len(trs)
            tokens = sum((r["usage"] or {}).get("total_tokens") or 0 for r in trs)
            seconds = sum(r["latency_s"] or 0 for r in trs)
            per_task.append({
                "task_id": task["id"],
                "title": task["title"],
                "solved": bool(success),
                "attempts_used": attempts_used,
                "attempts_made": len(trs),
                "tokens_total": tokens,
                "seconds_total": round(seconds, 1),
                "first_attempt_ok": bool(trs[0]["ok"]),
                "detail": success["detail"] if success else (trs[-1]["detail"] or ""),
            })
            all_tokens += tokens
            all_attempts += len(trs)
            all_seconds += seconds
            if success:
                solved += 1
                tokens_solved += tokens
                attempts_solved += attempts_used
                if trs[0]["ok"]:
                    first_attempt_ok += 1
        n = len(per_task)
        out[label] = {
            "per_task": per_task,
            "solved": solved,
            "tasks": n,
            "first_attempt_success_rate": round(first_attempt_ok / n, 3) if n else 0,
            "retry_tax": round(all_attempts / solved, 3) if solved else None,
            "tokens_all_attempts": all_tokens,
            "tokens_per_solved_task": round(tokens_solved / solved, 1) if solved else None,
            "tokens_per_success_incl_failed_waste": (
                round(all_tokens / solved, 1) if solved else None),
            "attempts_per_solved_task": (
                round(attempts_solved / solved, 3) if solved else None),
            "seconds_total": round(all_seconds, 1),
        }
    return out


def render_report(agg, arms) -> str:
    labels = [label for label, _ in arms]
    lines = ["# Task completion cost — measured", "",
             f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
             "Cost is reported in TOKENS; the cash cost depends on the provider and "
             "plan, so it is not invented here.",
             "Each task is retried with the same prompt (the verifier's error is "
             "not fed back to the model).", "",
             "## Per task", "",
             "| task | arm | result | attempts | tokens | seconds | verdict |",
             "|---|---|---|---:|---:|---:|---|"]
    for label in labels:
        for t in agg[label]["per_task"]:
            lines.append(
                f"| {t['task_id']} | {label} | {'PASS' if t['solved'] else 'FAIL'} | "
                f"{t['attempts_used']}/{t['attempts_made']} | {t['tokens_total']} | "
                f"{t['seconds_total']} | {t['detail'][:40]} |")
    lines += ["", "## Per arm", "",
              "| metric | " + " | ".join(labels) + " |",
              "|---|" + "---|" * len(labels)]
    rows = [
        ("tasks solved", lambda a: f"{a['solved']}/{a['tasks']}"),
        ("first-attempt success rate", lambda a: f"{a['first_attempt_success_rate'] * 100:.0f}%"),
        ("retry tax (attempts / solved)", lambda a: a["retry_tax"]),
        ("tokens per solved task", lambda a: a["tokens_per_solved_task"]),
        ("tokens per success incl. failed-task waste", lambda a: a["tokens_per_success_incl_failed_waste"]),
        ("attempts per solved task", lambda a: a["attempts_per_solved_task"]),
        ("total tokens", lambda a: a["tokens_all_attempts"]),
        ("total seconds (machine time)", lambda a: a["seconds_total"]),
    ]
    for name, fn in rows:
        lines.append(f"| {name} | " + " | ".join(str(fn(agg[l])) for l in labels) + " |")
    lines += ["", "## Limits (stated, not hidden)", "",
              f"- {len(TASKS)} tasks, one language, {len(labels)} arms — too small to "
              "generalise about any model.",
              "- The cash cost is unknown unless your provider tells you; tokens and "
              "wall-clock are what this tool measures.",
              "- Retries repeat the same prompt; the verifier error is not fed back.",
              "- A transport failure (HTTP 429/503/timeout) is retried inside one "
              "attempt and is not counted as a task attempt.",
              "- Human review minutes are not measured here — no observability product "
              "records them either."]
    return "\n".join(lines) + "\n"


def parse_arm(spec: str):
    """'LABEL=model-id' -> (LABEL, model-id)."""
    if "=" in spec:
        label, model = spec.split("=", 1)
        return label.strip(), model.strip()
    return f"arm{abs(hash(spec)) % 1000}", spec.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True,
                    help="LABEL=model-id (repeat for several arms)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--api-key-env", default=DEFAULT_KEY_ENV)
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="task ids to run (default: all)")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--report-only", action="store_true",
                    help="re-render the report from an existing results.json")
    args = ap.parse_args()

    arms = [parse_arm(m) for m in args.model]
    tasks = [t for t in TASKS if not args.tasks or t["id"] in args.tasks]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    report_path = out_dir / "report.md"

    if not args.report_only:
        key = os.environ.get(args.api_key_env, "")
        if not key:
            print(f"error: environment variable {args.api_key_env} is not set",
                  file=sys.stderr)
            return 2
        results = run(tasks, arms, args.base_url, key, results_path,
                      max_attempts=args.max_attempts)
    else:
        results = load_results(results_path)

    agg = aggregate(results, arms, tasks=tasks)
    report_path.write_text(render_report(agg, arms), encoding="utf-8")
    print(f"[report] {report_path}")
    print(f"[data] {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
