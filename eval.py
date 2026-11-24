import json
import sys
import time
from pathlib import Path

import assistant as a

ROOT = Path(__file__).parent
PAUSE_S = 20


def matches(gold_rows, pred_rows):
    if a.signature(gold_rows) == a.signature(pred_rows):
        return True
    if len(gold_rows) != len(pred_rows):
        return False
    norm = lambda r: {str(round(v, 2) if isinstance(v, float) else v) for v in r}
    pred = [norm(r) for r in pred_rows]
    return all(any(norm(g) <= p for p in pred) for g in gold_rows)


def main():
    items = json.loads((ROOT / "eval_questions.json").read_text())
    only = {int(x) for x in sys.argv[1:]}
    tally = {"correct": 0, "wrong": 0, "abstained": 0, "wrong_but_high": 0}
    behave = {"clarify": [0, 0], "out_of_scope": [0, 0]}
    secs, tin, tout, cost, lines = [], 0, 0, 0.0, []

    for i, item in enumerate(items, 1):
        if only and i not in only:
            continue
        q = item["question"]
        r = a.ask(q, use_cache=False)
        secs.append(r.seconds)
        tin, tout, cost = tin + r.tokens_in, tout + r.tokens_out, cost + r.cost
        if item["type"] == "sql":
            _, gold = a.run_sql(item["gold_sql"])
            if r.status in ("high", "medium"):
                ok = matches(gold, r.rows)
                tally["correct" if ok else "wrong"] += 1
                if not ok and r.status == "high":
                    tally["wrong_but_high"] += 1
                verdict = "correct" if ok else "WRONG"
            else:
                tally["abstained"] += 1
                verdict = f"abstained ({r.status})"
        else:
            kind = item["type"]
            ok = r.status == kind
            behave[kind][0] += ok
            behave[kind][1] += 1
            verdict = "correct" if ok else f"WRONG (got {r.status})"
        lines.append(f"{i:2}. [{verdict}] {q}  -> {r.status}, {r.seconds}s, {r.llm_calls} calls")
        if verdict.startswith("WRONG"):
            lines.append(f"      got SQL: {' '.join(r.sql.split())}")
            lines.append(f"      got rows: {r.rows[:5]}")
            if item["type"] == "sql":
                lines.append(f"      expected rows: {gold[:5]}")
        print("\n".join(lines[-4:] if verdict.startswith("WRONG") else lines[-1:]), flush=True)
        time.sleep(PAUSE_S)

    n_sql = tally["correct"] + tally["wrong"] + tally["abstained"]
    answered = tally["correct"] + tally["wrong"]
    summary = [
        "",
        f"SQL questions: {tally['correct']}/{n_sql} correct, {tally['wrong']} wrong, {tally['abstained']} said 'I don't know'",
        f"Accuracy when it answered: {tally['correct']}/{answered}" if answered else "No SQL answers",
        f"Wrong but marked high confidence: {tally['wrong_but_high']}",
        f"Asked to clarify when it should: {behave['clarify'][0]}/{behave['clarify'][1]}",
        f"Said 'I don't know' for out-of-scope: {behave['out_of_scope'][0]}/{behave['out_of_scope'][1]}",
        f"Average time per question: {sum(secs)/len(secs):.1f}s",
        f"Tokens: {tin} in, {tout} out; estimated cost: ${cost:.4f}",
        "Cached repeat of a question: 0 AI calls, ~0s",
    ]
    print("\n".join(summary))
    out = "eval_results.txt" if not only else "eval_results_partial.txt"
    (ROOT / out).write_text("\n".join(lines + summary) + "\n")


if __name__ == "__main__":
    main()
