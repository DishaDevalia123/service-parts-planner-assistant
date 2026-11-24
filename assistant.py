import json
import os
import re
import sqlite3
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

DB_PATH = ROOT / "planner.db"
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.getenv("LLM_API_KEY", "")
SQL_MODEL = os.getenv("SQL_MODEL", "openai/gpt-oss-120b")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "openai/gpt-oss-20b")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")
PRICE_IN = float(os.getenv("PRICE_IN_PER_M", "0"))
PRICE_OUT = float(os.getenv("PRICE_OUT_PER_M", "0"))
MAX_ROWS = 200
QUERY_TIMEOUT_S = 3

ALLOWED_TABLES = {
    "order_lines", "line_substitutes", "inventory", "fills", "remaining_inventory",
    "line_status", "shortage_by_part_warehouse", "data_info",
}

SCHEMA = """
Tables (SQLite):
- order_lines(line_id, order_id, order_date TEXT 'YYYY-MM-DD', qty, warehouse_id, part, parent_warehouse_id, age_days, is_old, is_bulk)
  One row per requested order line. order_id is NOT unique across warehouses, so count lines with line_id.
  is_old = 1 if older than 90 days before the as-of date. is_bulk = 1 if qty >= 10.
- line_substitutes(line_id, priority, compatible_part): parts allowed as substitutes for that line, in priority order.
- inventory(part, warehouse_id, on_hand_qty): starting stock BEFORE allocation.
- fills(line_id, fill_type, source_warehouse_id, source_part, filled_qty): result of the approved allocation.
  fill_type is one of 'own_part', 'own_substitute', 'parent_part', 'parent_substitute'.
  Allocation rule: oldest order first; own warehouse part -> own warehouse substitute -> parent warehouse part -> parent warehouse substitute.
- remaining_inventory(part, warehouse_id, remaining_qty): stock left AFTER allocation.
- line_status (view): line_id, order_id, order_date, warehouse_id, part, requested_qty, filled_qty, short_qty, is_complete, is_bulk, is_old, age_days
- shortage_by_part_warehouse (view): part, warehouse_id, requested_qty, filled_qty, short_qty, lines, incomplete_lines
- data_info(as_of_date, old_order_days, bulk_qty)

Facts: parts are 'Part A', 'Part B', 'Part C', 'Part D'. Warehouses are 1-4 (integers).
Warehouse 1 is the parent of warehouses 1 and 2. Warehouse 4 is the parent of warehouse 3. Warehouse 4 has no orders of its own.

Business definitions (always use these):
- demand = SUM(qty) from order_lines (units). "orders" in a count means order lines (COUNT of line_id).
- filled = SUM(filled_qty); short / shortage = SUM(short_qty) from line_status.
- fill rate = filled units / requested units, as a percent rounded to 1 decimal.
- complete line = is_complete = 1.
"""

EXAMPLES = """
Verified examples:
Q: What is the total demand?
{"action":"sql","sql":"SELECT COUNT(*) AS order_lines, SUM(qty) AS demand_units FROM order_lines","assumption":""}
Q: Fill rate by part
{"action":"sql","sql":"SELECT part, SUM(filled_qty) AS filled, SUM(requested_qty) AS requested, ROUND(100.0*SUM(filled_qty)/SUM(requested_qty),1) AS fill_rate_pct FROM line_status GROUP BY part ORDER BY part","assumption":""}
Q: How much stock of Part C do we have at warehouse 2?
{"action":"clarify","question":"Do you mean starting stock, or stock left after filling the open orders?","options":["Starting stock","Stock left after filling orders"]}
Q: When will the supplier deliver Part B?
{"action":"out_of_scope","reason":"The data has orders and stock only. It has no supplier or delivery dates."}
"""

SYSTEM_PROMPT = f"""You turn a service-parts planner's question into one SQLite query.
{SCHEMA}
{EXAMPLES}
Rules:
- Reply with ONE JSON object only, no other text.
- Use action "sql" when the question can be answered from these tables. Write a single SELECT (CTEs allowed). Use only the tables above.
- Use action "clarify" when the question has two readings that give different numbers (for example starting vs remaining stock, or "late" when there are no due dates). Give 2-3 short options.
- line_id is an internal key planners never see. When the answer names an order or order line, return order_id, warehouse_id and part (never only line_id).
- If the question already says which reading (for example "starting stock" or "older than 30 days"), use that and answer with sql.
- Use action "out_of_scope" when the data cannot answer it (suppliers, prices, lead times, future forecasts, anything not in the tables).
- Put any assumption you made in "assumption" (empty string if none).
"""

SUMMARY_PROMPT = """Write a 1-3 sentence answer for a parts planner, using ONLY numbers that appear in the result rows.
Do not calculate new numbers. Do not guess. Plain business language."""


@dataclass
class Result:
    question: str
    status: str
    answer: str = ""
    sql: str = ""
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    options: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    assumption: str = ""
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    cached: bool = False

    @property
    def cost(self):
        return (self.tokens_in * PRICE_IN + self.tokens_out * PRICE_OUT) / 1_000_000


class Meter:
    def __init__(self):
        self.calls = self.tin = self.tout = 0
        self.lock = threading.Lock()

    def add(self, usage):
        with self.lock:
            self.calls += 1
            if usage:
                self.tin += usage.prompt_tokens or 0
                self.tout += usage.completion_tokens or 0


_client = None
_cache = {}


def client():
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError("LLM_API_KEY is missing. Add it to the .env file.")
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    return _client


def call_llm(model, messages, meter, temperature=0.0):
    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if REASONING_EFFORT:
        kwargs["reasoning_effort"] = REASONING_EFFORT
    for attempt in range(4):
        try:
            resp = client().chat.completions.create(**kwargs)
            meter.add(resp.usage)
            return resp.choices[0].message.content or ""
        except RateLimitError as e:
            wait = 2 ** attempt * 5
            try:
                wait = float(e.response.headers.get("retry-after", wait))
            except Exception:
                pass
            time.sleep(min(wait, 60))
    raise RuntimeError("Rate limit: too many requests. Wait a minute and try again.")


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"action": "error", "reason": "no JSON in reply"}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"action": "error", "reason": "bad JSON in reply"}


def _authorizer(action, arg1, arg2, dbname, source):
    if action == sqlite3.SQLITE_SELECT or action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if (arg1 in ALLOWED_TABLES or source) else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_DENY


def run_sql(sql):
    sql = sql.strip().rstrip(";").strip()
    if ";" in sql:
        raise ValueError("Only one statement is allowed.")
    if not re.match(r"(?is)^\s*(select|with)\b", sql):
        raise ValueError("Only SELECT queries are allowed.")
    con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        con.set_authorizer(_authorizer)
        deadline = time.time() + QUERY_TIMEOUT_S
        con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10000)
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(MAX_ROWS)
        return cols, [list(r) for r in rows]
    finally:
        con.close()


def signature(rows):
    norm = []
    for r in rows:
        vals = [round(v, 2) if isinstance(v, float) else v for v in r]
        norm.append(tuple(sorted(map(str, vals))))
    return tuple(sorted(norm))


def attempt(question, meter, temperature):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    out = parse_json(call_llm(SQL_MODEL, messages, meter, temperature))
    if out.get("action") != "sql":
        return out
    sql = out.get("sql", "")
    for tries in range(2):
        try:
            out["columns"], out["rows"] = run_sql(sql)
            out["sql"] = sql
            out["repaired"] = tries > 0
            return out
        except Exception as e:
            if tries == 1:
                return {"action": "error", "reason": f"query failed after repair: {e}", "sql": sql}
            messages += [{"role": "assistant", "content": json.dumps(out)},
                         {"role": "user", "content": f"That SQL failed with: {e}. Reply with corrected JSON."}]
            fixed = parse_json(call_llm(SQL_MODEL, messages, meter, 0.0))
            if fixed.get("action") != "sql":
                return fixed
            out, sql = fixed, fixed.get("sql", "")


def numbers_in(text):
    return {float(n.replace(",", "")) for n in re.findall(r"-?\d[\d,]*\.?\d*", text)}


def summarize(question, cols, rows, meter):
    table = json.dumps({"columns": cols, "rows": rows[:30]}, default=str)
    text = call_llm(SUMMARY_MODEL, [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": f"Question: {question}\nResult: {table}"},
    ], meter).strip()
    allowed = numbers_in(table) | numbers_in(question)
    unsupported = [n for n in numbers_in(text) if n not in allowed]
    return text, unsupported


def normalize(q):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", q.lower())).strip()


AMBIGUOUS_TERMS = [
    {
        "term": "stock",
        "trigger": r"\b(stock|inventory|on hand|on-hand)\b|\bhow much part [a-d]\b|\bhow many part [a-d]\b",
        "resolved": r"\b(start\w*|initial|before|left|remaining|after|end)\b",
        "question": "Do you mean starting stock, or stock left after filling the open orders?",
        "options": ["Starting stock", "Stock left after filling orders"],
    },
    {
        "term": "late",
        "trigger": r"\b(late|overdue|delayed|past due)\b",
        "resolved": r"\bdays\b|\bnot (fully )?filled\b",
        "question": "The data has no due dates. What should count as late?",
        "options": ["Older than 30 days and not fully filled", "Older than 90 days and not fully filled",
                    "Any order not fully filled"],
    },
]


def check_ambiguity(question):
    q = question.lower()
    for rule in AMBIGUOUS_TERMS:
        if re.search(rule["trigger"], q) and not re.search(rule["resolved"], q):
            return Result(question, "clarify", answer=rule["question"], options=rule["options"],
                          checks=[f"'{rule['term']}' has more than one meaning in this data: asked instead of guessing (no AI call)"])
    return None


def ask(question, use_cache=True):
    key = normalize(question)
    if use_cache and key in _cache:
        hit = Result(**{**_cache[key].__dict__})
        hit.cached, hit.seconds, hit.llm_calls, hit.tokens_in, hit.tokens_out = True, 0.0, 0, 0, 0
        hit.checks = ["Same question asked before: answer reused from cache, no AI call"] + hit.checks
        return hit

    start, meter = time.time(), Meter()
    ruled = check_ambiguity(question)
    if ruled:
        ruled.seconds = round(time.time() - start, 2)
        return ruled
    with ThreadPoolExecutor(2) as pool:
        tries = list(pool.map(lambda t: attempt(question, meter, t), [0.0, 0.7]))
    res = decide(question, tries, meter)
    res.llm_calls, res.tokens_in, res.tokens_out = meter.calls, meter.tin, meter.tout
    res.seconds = round(time.time() - start, 2)
    if res.status in ("high", "medium", "clarify", "out_of_scope"):
        _cache[key] = res
    return res


def decide(question, tries, meter):
    actions = [t.get("action") for t in tries]
    sql_tries = [t for t in tries if t.get("action") == "sql"]
    sigs = [signature(t["rows"]) for t in sql_tries]

    agree = len(sql_tries) == 2 and sigs[0] == sigs[1]
    if not agree and not (actions[0] == actions[1] and actions[0] in ("clarify", "out_of_scope")):
        extra = attempt(question, meter, 0.7)
        tries.append(extra)
        actions.append(extra.get("action"))
        if extra.get("action") == "sql":
            sql_tries.append(extra)
            sigs.append(signature(extra["rows"]))

    top_action, n_action = Counter(actions).most_common(1)[0]
    if top_action == "clarify" and n_action >= 2:
        t = next(t for t in tries if t.get("action") == "clarify")
        return Result(question, "clarify", answer=t.get("question", "Can you clarify?"),
                      options=t.get("options", []), checks=[f"{n_action} of {len(tries)} attempts asked for clarification"])
    if top_action == "out_of_scope" and n_action >= 2:
        t = next(t for t in tries if t.get("action") == "out_of_scope")
        return Result(question, "out_of_scope", answer="I don't know. " + t.get("reason", "The data can't answer this."),
                      checks=[f"{n_action} of {len(tries)} attempts said the data can't answer this"])

    if sigs:
        best_sig, votes = Counter(sigs).most_common(1)[0]
    else:
        best_sig, votes = None, 0
    if votes < 2:
        reasons = "; ".join(t.get("reason", "") for t in tries if t.get("action") == "error")
        return Result(question, "low", answer="I don't know. My attempts gave different answers, so I won't guess. "
                      "Try rephrasing, or name the part, warehouse or measure you mean.",
                      sql=sql_tries[0]["sql"] if sql_tries else "",
                      checks=[f"Attempts did not agree ({len(tries)} tried)"] + ([reasons] if reasons else []))

    chosen = next(t for t, s in zip(sql_tries, sigs) if s == best_sig)
    status = "high" if votes == len(tries) else "medium"
    checks = [f"{votes} of {len(tries)} independent attempts returned the same result",
              "Read-only database; only approved tables; single SELECT"]
    if chosen.get("repaired"):
        checks.append("First SQL failed and was repaired once")
        status = "medium"
    if not chosen["rows"]:
        checks.append("Query returned no rows: check the part or warehouse name")
        status = "medium"
    res = Result(question, status, sql=chosen["sql"], columns=chosen["columns"], rows=chosen["rows"],
                 assumption=chosen.get("assumption", ""), checks=checks)
    if chosen["rows"]:
        text, unsupported = summarize(question, chosen["columns"], chosen["rows"], meter)
        if unsupported:
            res.answer = "Here is the result from the database (summary skipped: it contained numbers not in the data)."
            res.checks.append(f"Summary check failed: {unsupported} not in result rows")
        else:
            res.answer = text
            res.checks.append("Every number in the summary appears in the query result")
    else:
        res.answer = "The query ran but found no matching rows."
    return res
