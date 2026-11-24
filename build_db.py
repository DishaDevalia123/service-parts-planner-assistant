import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
XLSX = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "case.xlsx"
DB = ROOT / "planner.db"
OLD_ORDER_DAYS = 90
BULK_QTY = 10


def load():
    o = pd.read_excel(XLSX, sheet_name="Orders").dropna(subset=["Order ID"])
    o = o.rename(columns={
        "Order ID": "order_id",
        "Order Create Date": "order_date",
        "Requested Qty": "qty",
        "Fulfiller Warehouse": "warehouse_id",
        "Requested Part Number": "part",
        "Parent WarehouseID": "parent_warehouse_id",
        "Compatible Part Number": "compatible_part",
    })
    o["order_id"] = o["order_id"].astype(int)
    o["warehouse_id"] = o["warehouse_id"].astype(int)
    o["parent_warehouse_id"] = o["parent_warehouse_id"].astype(int)
    o["qty"] = o["qty"].astype(int)
    o["order_date"] = pd.to_datetime(o["order_date"], format="%m/%d/%Y")

    inv = pd.read_excel(XLSX, sheet_name="Inventory").rename(columns={
        "Part Number": "part", "Warehouse ID": "warehouse_id", "On Hand Qty": "on_hand_qty"})
    inv["warehouse_id"] = inv["warehouse_id"].astype(int)
    return o, inv


def build_tables(o, inv):
    key = ["order_id", "order_date", "qty", "warehouse_id", "part", "parent_warehouse_id"]
    lines = o[key].drop_duplicates().sort_values(["order_date", "order_id"]).reset_index(drop=True)
    lines.insert(0, "line_id", range(1, len(lines) + 1))
    as_of = o["order_date"].max()
    lines["age_days"] = (as_of - lines["order_date"]).dt.days
    lines["is_old"] = (lines["age_days"] > OLD_ORDER_DAYS).astype(int)
    lines["is_bulk"] = (lines["qty"] >= BULK_QTY).astype(int)

    subs = o.merge(lines[["line_id"] + key], on=key)[["line_id", "compatible_part"]]
    subs = subs[subs["compatible_part"].notna()].drop_duplicates()
    subs.insert(1, "priority", subs.groupby("line_id").cumcount() + 1)
    return lines, subs, as_of


def run_fill_model(lines, subs, inv):
    stock = {(r.part, r.warehouse_id): int(r.on_hand_qty) for r in inv.itertuples()}
    sub_map = subs.sort_values("priority").groupby("line_id")["compatible_part"].apply(list).to_dict()
    fills = []
    for ln in lines.itertuples():
        need = ln.qty
        alts = [p for p in sub_map.get(ln.line_id, []) if p != ln.part]
        steps = [("own_part", ln.warehouse_id, ln.part)]
        steps += [("own_substitute", ln.warehouse_id, p) for p in alts]
        if ln.parent_warehouse_id != ln.warehouse_id:
            steps.append(("parent_part", ln.parent_warehouse_id, ln.part))
            steps += [("parent_substitute", ln.parent_warehouse_id, p) for p in alts]
        for fill_type, wh, part in steps:
            if need == 0:
                break
            take = min(need, stock.get((part, wh), 0))
            if take > 0:
                stock[(part, wh)] -= take
                need -= take
                fills.append((ln.line_id, fill_type, wh, part, take))
    fills = pd.DataFrame(fills, columns=["line_id", "fill_type", "source_warehouse_id", "source_part", "filled_qty"])
    remaining = pd.DataFrame([(p, w, q) for (p, w), q in stock.items()],
                             columns=["part", "warehouse_id", "remaining_qty"])
    return fills, remaining


VIEWS = """
CREATE VIEW line_status AS
SELECT l.line_id, l.order_id, l.order_date, l.warehouse_id, l.part, l.qty AS requested_qty,
       COALESCE(f.filled, 0) AS filled_qty,
       l.qty - COALESCE(f.filled, 0) AS short_qty,
       CASE WHEN COALESCE(f.filled, 0) = l.qty THEN 1 ELSE 0 END AS is_complete,
       l.is_bulk, l.is_old, l.age_days
FROM order_lines l
LEFT JOIN (SELECT line_id, SUM(filled_qty) AS filled FROM fills GROUP BY line_id) f
  ON f.line_id = l.line_id;

CREATE VIEW shortage_by_part_warehouse AS
SELECT part, warehouse_id, SUM(requested_qty) AS requested_qty, SUM(filled_qty) AS filled_qty,
       SUM(short_qty) AS short_qty, COUNT(*) AS lines, SUM(1 - is_complete) AS incomplete_lines
FROM line_status GROUP BY part, warehouse_id;
"""


def main():
    o, inv = load()
    lines, subs, as_of = build_tables(o, inv)
    fills, remaining = run_fill_model(lines, subs, inv)
    lines["order_date"] = lines["order_date"].dt.strftime("%Y-%m-%d")
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    lines.to_sql("order_lines", con, index=False)
    subs.to_sql("line_substitutes", con, index=False)
    inv.to_sql("inventory", con, index=False)
    fills.to_sql("fills", con, index=False)
    remaining.to_sql("remaining_inventory", con, index=False)
    pd.DataFrame([{"as_of_date": as_of.strftime("%Y-%m-%d"), "old_order_days": OLD_ORDER_DAYS,
                   "bulk_qty": BULK_QTY}]).to_sql("data_info", con, index=False)
    con.executescript(VIEWS)
    con.commit()

    q = lambda s: con.execute(s).fetchall()
    print("order lines:", q("SELECT COUNT(*), SUM(qty) FROM order_lines")[0])
    print("filled:", q("SELECT SUM(filled_qty) FROM fills")[0][0])
    print("by fill type:", q("SELECT fill_type, SUM(filled_qty) FROM fills GROUP BY 1 ORDER BY 1"))
    print("complete lines:", q("SELECT SUM(is_complete), COUNT(*) FROM line_status")[0])
    print("short by part:", q("SELECT part, SUM(short_qty) FROM line_status GROUP BY 1"))
    print("stock left:", q("SELECT SUM(remaining_qty) FROM remaining_inventory")[0][0])
    con.close()


if __name__ == "__main__":
    main()
