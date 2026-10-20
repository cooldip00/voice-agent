"""The fake bank: a small SQLite database of customers, loans, payments, and what calls recorded.

Amounts are whole rupees (int). Dates are ISO strings ("2026-10-20"), which SQLite sorts correctly.
"""

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE customers (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    phone         TEXT NOT NULL UNIQUE,
    date_of_birth TEXT NOT NULL
);
CREATE TABLE loans (
    id          TEXT PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    product     TEXT NOT NULL,
    emi         INTEGER NOT NULL,
    amount_due  INTEGER NOT NULL,   -- overdue now: one EMI, or more if several were missed
    due_date    TEXT NOT NULL       -- the oldest unpaid EMI
);
CREATE TABLE payments (
    id      INTEGER PRIMARY KEY,
    loan_id TEXT NOT NULL REFERENCES loans(id),
    paid_on TEXT NOT NULL,
    amount  INTEGER NOT NULL
);
CREATE TABLE promises (
    id         INTEGER PRIMARY KEY,
    loan_id    TEXT NOT NULL REFERENCES loans(id),
    call_id    TEXT NOT NULL,
    amount     INTEGER NOT NULL,
    pay_by     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE callbacks (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    call_id     TEXT NOT NULL,
    call_at     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE disputes (
    id         INTEGER PRIMARY KEY,
    loan_id    TEXT NOT NULL REFERENCES loans(id),
    call_id    TEXT NOT NULL,
    kind       TEXT NOT NULL,
    reason     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE escalations (
    id         INTEGER PRIMARY KEY,
    loan_id    TEXT NOT NULL REFERENCES loans(id),
    call_id    TEXT NOT NULL,
    reason     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

# One customer per test-bench scenario (step 6). All names and numbers are made up.
# (id, name, phone, date of birth, loan id, product, EMI, EMIs overdue, days overdue, scenario)
CUSTOMERS = [
    (1, "Rahul Sharma", "+91 90000 00001", "1990-03-05", "PL-1001", "personal loan", 4500, 1, 14, "happy path"),
    (2, "Priya Nair", "+91 90000 00002", "1988-11-21", "TW-1002", "two-wheeler loan", 3200, 1, 10, "says she already paid"),
    (3, "Arjun Mehta", "+91 90000 00003", "1985-07-14", "PL-1003", "personal loan", 6800, 1, 20, "lost his job"),
    (4, "Sneha Reddy", "+91 90000 00004", "1993-01-30", "CD-1004", "consumer durable loan", 2100, 2, 35, "disputes the amount"),
    (5, "Vikram Singh", "+91 90000 00005", "1979-09-09", "PL-1005", "personal loan", 5600, 1, 7, "asks to be called later"),
    (6, "Anjali Gupta", "+91 90000 00006", "1996-05-18", "TW-1006", "two-wheeler loan", 2900, 1, 12, "someone else answers"),
]


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create(path: Path, today: date) -> sqlite3.Connection:
    """Make a fresh bank at `path` (replacing any old one). Due dates are relative to `today`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    conn = connect(path)
    conn.executescript(SCHEMA)
    for cid, name, phone, dob, loan_id, product, emi, missed, days_overdue, _ in CUSTOMERS:
        due = today - timedelta(days=days_overdue)
        conn.execute("INSERT INTO customers VALUES (?, ?, ?, ?)", (cid, name, phone, dob))
        conn.execute(
            "INSERT INTO loans VALUES (?, ?, ?, ?, ?, ?)",
            (loan_id, cid, product, emi, emi * missed, due.isoformat()),
        )
        # three EMIs paid on time before the first missed one, a month apart
        for k in range(1, 4):
            paid_on = due - timedelta(days=30 * k)
            conn.execute(
                "INSERT INTO payments (loan_id, paid_on, amount) VALUES (?, ?, ?)",
                (loan_id, paid_on.isoformat(), emi),
            )
    conn.commit()
    return conn


def get_customer(conn: sqlite3.Connection, customer_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
    return dict(row) if row else None


def get_loan(conn: sqlite3.Connection, customer_id: int) -> dict | None:
    """The customer's loan. The fake bank has one loan per customer."""
    row = conn.execute("SELECT * FROM loans WHERE customer_id = ?", (customer_id,)).fetchone()
    return dict(row) if row else None


def last_payment(conn: sqlite3.Connection, loan_id: str) -> dict | None:
    row = conn.execute(
        "SELECT paid_on, amount FROM payments WHERE loan_id = ? ORDER BY paid_on DESC LIMIT 1", (loan_id,)
    ).fetchone()
    return dict(row) if row else None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def add_promise(conn: sqlite3.Connection, loan_id: str, call_id: str, amount: int, pay_by: date) -> None:
    conn.execute(
        "INSERT INTO promises (loan_id, call_id, amount, pay_by, created_at) VALUES (?, ?, ?, ?, ?)",
        (loan_id, call_id, amount, pay_by.isoformat(), _now()),
    )
    conn.commit()


def add_callback(conn: sqlite3.Connection, customer_id: int, call_id: str, call_at: datetime) -> None:
    conn.execute(
        "INSERT INTO callbacks (customer_id, call_id, call_at, created_at) VALUES (?, ?, ?, ?)",
        (customer_id, call_id, call_at.isoformat(timespec="minutes"), _now()),
    )
    conn.commit()


def add_dispute(conn: sqlite3.Connection, loan_id: str, call_id: str, kind: str, reason: str) -> None:
    conn.execute(
        "INSERT INTO disputes (loan_id, call_id, kind, reason, created_at) VALUES (?, ?, ?, ?, ?)",
        (loan_id, call_id, kind, reason, _now()),
    )
    conn.commit()


def add_escalation(conn: sqlite3.Connection, loan_id: str, call_id: str, reason: str) -> None:
    conn.execute(
        "INSERT INTO escalations (loan_id, call_id, reason, created_at) VALUES (?, ?, ?, ?)",
        (loan_id, call_id, reason, _now()),
    )
    conn.commit()


def update_escalation(conn: sqlite3.Connection, call_id: str, reason: str) -> None:
    conn.execute("UPDATE escalations SET reason = ? WHERE call_id = ?", (reason, call_id))
    conn.commit()
