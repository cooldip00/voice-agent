from datetime import timedelta

from agent import bank
from tests.conftest import TODAY


def test_every_customer_has_one_overdue_loan(conn):
    assert len(bank.CUSTOMERS) == 6
    for cid, *_ in bank.CUSTOMERS:
        loan = bank.get_loan(conn, cid)
        assert loan["amount_due"] >= loan["emi"] > 0
        assert loan["due_date"] < TODAY.isoformat()


def test_due_dates_follow_today(conn):
    assert bank.get_loan(conn, 1)["due_date"] == (TODAY - timedelta(days=14)).isoformat()


def test_two_missed_emis_double_the_amount_due(conn):
    loan = bank.get_loan(conn, 4)
    assert loan["amount_due"] == 2 * loan["emi"]


def test_last_payment_is_the_month_before_the_missed_emi(conn):
    loan = bank.get_loan(conn, 1)
    paid = bank.last_payment(conn, loan["id"])
    assert paid["amount"] == loan["emi"]
    assert paid["paid_on"] == (TODAY - timedelta(days=14 + 30)).isoformat()


def test_unknown_customer(conn):
    assert bank.get_customer(conn, 99) is None
    assert bank.get_loan(conn, 99) is None


def test_create_replaces_old_bank(tmp_path):
    path = tmp_path / "bank.db"
    c = bank.create(path, TODAY)
    bank.add_dispute(c, "PL-1001", "call-1", "other", "test")
    c.close()
    c = bank.create(path, TODAY)
    assert c.execute("SELECT COUNT(*) FROM disputes").fetchone()[0] == 0
    c.close()
