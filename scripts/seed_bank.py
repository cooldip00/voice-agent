"""Create the fake bank at data/bank.db (replacing any old one), with due dates relative to today.

Run:  uv run python -m scripts.seed_bank
"""

from datetime import date

from agent import bank, config


def main():
    today = date.today()
    conn = bank.create(config.BANK_DB, today)
    print(f"created {config.BANK_DB.relative_to(config.ROOT)} (dates relative to {today})\n")
    print(f"{'id':<3} {'name':<14} {'date of birth':<14} {'loan':<8} {'due':>6}  {'overdue':>8}  scenario")
    for cid, *_, scenario in bank.CUSTOMERS:
        c, loan = bank.get_customer(conn, cid), bank.get_loan(conn, cid)
        days = (today - date.fromisoformat(loan["due_date"])).days
        print(f"{cid:<3} {c['name']:<14} {c['date_of_birth']:<14} {loan['id']:<8} {loan['amount_due']:>6}  "
              f"{days:>5} d   {scenario}")
    conn.close()


if __name__ == "__main__":
    main()
