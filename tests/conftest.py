"""Shared fixtures: a fresh fake bank per test and a fixed clock, so results never depend on today."""

from datetime import date, datetime

import pytest

from agent import bank
from agent.call_state import CallState
from agent.tools import CallTools

TODAY = date(2026, 10, 20)
NOW = datetime(2026, 10, 20, 11, 0)
RAHUL_DOB = "1990-03-05"   # customer 1


@pytest.fixture
def conn(tmp_path):
    c = bank.create(tmp_path / "bank.db", TODAY)
    yield c
    c.close()


@pytest.fixture
def state():
    return CallState(call_id="test-call", customer_id=1)


@pytest.fixture
def tools(state, conn):
    return CallTools(state, conn, clock=lambda: NOW)


@pytest.fixture
def verified(tools):
    assert tools.verify_identity(RAHUL_DOB)["verified"]
    return tools
