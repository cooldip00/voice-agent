"""Hardship goes to a human at once, but the call closes only after the caller has finished."""

import asyncio

from agent.hardship import HardshipWait
from agent.tools import TOOL_NAMES

WAIT = 0.05
LOST_JOB = "I lost my job last month."
CAN_PAY = "I lost my job last month. I can only pay five hundred rupees this week."


def waiting(tools):
    closed = []

    async def close():
        closed.append(tools.state.escalation)

    return HardshipWait(tools, close, WAIT), closed


def notes(conn):
    return [r[0] for r in conn.execute("SELECT reason FROM escalations")]


def test_escalates_at_once_and_closes_after_the_wait(verified, conn):
    async def call():
        hardship, closed = waiting(verified)
        assert await hardship.heard(LOST_JOB)
        assert notes(conn) == [LOST_JOB]          # a hang-up now still leaves the case with a human
        assert closed == []
        await asyncio.sleep(WAIT * 3)
        return closed

    assert asyncio.run(call()) == [LOST_JOB]


def test_what_the_caller_adds_goes_in_the_note(verified, conn):
    # The bench bug: Smart Turn ended the turn at "last month." and the human never heard the 500.
    async def call():
        hardship, closed = waiting(verified)
        await hardship.heard(LOST_JOB)
        hardship.caller_started()
        await asyncio.sleep(WAIT * 3)             # still talking: no close over them
        assert closed == []
        hardship.caller_stopped()
        assert await hardship.heard(CAN_PAY)      # the joined turn
        await asyncio.sleep(WAIT * 3)
        return closed

    assert asyncio.run(call()) == [CAN_PAY]
    assert notes(conn) == [CAN_PAY]


def test_a_cough_still_closes(verified):
    async def call():
        hardship, closed = waiting(verified)
        await hardship.heard(LOST_JOB)
        hardship.caller_started()
        hardship.caller_stopped()                 # no words came of it
        await asyncio.sleep(WAIT * 3)
        return closed

    assert asyncio.run(call()) == [LOST_JOB]


def test_hang_up_cancels_the_close(verified, conn):
    async def call():
        hardship, closed = waiting(verified)
        await hardship.heard(LOST_JOB)
        hardship.cancel()
        await asyncio.sleep(WAIT * 3)
        return closed

    assert asyncio.run(call()) == []
    assert notes(conn) == [LOST_JOB]


def test_left_to_the_llm(tools, verified, conn):
    async def heard(t, said):
        hardship, closed = waiting(t)
        took = await hardship.heard(said)
        await asyncio.sleep(WAIT * 3)
        return took, closed

    assert asyncio.run(heard(verified, "I can pay on Friday.")) == (False, [])
    verified.state.outcome = "promise"
    assert asyncio.run(heard(verified, LOST_JOB)) == (False, [])
    assert notes(conn) == []


def test_not_before_identity(tools, conn):
    async def call():
        hardship, closed = waiting(tools)
        return await hardship.heard(LOST_JOB)

    assert asyncio.run(call()) is False
    assert notes(conn) == []


def test_update_escalation_is_code_only(tools, verified, conn):
    assert "update_escalation" not in TOOL_NAMES
    verified.update_escalation("can pay 500")       # nothing escalated yet: no-op
    assert verified.state.escalation is None and notes(conn) == []
    verified.escalate_to_human("lost job")
    verified.update_escalation("  ")                # empty: keeps the note
    assert notes(conn) == ["lost job"]
