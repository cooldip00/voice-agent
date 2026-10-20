"""Hardship goes to a human, and code, not the LLM, makes sure of it.

Escalating on the first hardship sentence cut callers off: Smart Turn ended Arjun's turn at "I lost my job
last month.", the bot's closing line talked over "I can only pay five hundred rupees this week", and the
note for the human lost the 500. So code escalates at once (a hang-up in the pause still leaves the case
with a human), then waits for the caller to finish, adds what they said to the note, and only then closes.
"""

import asyncio
from collections.abc import Awaitable, Callable

from agent.call_state import hardship_said
from agent.tools import CallTools


class HardshipWait:
    def __init__(self, tools: CallTools, close: Callable[[], Awaitable[None]], wait_secs: float):
        self.tools = tools
        self.close = close            # says the closing line and hangs up
        self.wait_secs = wait_secs
        self.waiting = False          # escalated by code, the caller may still be talking
        self._timer: asyncio.Task | None = None

    async def heard(self, said: str) -> bool:
        """A finished caller turn (back-to-back turns already joined). True if code takes it: the LLM doesn't run."""
        state = self.tools.state
        if state.outcome is not None:
            return False
        if self.waiting:
            self.tools.update_escalation(said)   # the joined turn: the first sentence and what came after
        elif not (state.verified and hardship_said(said) and self.tools.call("escalate_to_human", {"reason": said})["ok"]):
            return False
        self.waiting = True
        self._start()
        return True

    def caller_started(self) -> None:
        """They're talking again: don't close over them."""
        if self.waiting:
            self._stop()

    def caller_stopped(self) -> None:
        """They paused. If no words come of it (a cough), the call still closes after the wait."""
        if self.waiting:
            self._start()

    def cancel(self) -> None:
        """The caller hung up."""
        self.waiting = False
        self._stop()

    def _start(self) -> None:
        self._stop()
        self._timer = asyncio.create_task(self._close_after_wait())

    def _stop(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None

    async def _close_after_wait(self) -> None:
        await asyncio.sleep(self.wait_secs)
        self.waiting = False
        self._timer = None
        await self.close()
