"""A test-bench scenario: who is called, how the caller answers, and how the call must end.

One TOML file per scenario in testbench/scenarios/. The caller doesn't read a fixed script: after each bot
turn it picks the first rule whose `when` pattern is in what it heard. That keeps its answers on topic
when the bot asks things in a different order, or twice. When it would say the same line a third time
running (the bot is going round in circles, or nothing it says matches), it gives up and hangs up, like a
real caller would.

Rules answer what a reasonable person would say to that line, nothing more: a caller tuned until the bot
passes tests the caller, not the bot.

Patterns are TOML strings: write a regex backslash twice ("confirm\\b"), or TOML reads "\b" as a backspace.

    customer = 3
    goal = "Lost his job, can pay 500: the bot hands the case to a human"

    [[reply]]
    when = "date of birth|born"                     # regex, searched in the bot's words, any case
    say = "The fourteenth of July, nineteen eighty five."

    [[reply]]
    when = "when (will|can|could) you"
    say = ["First answer.", "Second time asked."]   # a list: the next one each time, then the last repeats
    wait = 20                                       # silent this long first (the silence test)
    interrupt = "Wait, how much exactly?"           # said 1.5 s into the bot's next turn (the barge-in test)

    [expect]
    outcome = "escalated"                           # or a list of acceptable outcomes
"""

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

SCENARIO_DIR = Path(__file__).parent / "scenarios"
FALLBACK = "Sorry, could you say that again?"
GIVE_UP = "Sorry, I have to go now. Bye."
GIVE_UP_AT = 3   # about to say the same line this many times running: the caller gives up instead


@dataclass
class Reply:
    say: str
    wait: float = 0.0           # seconds of silence before saying it
    interrupt: str | None = None
    hang_up: bool = False       # the caller hangs up after saying it


@dataclass
class Rule:
    when: re.Pattern
    says: list[str]
    wait: float = 0.0
    interrupt: str | None = None
    used: int = 0


@dataclass
class Scenario:
    name: str
    customer: int
    goal: str
    rules: list[Rule]
    expect: dict
    fallback: str = FALLBACK
    max_turns: int = 14
    lines: list[str] = field(default_factory=list)   # everything the caller may say, made into audio up front
    said: list[str] = field(default_factory=list)    # the caller's answers so far

    def reply_to(self, heard: str) -> Reply:
        """The caller's answer to one bot turn. Rules are tried in file order; the first match answers."""
        reply = Reply(self.fallback)
        for rule in self.rules:
            if rule.when.search(heard):
                say = rule.says[min(rule.used, len(rule.says) - 1)]
                rule.used += 1
                # wait and interrupt happen once: a second "when can you pay?" gets a normal answer
                first = rule.used == 1
                reply = Reply(say, rule.wait if first else 0.0, rule.interrupt if first else None)
                break
        if self.said[-(GIVE_UP_AT - 1):] == [reply.say] * (GIVE_UP_AT - 1):
            reply = Reply(GIVE_UP, hang_up=True)
        self.said.append(reply.say)
        return reply


def load(path: Path) -> Scenario:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    rules = []
    for r in data.get("reply", []):
        says = r["say"] if isinstance(r["say"], list) else [r["say"]]
        rules.append(Rule(re.compile(r["when"], re.IGNORECASE), says, float(r.get("wait", 0)), r.get("interrupt")))
    fallback = data.get("fallback", FALLBACK)
    lines = [s for r in rules for s in r.says] + [r.interrupt for r in rules if r.interrupt] + [fallback, GIVE_UP]
    return Scenario(path.stem, int(data["customer"]), data.get("goal", ""), rules, data.get("expect", {}),
                    fallback, int(data.get("max_turns", 14)), lines)


def load_all(names: list[str] | None = None) -> list[Scenario]:
    paths = sorted(SCENARIO_DIR.glob("*.toml"))
    if names:
        missing = set(names) - {p.stem for p in paths}
        if missing:
            raise SystemExit(f"No scenario named {', '.join(sorted(missing))}. Have: {', '.join(p.stem for p in paths)}")
        paths = [p for p in paths if p.stem in names]
    return [load(p) for p in paths]
