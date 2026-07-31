"""Strict spoken-command parsing for the stage demo."""

from __future__ import annotations

from dataclasses import dataclass
import re


ALLOWED_FRUITS = ("apple", "banana", "pear")
_BARE_FRUIT_RE = re.compile(
    r"^(apple|banana|pear|pair)[.!?]*$",
    re.IGNORECASE,
)
_FIND_RE = re.compile(
    r"^(?:(?:hey[,\s]+)?woof[,\s]+)?"
    r"(?:please[,\s]+)?find\s+(?:the\s+)?"
    r"(apple|banana|pear|pair)"
    r"(?:\s+please)?[.!?]*$",
    re.IGNORECASE,
)
_STOP_RE = re.compile(
    r"^(?:(?:hey[,\s]+)?woof[,\s]+)?"
    r"(?:stop|abort|cancel)(?:\s+(?:now|woof|mission))?[.!?]*$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class VoiceCommand:
    kind: str
    transcript: str
    target: str | None = None


def parse_voice_command(transcript: str) -> VoiceCommand | None:
    """Return an allowlisted command or ``None`` for all other speech."""

    cleaned = " ".join(transcript.strip().split())
    if not cleaned:
        return None
    match = _BARE_FRUIT_RE.fullmatch(cleaned) or _FIND_RE.fullmatch(cleaned)
    if match:
        target = match.group(1).casefold()
        if target == "pair":
            target = "pear"
        return VoiceCommand(
            kind="find",
            transcript=cleaned,
            target=target,
        )
    if _STOP_RE.fullmatch(cleaned):
        return VoiceCommand(kind="stop", transcript=cleaned)
    return None
