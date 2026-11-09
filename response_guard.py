"""Prevent unsupported claims that the robot just completed navigation.

A language model response is not robot evidence. Only the current command's
navigation outcome and remembered target verification can authorize ARRIVED.
"""
import re

_CURRENT_ARRIVAL_CLAIM = re.compile(
    r"\b(?:i(?:['\u2019]ve|\s+have)?\s+(?:successfully\s+)?(?:arrived|reached)\b"
    r"|navigation\s+(?:succeeded|completed\s+successfully)\b"
    r"|(?:status|outcome)\s*[:=]\s*\**\s*arrived\b)",
    re.IGNORECASE,
)


def guard_unverified_arrival(response, *, nav_outcome, identity_confirmed):
    """Return (reply, blocked); never promote a text claim into ARRIVED."""
    if not isinstance(response, str) or not _CURRENT_ARRIVAL_CLAIM.search(response):
        return response, False
    if nav_outcome == "ARRIVED" and identity_confirmed:
        return response, False
    result = ("Arrival is not verified for this command. No navigation episode "
              "in this command ended with a confirmed ARRIVED result. "
              "The robot's position and target arrival must be checked by "
              "navigate_to_object; a conversational reply is not evidence.")
    if nav_outcome and nav_outcome not in ("RUNNING", "ARRIVED"):
        result += f" Recorded navigation outcome: {nav_outcome}."
    return result, True
