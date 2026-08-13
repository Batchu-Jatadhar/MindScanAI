"""AI companion: a listening space with guardrails.

Two deliberate design choices:

1. Risk detection is **deterministic**, not delegated to the model. A regex
   pass over the user's own words decides whether helplines are surfaced, so
   that path cannot be talked around, jailbroken, or silently missed by a
   model having an off moment. The LLM's tone adapts on top of that signal —
   it never gates it.

2. The companion is explicitly *not* a therapist. It reflects, asks, and
   normalises; it does not diagnose, plan treatment, or assess risk itself.
"""

import logging
import re

import httpx

from backend.config import settings
from backend.core.llm.groq_client import CHAT_URL, JSON_FALLBACK_MODEL
from backend.features.crisis_detector import HELPLINES

log = logging.getLogger(__name__)

MAX_HISTORY = 12  # keeps latency and token spend bounded

# Phrases that should surface support resources regardless of what the model
# replies. Deliberately broad — a false positive costs a helpline card the user
# can ignore; a false negative costs far more.
RISK_PATTERNS = [
    # "kill myself" always counts; bare "killing me" does not, because
    # "this deadline is killing me" is far too common an idiom. Firing on it
    # would put a helpline card in front of people constantly and train them
    # to dismiss it — which costs exactly when it matters.
    r"\bkill(ing)?\s+myself\b",
    r"\bkill\s+me\b",
    r"\bend(ing)?\s+(it|my life)\b",
    r"\btake\s+my\s+own\s+life\b",
    r"\bsuicid(e|al)\b",
    r"\bself[-\s]?harm\b",
    r"\bcut(ting)?\s+myself\b",
    r"\bhurt(ing)?\s+myself\b",
    r"\bdon'?t\s+want\s+to\s+(live|be here|wake up)\b",
    r"\bno\s+(reason|point)\s+(to|in)\s+living\b",
    r"\bbetter\s+off\s+(dead|without me)\b",
    r"\bwant\s+to\s+die\b",
    r"\boverdose\b",
]
_RISK_RE = re.compile("|".join(RISK_PATTERNS), re.IGNORECASE)


def detect_risk(text: str) -> bool:
    return bool(_RISK_RE.search(text or ""))


BASE_PROMPT = """You are the MindScan companion — a warm, steady presence someone \
can talk to about their day. You are NOT a therapist, doctor, or diagnostician.

How to be useful:
- Listen first. Reflect back what you actually heard, in their words, before \
offering anything.
- Ask one open question at a time. Never interrogate.
- Normalise without minimising. "That sounds exhausting" beats "at least...".
- Keep replies short — two to four sentences usually. This is a conversation, \
not an essay.
- Offer a concrete, small suggestion only when it fits, and never more than one \
at a time.

Hard limits:
- Never diagnose or name a condition someone might have.
- Never give medication advice, dosages, or tell someone to change medication.
- Never promise outcomes or say things will definitely improve.
- Don't claim to be human, and don't pretend to remember past sessions you \
weren't given.
- If they describe something medical and urgent, say plainly that you can't \
assess it and point them to a professional.

Write plainly, in second person, no clinical jargon, no bullet lists, no \
emoji."""

CRISIS_ADDENDUM = """

IMPORTANT — this person has said something suggesting they may be at risk of \
harming themselves. For this reply:
- Take it seriously and say so directly. Do not change the subject or move on.
- Do not panic, lecture, or moralise. Stay calm and warm.
- Ask directly and without euphemism whether they are safe right now.
- Tell them helpline details are shown alongside this message and encourage \
reaching out to one, or to someone they trust.
- Do not attempt to assess how serious the risk is, and do not try to talk them \
out of how they feel."""


def _context_block(assessment: dict | None) -> str:
    if not assessment:
        return "No recent screening is available for this person."
    scores = assessment.get("scores") or {}
    parts = [f"Their most recent screening status was {assessment.get('status', 'unknown').replace('_', ' ')}."]
    if scores:
        parts.append(
            "Scores: depression {d}, anxiety {a}, stress {s} (each out of 34/24/39).".format(
                d=scores.get("depression"), a=scores.get("anxiety"), s=scores.get("stress")
            )
        )
    parts.append(
        "Use this only as quiet background. Do not open by reciting it, and do not "
        "treat it as a diagnosis."
    )
    return " ".join(parts)


def chat(messages: list[dict], assessment: dict | None = None) -> dict:
    """One companion turn.

    `messages` is the prior conversation as {role, content}, oldest first, with
    the newest user message last.
    """
    if not settings.groq_llm_ready:
        return {
            "available": False,
            "reason": "The companion needs a Groq API key (set GROQ_LLM_API_KEY and ENABLE_GROQ=true).",
        }

    history = [m for m in messages if m.get("role") in ("user", "assistant") and m.get("content")]
    if not history or history[-1]["role"] != "user":
        return {"available": False, "reason": "No user message to respond to."}

    latest = history[-1]["content"]
    at_risk = detect_risk(latest)

    system = BASE_PROMPT + ("\n\n" + _context_block(assessment))
    if at_risk:
        system += CRISIS_ADDENDUM

    payload_messages = [{"role": "system", "content": system}] + [
        {"role": m["role"], "content": m["content"][:2000]} for m in history[-MAX_HISTORY:]
    ]

    reply = None
    last_reason = "Companion unavailable."
    for model in (settings.groq_llm_model, JSON_FALLBACK_MODEL):
        try:
            r = httpx.post(
                CHAT_URL,
                headers={
                    "Authorization": f"Bearer {settings.groq_llm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": payload_messages,
                    "temperature": 0.75,
                    "max_tokens": 400,
                },
                timeout=settings.groq_timeout_seconds,
            )
            if r.status_code >= 400:
                log.warning("Companion HTTP %s on %s: %s", r.status_code, model, r.text[:300])
                last_reason = f"Companion unavailable (HTTP {r.status_code})."
                continue
            reply = (r.json()["choices"][0]["message"]["content"] or "").strip()
            break
        except Exception as exc:
            log.warning("Companion failed on %s: %s", model, exc)
            last_reason = f"Companion unavailable ({type(exc).__name__})."

    if not reply:
        # Even when the model is unreachable, a risk disclosure must still get
        # a response and resources rather than a bare error.
        if at_risk:
            return {
                "available": True,
                "reply": (
                    "I can't reach my language service right now, but what you just said matters "
                    "and I don't want to leave it there. Please consider calling one of the "
                    "numbers shown here — someone trained to listen will pick up."
                ),
                "at_risk": True,
                "resources": HELPLINES,
                "degraded": True,
            }
        return {"available": False, "reason": last_reason}

    return {
        "available": True,
        "reply": reply,
        "at_risk": at_risk,
        "resources": HELPLINES if at_risk else [],
        "disclaimer": "A supportive conversation, not therapy or medical advice.",
    }
