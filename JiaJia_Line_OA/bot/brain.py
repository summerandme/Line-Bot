"""JiaJia reply brain: turns one customer message into the messages the bot sends back.

Gemini reads the knowledge base and classifies + answers the message. Everything
that must be exact (the sign-up link, the "can't answer" hand-off, admin hours)
comes from fixed templates in this file, and every number in Gemini's answer is
checked against the knowledge base before it is sent.
"""

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from google import genai
from google.genai import types

from bot.store import make_store

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE = (ROOT / "knowledge.md").read_text(encoding="utf-8")

# Tried in order: if one model is overloaded or times out, the next one answers.
MODELS = [m.strip() for m in os.environ.get(
    "GEMINI_MODELS", "gemini-3.5-flash,gemini-3.8-flash,gemini-3.1-flash-lite"
).split(",") if m.strip()]
MODEL = MODELS[0]

CHINA_TZ = timezone(timedelta(hours=8))
ADMIN_OPEN_HOUR = 9    # 09:00 China time
ADMIN_CLOSE_HOUR = 18  # 18:00 China time
ADMIN_PAUSE_HOURS = 12  # bot stays quiet in a chat after the customer asks for a human
HISTORY_TURNS = 10      # remembered messages per customer (user + bot)
HISTORY_TTL_SECONDS = 24 * 3600

FORM_URL = "linktr.ee/jiajia.roomhunt"

# ---------------------------------------------------------------- templates

CLOSING = {
    "th": (
        "อยากเริ่มหาห้องกับ JiaJia ใช่ไหมคะ 🏠✨\n"
        f"👉 กรอกฟอร์มสั้น ๆ ที่นี่ได้เลย: {FORM_URL}\n"
        "หลังกรอกเสร็จ ทีมจะติดต่อกลับทาง WeChat เพื่อส่งห้องให้เลือกค่ะ 📲"
    ),
    "en": (
        "Ready to start your room search with JiaJia? 🏠✨\n"
        f"👉 Fill in this short form: {FORM_URL}\n"
        "Once you submit it, our team will reach out on WeChat to send you room options 📲"
    ),
}

HOURS_LINE = {
    "th": {
        "open": "แอดมินจะรีบตอบกลับโดยเร็วที่สุด (เวลาทำการ 09:00–18:00 น. เวลาจีน / 08:00–17:00 น. เวลาไทย)",
        "closed": "ตอนนี้อยู่นอกเวลาทำการ แอดมินจะตอบกลับในเวลาทำการถัดไป 09:00 น. เวลาจีน (08:00 น. เวลาไทย) ค่ะ",
    },
    "en": {
        "open": "Our admin will get back to you as soon as possible (9:00–18:00 China time, GMT+8).",
        "closed": "We're outside office hours right now — our admin will reply from 9:00 China time (GMT+8).",
    },
}

CANNOT_ANSWER = {
    "th": (
        "เรื่องนี้ขอให้แอดมินตอบให้ตรงกับเคสของคุณนะคะ 😊 {hours}\n\n"
        "ระหว่างนี้ขอเล่าไว้ก่อนว่า ปกติทีมจะเริ่มหาและพาดูห้องจริงช่วง 1–1.5 เดือนก่อนวันที่คุณมาถึง "
        "ถ้ายังเหลือเวลามากกว่านั้น กรอกฟอร์มไว้ก่อนได้เลยค่ะ ทีมจะคัดห้องที่เหมาะกับงบและทำเลให้ "
        "และส่งตัวอย่างห้องให้ดูว่าคุณชอบแนวไหน 🏠✨\n"
        f"👉 {FORM_URL}\n"
        "หลังกรอกเสร็จ ทีมจะติดต่อกลับทาง WeChat เพื่อส่งห้องให้เลือกค่ะ 📲"
    ),
    "en": (
        "Good question! Let our admin check this for your specific case 😊 {hours}\n\n"
        "In the meantime: we usually start searching and viewing real rooms about 1–1.5 months before you arrive. "
        "If you're arriving later than that, just fill in this short form for now — we'll shortlist rooms that fit "
        "your budget and area and send you some sample rooms, so we know your style 🏠✨\n"
        f"👉 {FORM_URL}\n"
        "Once you submit it, our team will reach out on WeChat to send you room options 📲"
    ),
}

TALK_TO_ADMIN = {
    "th": "ได้เลยค่ะ ส่งเรื่องให้แอดมินแล้ว 🙏 {hours}\nระหว่างนี้บอทจะไม่ตอบแทรกในแชทนี้นะคะ",
    "en": "Sure! I've passed this to our admin 🙏 {hours}\nI'll stay quiet in this chat so you can talk to them directly.",
}

NON_TEXT = {
    "th": "ขอบคุณค่ะ 😊 ตอนนี้บอทอ่านได้เฉพาะข้อความตัวอักษรนะคะ พิมพ์คำถามมาได้เลย เช่น \"แพ็กเกจราคาเท่าไหร่\"",
    "en": "Thanks! 😊 I can only read text for now — just type your question, e.g. \"How much are your packages?\"",
}

# ---------------------------------------------------------------- prompt

SYSTEM_PROMPT = f"""You are the LINE chat assistant for JiaJia 家家, a small founder-led team that helps international \
students and workers find rooms in China. You answer customers 24/7.

Voice: warm, friendly, peer-to-peer — like a local friend who has been through this. Honest and reassuring, never \
salesy ("amazing deal!"). A few emojis are fine. Keep replies short enough to read on a phone (usually 2–6 short \
lines). Write plain text only: no markdown, no asterisks, no tables, no headings.

Language: reply in the customer's language. Thai customers: write Thai and end sentences politely with ค่ะ/นะคะ. \
Everyone else: English.
Currency: in Thai replies give package prices in THB with RMB in brackets, e.g. "6,700 บาท (1,340 RMB)". In English \
replies give RMB. Add-ons only have RMB prices — give them in RMB in both languages.

Facts: use ONLY the knowledge base below. Copy prices exactly as written. Never estimate, convert currencies, \
calculate new amounts, or invent rent prices, room availability, discounts, dates or policies. If the knowledge base \
doesn't contain the answer (rent for a specific area, Survival Kit or Wi-Fi prices, airport pick-up price, \
discounts, anything about a specific booking), choose intent "cannot_answer" — the system then sends the hand-off \
message, so leave "reply" as an empty string in that case.

Pick exactly one intent:
- "answer": you can fully answer from the knowledge base (prices, what's included, fees, steps, FAQ, greetings, thanks).
- "interested": the customer shows buying intent — wants to book or start, asks how to sign up / pay / begin, says \
which package they want, or gives move-in details. Write a short, warm reply (1–3 lines) about their choice; the \
system appends the sign-up form link and the WeChat note right after it, so do NOT mention the form, the link or \
WeChat yourself.
- "cannot_answer": the knowledge base doesn't cover it (see above). reply = "".
- "talk_to_admin": the customer explicitly asks to talk to a human / admin / staff. reply = "".

Set "language" to "th" if the customer wrote Thai, otherwise "en".

<knowledge_base>
{KNOWLEDGE}
</knowledge_base>"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["answer", "interested", "cannot_answer", "talk_to_admin"]},
        "language": {"type": "string", "enum": ["th", "en"]},
        "reply": {"type": "string"},
    },
    "required": ["intent", "language", "reply"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------- helpers

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_THAI_RE = re.compile(r"[฀-๿]")


def _numbers(text: str) -> set:
    out = set()
    for raw in _NUM_RE.findall(text):
        n = raw.replace(",", "").rstrip(".")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        n = n.lstrip("0") or "0"
        out.add(n)
    return out


_KNOWN_NUMBERS = _numbers(KNOWLEDGE) | _numbers(json.dumps([CLOSING, HOURS_LINE, CANNOT_ANSWER], ensure_ascii=False))


def unknown_numbers(reply: str, customer_text: str) -> set:
    """Numbers in the reply that are not in the knowledge base or the customer's own message.

    Small whole numbers (counts like "2 people", "3 months") are allowed.
    """
    allowed = _KNOWN_NUMBERS | _numbers(customer_text)
    bad = set()
    for n in _numbers(reply):
        if n in allowed:
            continue
        if n.isdigit() and int(n) <= 12:
            continue
        bad.add(n)
    return bad


def guess_language(text: str) -> str:
    return "th" if _THAI_RE.search(text or "") else "en"


def admin_hours_line(lang: str, now: datetime = None) -> str:
    now = (now or datetime.now(timezone.utc)).astimezone(CHINA_TZ)
    is_open = ADMIN_OPEN_HOUR <= now.hour < ADMIN_CLOSE_HOUR
    return HOURS_LINE[lang]["open" if is_open else "closed"]


# ---------------------------------------------------------------- brain


class Brain:
    def __init__(self, client: genai.Client = None, store=None):
        self.client = client or genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
            http_options=types.HttpOptions(timeout=12_000),  # ms per model attempt
        )
        self.store = store or make_store()
        self.last_model = None

    # -- state (kept in the store so it survives between serverless calls)

    def is_paused(self, user_id: str) -> bool:
        return bool(self.store.get(f"pause:{user_id}"))

    def resume(self, user_id: str) -> None:
        self.store.delete(f"pause:{user_id}")

    def _get_history(self, user_id: str) -> list:
        return list(self.store.get(f"history:{user_id}") or [])

    def _remember(self, user_id: str, user_text: str, bot_text: str) -> None:
        msgs = self._get_history(user_id) + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": bot_text},
        ]
        self.store.set(f"history:{user_id}", msgs[-HISTORY_TURNS:], HISTORY_TTL_SECONDS)

    # -- model call

    def _ask_model(self, user_id: str, text: str) -> dict:
        history = self._get_history(user_id) + [{"role": "user", "content": text}]
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in history
        ]
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_json_schema=OUTPUT_SCHEMA,
            temperature=0.3,
            max_output_tokens=8192,
        )
        errors = []
        for model in MODELS:
            try:
                response = self.client.models.generate_content(model=model, contents=contents, config=config)
                raw = response.text
                if not raw:
                    reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
                    raise ValueError(f"empty response: {reason}")
                result = json.loads(raw)
                self.last_model = model
                return result
            except Exception as exc:  # overloaded / timeout / bad output: try the next model
                errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:120]}")
        raise RuntimeError(" | ".join(errors))

    # -- public

    def reply(self, user_id: str, text: str, now: datetime = None) -> dict:
        """Return {"messages": [str, ...], "intent": str, "note": str}. Empty messages = stay quiet."""
        text = (text or "").strip()
        if self.is_paused(user_id):
            return {"messages": [], "intent": "paused", "note": "chat handed to admin"}

        note = ""
        try:
            result = self._ask_model(user_id, text)
            intent, lang, body = result["intent"], result["language"], result["reply"].strip()
        except Exception as exc:  # API down, bad JSON, refusal: never leave the customer hanging
            intent, lang, body = "cannot_answer", guess_language(text), ""
            note = f"fallback after error: {type(exc).__name__}: {exc}"

        if lang not in ("th", "en"):
            lang = guess_language(text)

        if intent in ("answer", "interested"):
            bad = unknown_numbers(body, text)
            if bad or not body:
                note = f"blocked reply (unknown numbers {sorted(bad)})" if bad else "empty reply"
                intent, body = "cannot_answer", ""

        hours = admin_hours_line(lang, now)
        if intent == "answer":
            messages = [body]
        elif intent == "interested":
            messages = [body, CLOSING[lang]]
        elif intent == "talk_to_admin":
            self.store.set(f"pause:{user_id}", True, ADMIN_PAUSE_HOURS * 3600)
            messages = [TALK_TO_ADMIN[lang].format(hours=hours)]
        else:
            intent = "cannot_answer"
            messages = [CANNOT_ANSWER[lang].format(hours=hours)]

        self._remember(user_id, text, "\n\n".join(messages))
        return {"messages": messages, "intent": intent, "note": note}

    def non_text_reply(self) -> list:
        """Stickers, photos, voice: we can't tell the language, so answer in both."""
        return [NON_TEXT["th"] + "\n\n" + NON_TEXT["en"]]
