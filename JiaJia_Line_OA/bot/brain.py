"""JiaJia reply brain: turns one customer message into the messages the bot sends back.

Gemini reads the knowledge base and classifies + answers the message. Everything
that must be exact (the sign-up link, the "can't answer" hand-off, admin hours)
comes from fixed templates in this file, and every number in Gemini's answer is
checked against the knowledge base before it is sent.
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from google import genai
from google.genai import types

from bot.sheet import sheet_text
from bot.store import make_store

log = logging.getLogger("jiajia.brain")

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
HANDOFF_BURST_SECONDS = 120  # several messages in a row get one "admin will contact you", not one each
FORM_ONCE_DAYS = 60          # the sign-up form link is sent once per chat (remembered this long)
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

# Sent instead of an AI reply when an admin needs to take over. "first" the first time in a chat,
# "again" if the customer asks more, "referral" when someone recommended JiaJia to them.
HANDOFF = {
    "th": {
        "first": "เดี๋ยวทางแอดมินจะรีบติดต่อกลับนะคะ 🙏✨",
        "again": "รับเรื่องไว้แล้วนะคะ 📝 เดี๋ยวแอดมินจะรีบตอบกลับโดยเร็วค่ะ 🙏",
        "referral": "ยินดีต้อนรับค่ะ 🥰 ขอบคุณที่ได้รับการแนะนำมานะคะ รับเรื่องไว้แล้ว เดี๋ยวแอดมินจะรีบติดต่อกลับโดยเร็วค่ะ 🙏✨",
    },
    "en": {
        "first": "Our admin will get back to you shortly 🙏✨",
        "again": "Got it, noted 📝 Our admin will get back to you as soon as possible 🙏",
        "referral": "Welcome! 🥰 Thanks for coming to us through a referral. Noted, and our admin will contact you shortly 🙏✨",
    },
}

# Added under a hand-off when the team is offline.
CLOSED_NOTE = {
    "th": "🌙 ขณะนี้อยู่นอกเวลาทำการนะคะ แอดมินจะตอบกลับในเวลาทำการ 09:00–18:00 น. เวลาจีน (08:00–17:00 น. เวลาไทย) ค่ะ",
    "en": "🌙 We're outside office hours right now. Our admin will reply during office hours, 9:00–18:00 China time (GMT+8).",
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
salesy ("amazing deal!"). Use 2–3 friendly emojis in every reply (e.g. 😊 🏠 ✨ 🙏 📲 🥰). Keep replies short: \
usually 1–3 short lines, never more than 4. Write plain text only: no markdown, no asterisks, no tables, no headings.
Don't repeat yourself: read the chat history first. Never re-send a greeting, a price, a list or an explanation \
you already gave in this chat unless the customer asks for it again; answer only the new part of the question. \
Don't start every message with a greeting or a thank-you.

Language: the system tells you below which language to reply in. Follow it even if the customer's message mixes in \
English words, names or places (e.g. "fudan", "COZI"), and even if earlier replies in the chat were in English. \
Thai replies end sentences politely with ค่ะ/นะคะ.
Addressing (Thai replies): call the customer "K.<their LINE name>" exactly as the system gives it, e.g. "K.Nok". If \
no usable name is given, call them "คุณลูกค้า". Never use พี่, น้อง or other words for the customer. Use the name \
at most once per reply, usually in the first line, and not in every single message of a chat.
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
- "referral": the customer says someone recommended JiaJia or gave them this contact (e.g. "ได้คอนแทคมาจากพี่ปุ้ย", \
"my friend referred me"). reply = "" (the system welcomes them and says an admin will contact them).
If the customer asks again about something you already answered in this chat, they need a person: choose \
"cannot_answer" instead of answering the same thing again.

Set "language" to the language the system tells you to reply in.

<knowledge_base>
{KNOWLEDGE}
</knowledge_base>"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["answer", "interested", "cannot_answer", "talk_to_admin", "referral"]},
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


_KNOWN_NUMBERS = _numbers(KNOWLEDGE) | _numbers(json.dumps([CLOSING, HOURS_LINE, HANDOFF, CLOSED_NOTE], ensure_ascii=False))


def unknown_numbers(reply: str, customer_text: str, extra_known: str = "") -> set:
    """Numbers in the reply that are not in the knowledge base, the live sheet or the customer's message.

    Small whole numbers (counts like "2 people", "3 months") are allowed.
    """
    allowed = _KNOWN_NUMBERS | _numbers(customer_text) | _numbers(extra_known)
    bad = set()
    for n in _numbers(reply):
        if n in allowed:
            continue
        if n.isdigit() and int(n) <= 12:
            continue
        bad.add(n)
    return bad


def clean_name(name: str) -> str:
    """A LINE display name made safe to put in the prompt; "" if there is no real name in it."""
    name = " ".join((name or "").replace('"', "").split())[:30].strip()
    return name if re.search(r"\w", name) else ""


def guess_language(text: str) -> str:
    return "th" if _THAI_RE.search(text or "") else "en"


def chat_language(text: str, history: list) -> str:
    """Thai if this message or one of the customer's last 3 messages has Thai in it, else English.

    So a Thai customer who types only "COZI" or "ok" still gets Thai back.
    """
    recent = [m["content"] for m in history if m["role"] == "user"][-3:]
    return "th" if any(_THAI_RE.search(t or "") for t in [text, *recent]) else "en"


def is_open(now: datetime = None) -> bool:
    now = (now or datetime.now(timezone.utc)).astimezone(CHINA_TZ)
    return ADMIN_OPEN_HOUR <= now.hour < ADMIN_CLOSE_HOUR


def admin_hours_line(lang: str, now: datetime = None) -> str:
    return HOURS_LINE[lang]["open" if is_open(now) else "closed"]


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

    def _ask_model(self, user_id: str, text: str, live_sheet: str = "", name: str = "",
                   lang: str = "en", form_sent: bool = False) -> dict:
        history = self._get_history(user_id) + [{"role": "user", "content": text}]
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in history
        ]
        system = SYSTEM_PROMPT
        system += "\n\nReply in " + ("Thai (ภาษาไทย)." if lang == "th" else "English.")
        if form_sent:
            system += (" The sign-up form link was already sent in this chat: don't mention the form again. If the "
                       "customer gives booking details, thank them warmly and say an admin will follow up.")
        if name:
            system += f'\n\nThe customer\'s LINE name is "{name}", so call them "K.{name}" in Thai replies.'
        else:
            system += '\n\nThe customer has no usable LINE name, so call them "คุณลูกค้า" in Thai replies.'
        if live_sheet:
            system += (
                "\n\nLatest FAQ from the shop's Google Sheet (also part of the knowledge base; if it disagrees "
                f"with the knowledge base above, trust this sheet):\n<live_sheet>\n{live_sheet}\n</live_sheet>"
            )
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=OUTPUT_SCHEMA,
            # temperature left at the default: Gemini 3.x models are tuned for 1.0
            max_output_tokens=8192,  # thinking + answer share this budget; keep it roomy
        )
        errors = []
        for model in MODELS:
            try:
                response = self.client.models.generate_content(model=model, contents=contents, config=config)
                cand = response.candidates[0] if response.candidates else None
                usage = getattr(response, "usage_metadata", None)
                log.info(
                    "model=%s finish=%s thoughts=%s answer=%s", model,
                    getattr(cand, "finish_reason", None),
                    getattr(usage, "thoughts_token_count", None), getattr(usage, "candidates_token_count", None),
                )
                raw = response.text
                if not raw:
                    raise ValueError(f"empty response: {getattr(cand, 'finish_reason', 'no candidates')}")
                result = json.loads(raw)
                self.last_model = model
                return result
            except Exception as exc:  # overloaded / timeout / bad output: try the next model
                errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:120]}")
        raise RuntimeError(" | ".join(errors))

    # -- public

    def reply(self, user_id: str, text: str, now: datetime = None, name: str = "") -> dict:
        """Return {"messages": [str, ...], "intent": str, "note": str}. Empty messages = stay quiet."""
        text = (text or "").strip()
        if self.is_paused(user_id):
            return {"messages": [], "intent": "paused", "note": "chat handed to admin"}

        note = ""
        live_sheet = sheet_text()
        name = clean_name(name)
        lang = chat_language(text, self._get_history(user_id))  # decided here, not by the model
        form_sent = bool(self.store.get(f"form:{user_id}"))
        try:
            result = self._ask_model(user_id, text, live_sheet, name, lang, form_sent)
            intent, body = result["intent"], result["reply"].strip()
        except Exception as exc:  # API down, bad JSON, refusal: never leave the customer hanging
            intent, body = "cannot_answer", ""
            note = f"fallback after error: {type(exc).__name__}: {exc}"

        if intent in ("answer", "interested"):
            bad = unknown_numbers(body, text, live_sheet + " " + name)
            if bad or not body:
                note = f"blocked reply (unknown numbers {sorted(bad)})" if bad else "empty reply"
                intent, body = "cannot_answer", ""

        if intent == "answer":
            messages = [body]
        elif intent == "interested":
            messages = [body]
            if not form_sent:  # the form link goes out once per chat
                messages.append(CLOSING[lang])
                self.store.set(f"form:{user_id}", True, FORM_ONCE_DAYS * 86400)
        elif intent == "talk_to_admin":
            self.store.set(f"pause:{user_id}", True, ADMIN_PAUSE_HOURS * 3600)
            messages = [TALK_TO_ADMIN[lang].format(hours=admin_hours_line(lang, now))]
        else:
            if intent != "referral":
                intent = "cannot_answer"
            t = (now or datetime.now(timezone.utc)).timestamp()
            last = self.store.get(f"handoff:{user_id}")
            if last and t - float(last) < HANDOFF_BURST_SECONDS:  # messages in a row: one hand-off is enough
                note = (note + " | " if note else "") + "hand-off just sent, staying quiet"
                return {"messages": [], "intent": intent, "note": note}
            self.store.set(f"handoff:{user_id}", t, FORM_ONCE_DAYS * 86400)
            msg = HANDOFF[lang]["referral" if intent == "referral" else ("again" if last else "first")]
            if not is_open(now):
                msg += "\n" + CLOSED_NOTE[lang]
            messages = [msg]

        self._remember(user_id, text, "\n\n".join(messages))
        return {"messages": messages, "intent": intent, "note": note}

    def non_text_reply(self) -> list:
        """Stickers, photos, voice: we can't tell the language, so answer in both."""
        return [NON_TEXT["th"] + "\n\n" + NON_TEXT["en"]]
