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

THAI_TZ = timezone(timedelta(hours=7))
ADMIN_OPEN_HOUR = 10   # 10:00 Thai time
ADMIN_CLOSE_HOUR = 18  # 18:00 Thai time
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
        "open": "แอดมินจะรีบตอบกลับโดยเร็วที่สุด (เวลาทำการ 10:00–18:00 น.)",
        "closed": "ตอนนี้อยู่นอกเวลาทำการ แอดมินจะตอบกลับในเวลาทำการถัดไป 10:00 น. ค่ะ",
    },
    "en": {
        "open": "Our admin will get back to you as soon as possible (office hours 10:00–18:00 Thailand time, GMT+7).",
        "closed": "We're outside office hours right now — our admin will reply from 10:00 Thailand time (GMT+7).",
    },
}

# Sent instead of an AI reply when an admin needs to take over. "first" the first time in a chat,
# "again" if the customer asks more, "referral" when someone recommended JiaJia to them.
HANDOFF = {
    "th": {
        "first": "เดี๋ยวทางแอดมินจะรีบติดต่อกลับนะคะ 🙏✨",
        "again": "รับเรื่องไว้แล้วนะคะ 📝 เดี๋ยวแอดมินจะรีบตอบกลับโดยเร็วค่ะ 🙏",
        "referral": "ขอบคุณที่ได้รับการแนะนำมานะคะ 🥰 รับเรื่องไว้แล้ว เดี๋ยวแอดมินจะรีบติดต่อกลับโดยเร็วค่ะ 🙏✨",
    },
    "en": {
        "first": "Our admin will get back to you shortly 🙏✨",
        "again": "Got it, noted 📝 Our admin will get back to you as soon as possible 🙏",
        "referral": "Thanks for coming to us through a referral 🥰 Noted, and our admin will contact you shortly 🙏✨",
    },
}

# Added under a hand-off when the team is offline.
CLOSED_NOTE = {
    "th": "🌙 ขณะนี้อยู่นอกเวลาทำการนะคะ แอดมินจะตอบกลับในเวลาทำการ 10:00–18:00 น. ค่ะ",
    "en": "🌙 We're outside office hours right now. Our admin will reply during office hours, 10:00–18:00 Thailand time (GMT+7).",
}

TALK_TO_ADMIN = {
    "th": "ได้เลยค่ะ ส่งเรื่องให้แอดมินแล้ว 🙏 {hours}\nระหว่างนี้ Xiao Jia จะไม่ตอบแทรกในแชทนี้ ให้แอดมินคุยกับคุณโดยตรงนะคะ 🙋‍♀️",
    "en": "Sure! I've passed this to our admin 🙏 {hours}\nXiao Jia will stay quiet in this chat so you can talk to them directly 🙋‍♀️",
}

# The question menu: shown in the welcome as a numbered list and under every reply as buttons.
# (button label, max 20 characters; the question it sends). The last item always hands over to the admin.
MENU = {
    "th": [
        ("แพ็กเกจและราคา", "แพ็กเกจมีอะไรบ้าง ราคาเท่าไหร่คะ"),
        ("บริการรวมอะไรบ้าง", "แต่ละแพ็กเกจรวมบริการอะไรบ้างคะ"),
        ("ขั้นตอนการหาห้อง", "ขั้นตอนการหาห้องกับ JiaJia เป็นยังไงคะ"),
        ("ค่าใช้จ่ายเช่าห้อง", "เช่าห้องที่จีนมีค่าใช้จ่ายอะไรบ้างคะ"),
        ("บริการเสริม", "มีบริการเสริมอะไรบ้างคะ"),
        ("คุยกับแอดมิน", "คุยกับแอดมิน"),
    ],
    "en": [
        ("Packages & prices", "What packages do you have and how much are they?"),
        ("What's included", "What does each package include?"),
        ("How it works", "How does finding a room with JiaJia work?"),
        ("Renting costs", "What costs are there when renting a room in China?"),
        ("Add-on services", "What add-on services do you have?"),
        ("Talk to admin", "Talk to admin"),
    ],
}

WELCOME = {
    "th": (
        "สวัสดีค่ะ{name} 😊 นี่คือ Xiao Jia（小家）ผู้ช่วยตอบคำถามของ JiaJia ค่ะ 🏠✨\n"
        "ถามมาได้เลยนะคะ หรือพิมพ์เลข / กดปุ่มด้านล่างเพื่อเลือกคำถามก็ได้ค่ะ 👇\n"
        "{menu}\n"
        "อยากคุยกับแอดมินโดยตรง พิมพ์ 6 หรือกด \"คุยกับแอดมิน\" ได้เลย Xiao Jia จะหยุดตอบให้ค่ะ 🙋‍♀️"
    ),
    "en": (
        "Hi{name}! 😊 I'm Xiao Jia（小家）, JiaJia's assistant 🏠✨\n"
        "Ask me anything, or type a number / tap a button below to pick a question 👇\n"
        "{menu}\n"
        "Want to talk to our admin directly? Type 6 or tap \"Talk to admin\" and I'll step aside 🙋‍♀️"
    ),
}
GREETING_AGAIN = {
    "th": "สวัสดีค่ะ 😊 มีอะไรให้ Xiao Jia ช่วยไหมคะ ✨",
    "en": "Hi again! 😊 What can Xiao Jia help you with? ✨",
}
WELCOME_MARK = "Xiao Jia（小家）"  # how we spot the welcome (and its numbered menu) in the chat history
WELCOME_ONCE_DAYS = 180

NON_TEXT = {
    "th": "ขอบคุณค่ะ 😊 ตอนนี้ Xiao Jia อ่านได้เฉพาะข้อความตัวอักษรนะคะ พิมพ์คำถามมาได้เลย เช่น \"แพ็กเกจราคาเท่าไหร่\"",
    "en": "Thanks! 😊 I can only read text for now — just type your question, e.g. \"How much are your packages?\"",
}

# ---------------------------------------------------------------- prompt

SYSTEM_PROMPT = f"""You are Xiao Jia (小家), the LINE chat assistant for JiaJia 家家, a small founder-led team that helps international \
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
only in your first reply of a chat and then now and then (at most every third reply), never in two replies in a row.
Currency: in Thai replies give package prices in THB with RMB in brackets, e.g. "6,700 บาท (1,340 RMB)". In English \
replies give RMB. Add-ons only have RMB prices — give them in RMB in both languages.

Facts: use ONLY the knowledge base below. Copy prices exactly as written. Never estimate, convert currencies, \
calculate new amounts, or invent rent prices, room availability, discounts, dates or policies. If the knowledge base \
doesn't contain the answer (rent for a specific area, Survival Kit or Wi-Fi prices, airport pick-up price, \
discounts, anything about a specific booking), choose intent "cannot_answer" — the system then sends the hand-off \
message, so leave "reply" as an empty string in that case.

Agent fee: if asked about the agent fee (ค่าเอเจ้น / ค่านายหน้า / 中介费), say only that yes, there is one: it is usually 50% of one month's rent, paid once, separately to the Chinese agent (not to JiaJia). Never say or hint that some rooms have no agent fee or that it might be waived. If they ask whether they can avoid it, don't answer "yes" — just explain that the fee applies.
Brand name: always write "JiaJia" in Latin letters, never translate or transliterate it (no บ้านจ้า, เจียเจีย).

Pick exactly one intent:
- "greeting": the customer only says hello / hi / สวัสดี with no question. reply = a short, warm greeting that \
invites their question.
- "answer": you can fully answer from the knowledge base (prices, what's included, fees, steps, FAQ, greetings, thanks).
- "interested": the customer shows buying intent — wants to book or start, asks how to sign up / pay / begin, says \
which package they want, or gives move-in details. Write a short, warm reply (1–3 lines) about their choice; the \
system appends the sign-up form link and the WeChat note right after it, so do NOT mention the form, the link, \
WeChat, "the system" or that more details will follow.
- "cannot_answer": the knowledge base doesn't cover it (see above). reply = "".
- "talk_to_admin": the customer explicitly asks to talk to a human / admin / staff. reply = "".
- "referral": the customer says someone recommended JiaJia or gave them this contact (e.g. "ได้คอนแทคมาจากพี่ปุ้ย", \
"my friend referred me"). reply = "" (the system welcomes them and says an admin will contact them).
Repeated question: only if the customer sends the SAME question again (nearly the same words) right after you \
already answered it, they need a person: choose "cannot_answer". A follow-up or a narrower question (e.g. "how much is \
SOLO?" after you listed all packages) is NOT a repeat: answer it briefly.

Set "language" to the language the system tells you to reply in.

<knowledge_base>
{KNOWLEDGE}
</knowledge_base>"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["greeting", "answer", "interested", "cannot_answer", "talk_to_admin", "referral"]},
        "language": {"type": "string", "enum": ["th", "en"]},
        "reply": {"type": "string"},
    },
    "required": ["intent", "language", "reply"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------- helpers

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_THAI_RE = re.compile(r"[฀-๿]")
# The customer asks for a person: handled in code so it works even when the AI is down.
_WANTS_ADMIN_RE = re.compile(
    r"^(อยาก|ขอ)*(คุย|ติดต่อ|ต่อสาย)(กับ)?(แอดมิน|admin|คนจริง|เจ้าหน้าที่|พนักงาน)|"
    r"^(i'?d like to |i would like to |can i |could i |i want to |i need to |please )?(talk|speak|chat) (to|with) (an? )?(admin|human|person|real person|staff|agent)",
    re.I,
)
# Questions the admin always answers personally, whatever the model thinks.
_ADMIN_ONLY_RE = re.compile(r"discount|promo|coupon|cheaper|lower price|ส่วนลด|ลดราคา|ลดได้|ลดหน่อย|โปรโมชั่น|มีโปร|โปรไหม|คูปอง|ถูกลง", re.I)


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


def menu_text(lang: str) -> str:
    return "\n".join(f"{i}. {label}" for i, (label, _) in enumerate(MENU[lang], 1))


def welcome_text(lang: str, name: str) -> str:
    if lang == "th":
        who = f" K.{name}" if name else " คุณลูกค้า"
    else:
        who = f" {name}" if name else ""
    return WELCOME[lang].format(name=who, menu=menu_text(lang))


def wants_admin(text: str) -> bool:
    return bool(_WANTS_ADMIN_RE.search(" ".join((text or "").lower().split())))


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
    now = (now or datetime.now(timezone.utc)).astimezone(THAI_TZ)
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
                   lang: str = "en", form_sent: bool = False, first: bool = False) -> dict:
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
        last_bot = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
        if first:  # the welcome right before this reply already greets them by name
            system += ("\n\nA welcome message that greets the customer by name is sent right before your reply: "
                       "don't greet them again and don't use their name.")
        elif name and f"K.{name}" in last_bot:  # used it last time: skip it this time
            system += f'\n\nDo NOT use the customer\'s name in this reply (you used it last time).'
        elif name:
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
        """Return {"messages": [str, ...], "quick_replies": [(label, text), ...], "intent": str, "note": str}.

        Empty messages = stay quiet. quick_replies are the buttons shown under the last message.
        """
        text = (text or "").strip()
        if self.is_paused(user_id):
            return {"messages": [], "quick_replies": [], "intent": "paused", "note": "chat handed to admin"}

        note = ""
        history = self._get_history(user_id)
        name = clean_name(name)
        lang = chat_language(text, history)  # decided here, not by the model
        first = not history and not self.store.get(f"welcomed:{user_id}")
        last_bot = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")

        # Right after the welcome, a bare number picks from the menu.
        if WELCOME_MARK in last_bot and text in {str(i) for i in range(1, len(MENU[lang]) + 1)}:
            text = MENU[lang][int(text) - 1][1]
            note = "picked from menu"

        if wants_admin(text):  # button, "6" or typed request: no AI needed, works even if the AI is down
            intent, body = "talk_to_admin", ""
        else:
            live_sheet = sheet_text()
            form_sent = bool(self.store.get(f"form:{user_id}"))
            try:
                result = self._ask_model(user_id, text, live_sheet, name, lang, form_sent, first)
                intent, body = result["intent"], result["reply"].strip()
            except Exception as exc:  # API down, bad JSON, refusal: never leave the customer hanging
                intent, body = ("greeting" if first else "cannot_answer"), ""  # a newcomer still gets the welcome
                note = f"fallback after error: {type(exc).__name__}: {exc}"

            if intent in ("answer", "interested") and _ADMIN_ONLY_RE.search(text):
                note = "discount question: admin answers"
                intent, body = "cannot_answer", ""

            if intent in ("answer", "interested"):
                bad = unknown_numbers(body, text, live_sheet + " " + name)
                if bad or not body:
                    note = f"blocked reply (unknown numbers {sorted(bad)})" if bad else "empty reply"
                    intent, body = "cannot_answer", ""

        if intent == "greeting":
            messages = [] if first else [body or GREETING_AGAIN[lang]]
        elif intent == "answer":
            messages = [body]
        elif intent == "interested":
            messages = [body]
            if not self.store.get(f"form:{user_id}"):  # the form link goes out once per chat
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
            if intent != "referral" and last and t - float(last) < HANDOFF_BURST_SECONDS:  # messages in a row
                note = (note + " | " if note else "") + "hand-off just sent, staying quiet"
                return {"messages": [], "quick_replies": [], "intent": intent, "note": note}
            self.store.set(f"handoff:{user_id}", t, FORM_ONCE_DAYS * 86400)
            msg = HANDOFF[lang]["referral" if intent == "referral" else ("again" if last else "first")]
            if not is_open(now):
                msg += "\n" + CLOSED_NOTE[lang]
            messages = [msg]

        if first:
            self.store.set(f"welcomed:{user_id}", True, WELCOME_ONCE_DAYS * 86400)
            if intent != "talk_to_admin":  # someone who asks for the admin straight away skips the intro
                messages = [welcome_text(lang, name)] + messages

        self._remember(user_id, text, "\n\n".join(messages))
        quick = [] if intent == "talk_to_admin" else MENU[lang]
        return {"messages": messages, "quick_replies": quick, "intent": intent, "note": note,
                "lang": lang, "first": first}

    def non_text_reply(self) -> list:
        """Stickers, photos, voice: we can't tell the language, so answer in both."""
        return [NON_TEXT["th"] + "\n\n" + NON_TEXT["en"]]
