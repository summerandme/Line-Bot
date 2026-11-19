"""LINE webhook server for the JiaJia bot.

LINE sends every customer message to POST /callback. We work out the reply and
send it with the reply token, falling back to a push message if the token expired.
Runs on Vercel (api/index.py) or any always-on host (gunicorn app:app).
"""

import logging
import os
import threading
import time

from dotenv import load_dotenv
from flask import Flask, abort, request

load_dotenv()

from linebot.v3 import WebhookHandler  # noqa: E402
from linebot.v3.exceptions import InvalidSignatureError  # noqa: E402
from linebot.v3.messaging import (  # noqa: E402
    ApiClient,
    Configuration,
    MessageAction,
    MessagingApi,
    PushMessageRequest,
    QuickReply,
    QuickReplyItem,
    ReplyMessageRequest,
    ShowLoadingAnimationRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent  # noqa: E402

from bot.brain import Brain  # noqa: E402

# LINE OA's own auto-response sends English info the moment a foreign customer writes.
# Xiao Jia's first reply to them waits until this many seconds after the message arrived,
# so it always lands after LINE's message instead of in the middle of it.
EN_FIRST_WAIT_SECONDS = float(os.environ.get("EN_FIRST_WAIT_SECONDS", "6"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("jiajia")

CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
CHANNEL_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")

app = Flask(__name__)
handler = WebhookHandler(CHANNEL_SECRET)
line_config = Configuration(access_token=CHANNEL_TOKEN)
brain = Brain()


def _send(user_id: str, reply_token: str, texts: list, buttons: list = ()) -> None:
    """Send up to 5 texts; buttons [(label, text), ...] show as tap-to-send choices under the last one."""
    messages = [TextMessage(text=t[:5000]) for t in texts if t][:5]
    if not messages:
        return
    if buttons:
        messages[-1].quick_reply = QuickReply(items=[
            QuickReplyItem(action=MessageAction(label=label[:20], text=text)) for label, text in buttons[:13]
        ])
    with ApiClient(line_config) as api_client:
        api = MessagingApi(api_client)
        try:
            api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=messages))
        except Exception as exc:  # token expired or already used: push instead
            log.warning("reply failed (%s), pushing instead", exc)
            api.push_message(PushMessageRequest(to=user_id, messages=messages))


def _loading(user_id: str) -> None:
    """Show LINE's "typing…" animation while Claude thinks (1:1 chats only)."""
    try:
        with ApiClient(line_config) as api_client:
            MessagingApi(api_client).show_loading_animation(
                ShowLoadingAnimationRequest(chat_id=user_id, loading_seconds=20)
            )
    except Exception as exc:
        log.info("loading animation skipped: %s", exc)


def _display_name(user_id: str) -> str:
    """The customer's LINE display name ("" if LINE won't give it), cached for a day."""
    key = f"name:{user_id}"
    cached = brain.store.get(key)
    if cached is not None:
        return cached
    try:
        with ApiClient(line_config) as api_client:
            name = MessagingApi(api_client).get_profile(user_id).display_name or ""
    except Exception as exc:  # user blocked the OA, API hiccup, ...: just use คุณลูกค้า
        log.info("profile lookup skipped: %s", exc)
        return ""
    brain.store.set(key, name, 24 * 3600)
    return name


def _work(event) -> None:
    started = time.monotonic()
    user_id = event.source.user_id
    try:
        if isinstance(event.message, TextMessageContent):
            if brain.is_paused(user_id):
                log.info("user=%s paused (admin chat), staying quiet", user_id)
                return
            _loading(user_id)
            result = brain.reply(user_id, event.message.text, name=_display_name(user_id))
            log.info("user=%s intent=%s note=%s", user_id, result["intent"], result["note"])
            texts, buttons = result["messages"], result["quick_replies"]
            if texts and result.get("first") and result.get("lang") == "en":
                time.sleep(max(0.0, EN_FIRST_WAIT_SECONDS - (time.monotonic() - started)))
        else:
            if brain.is_paused(user_id):
                return
            texts, buttons = brain.non_text_reply(), []
        _send(user_id, event.reply_token, texts, buttons)
    except Exception:
        log.exception("failed to handle message from %s", user_id)


# On Vercel (serverless) work stops once we answer LINE, so we finish the reply
# first. On an always-on server we answer LINE at once and reply in a thread.
SERVERLESS = bool(os.environ.get("VERCEL"))


@handler.add(MessageEvent)
def on_message(event):
    if getattr(event.source, "type", "") != "user":
        return  # ignore group chats
    if SERVERLESS:
        _work(event)
    else:
        threading.Thread(target=_work, args=(event,), daemon=True).start()


@app.post("/callback")
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


@app.get("/")
def health():
    return "JiaJia bot is running"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
