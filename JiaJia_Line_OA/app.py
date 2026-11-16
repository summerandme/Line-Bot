"""LINE webhook server for the JiaJia bot.

LINE sends every customer message to POST /callback. We work out the reply and
send it with the reply token, falling back to a push message if the token expired.
Runs on Vercel (api/index.py) or any always-on host (gunicorn app:app).
"""

import logging
import os
import threading

from dotenv import load_dotenv
from flask import Flask, abort, request

load_dotenv()

from linebot.v3 import WebhookHandler  # noqa: E402
from linebot.v3.exceptions import InvalidSignatureError  # noqa: E402
from linebot.v3.messaging import (  # noqa: E402
    ApiClient,
    Configuration,
    MessagingApi,
    PushMessageRequest,
    ReplyMessageRequest,
    ShowLoadingAnimationRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent  # noqa: E402

from bot.brain import Brain  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("jiajia")

CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
CHANNEL_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")

app = Flask(__name__)
handler = WebhookHandler(CHANNEL_SECRET)
line_config = Configuration(access_token=CHANNEL_TOKEN)
brain = Brain()


def _send(user_id: str, reply_token: str, texts: list) -> None:
    messages = [TextMessage(text=t[:5000]) for t in texts if t][:5]
    if not messages:
        return
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


def _work(event) -> None:
    user_id = event.source.user_id
    try:
        if isinstance(event.message, TextMessageContent):
            if brain.is_paused(user_id):
                log.info("user=%s paused (admin chat), staying quiet", user_id)
                return
            _loading(user_id)
            result = brain.reply(user_id, event.message.text)
            log.info("user=%s intent=%s note=%s", user_id, result["intent"], result["note"])
            texts = result["messages"]
        else:
            if brain.is_paused(user_id):
                return
            texts = brain.non_text_reply()
        _send(user_id, event.reply_token, texts)
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
