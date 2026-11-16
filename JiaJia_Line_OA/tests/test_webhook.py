"""Webhook + storage checks, offline (fake LINE API and fake Gemini)."""

import base64
import hashlib
import hmac
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["LINE_CHANNEL_SECRET"] = "testsecret"
os.environ["LINE_CHANNEL_ACCESS_TOKEN"] = "testtoken"
os.environ["VERCEL"] = "1"  # exercise the serverless (finish-before-answering) path

import app as A  # noqa: E402
from bot import brain as B  # noqa: E402
from bot.store import MemoryStore  # noqa: E402
from test_offline import FakeClient  # noqa: E402


def signed(body: dict):
    raw = json.dumps(body)
    sig = base64.b64encode(hmac.new(b"testsecret", raw.encode(), hashlib.sha256).digest()).decode()
    return raw, {"X-Line-Signature": sig, "Content-Type": "application/json"}


def text_event(text, user="Uabc", kind="text"):
    message = {"type": kind, "id": "1", "quoteToken": "q"}
    if kind == "text":
        message["text"] = text
    else:
        message.update({"packageId": "1", "stickerId": "1", "stickerResourceType": "STATIC"})
    return {
        "destination": "x",
        "events": [{
            "type": "message", "mode": "active", "timestamp": 0, "webhookEventId": "e", "replyToken": "rt",
            "deliveryContext": {"isRedelivery": False},
            "source": {"type": "user", "userId": user}, "message": message,
        }],
    }


class Webhook(unittest.TestCase):
    def setUp(self):
        self.sent = []
        A._send = lambda user_id, token, texts: self.sent.append((user_id, token, texts))
        A._loading = lambda user_id: None
        self.client = A.app.test_client()

    def use(self, payload):
        A.brain = B.Brain(FakeClient(payload), store=MemoryStore())

    def test_reply_is_sent_before_returning(self):
        self.use({"intent": "answer", "language": "th", "reply": "SOLO 6,700 บาท (1,340 RMB) ค่ะ"})
        raw, headers = signed(text_event("ราคา solo"))
        self.assertEqual(self.client.post("/callback", data=raw, headers=headers).status_code, 200)
        self.assertEqual(self.sent, [("Uabc", "rt", ["SOLO 6,700 บาท (1,340 RMB) ค่ะ"])])

    def test_sticker_gets_bilingual_prompt(self):
        self.use({"intent": "answer", "language": "en", "reply": "x"})
        raw, headers = signed(text_event("", kind="sticker"))
        self.client.post("/callback", data=raw, headers=headers)
        self.assertIn("พิมพ์คำถาม", self.sent[0][2][0])
        self.assertIn("type your question", self.sent[0][2][0])

    def test_bad_signature_rejected(self):
        raw, _ = signed(text_event("hi"))
        self.assertEqual(self.client.post("/callback", data=raw, headers={"X-Line-Signature": "bad"}).status_code, 400)
        self.assertEqual(self.sent, [])

    def test_paused_chat_stays_quiet(self):
        self.use({"intent": "talk_to_admin", "language": "en", "reply": ""})
        raw, headers = signed(text_event("human please"))
        self.client.post("/callback", data=raw, headers=headers)
        raw, headers = signed(text_event("hello?"))
        self.client.post("/callback", data=raw, headers=headers)
        self.assertEqual(len(self.sent), 1)  # only the hand-off message


class Store(unittest.TestCase):
    def test_memory_store_expiry(self):
        s = MemoryStore()
        s.set("k", [1, 2], ttl_seconds=60)
        self.assertEqual(s.get("k"), [1, 2])
        s.set("k", "old", ttl_seconds=-1)
        self.assertIsNone(s.get("k"))

    def test_history_survives_new_brain_with_shared_store(self):
        shared = MemoryStore()  # stands in for Upstash shared by every Vercel call
        B.Brain(FakeClient({"intent": "answer", "language": "en", "reply": "Hi!"}), store=shared).reply("u", "hello")
        fake = FakeClient({"intent": "answer", "language": "en", "reply": "ok"})
        B.Brain(fake, store=shared).reply("u", "and for 2 people?")
        self.assertEqual([c.parts[0].text for c in fake.calls[0]["contents"]], ["hello", "Hi!", "and for 2 people?"])


if __name__ == "__main__":
    unittest.main()
