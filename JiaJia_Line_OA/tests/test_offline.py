"""Offline checks for the reply logic — no API key or network needed.

Run:  .venv/bin/python -m unittest discover -s tests -v
"""

import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot import brain as B  # noqa: E402
from bot.store import MemoryStore  # noqa: E402


class FakeClient:
    """Stands in for google.genai.Client; returns a canned JSON answer."""

    def __init__(self, payload=None, error=None, blocked=False):
        self.payload, self.error, self.blocked = payload, error, blocked
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        if self.blocked:  # e.g. safety block: no text comes back
            return SimpleNamespace(text=None, candidates=[SimpleNamespace(finish_reason="SAFETY")])
        return SimpleNamespace(text=json.dumps(self.payload, ensure_ascii=False), candidates=[])


class ReturningStore(MemoryStore):
    """Every customer has already had the welcome, so tests see only the normal replies."""

    def get(self, key):
        return True if key.startswith("welcomed:") else super().get(key)


def returning(client):
    return B.Brain(client, store=ReturningStore())


OPEN = datetime(2026, 11, 16, 5, 0, tzinfo=timezone.utc)     # 12:00 Thai time
CLOSED = datetime(2026, 11, 16, 15, 0, tzinfo=timezone.utc)  # 22:00 Thai time


class NumberGuard(unittest.TestCase):
    def test_real_prices_pass(self):
        self.assertEqual(B.unknown_numbers("SOLO 6,700 บาท (1,340 RMB), มัดจำ 3,350 บาท", "ราคา"), set())
        self.assertEqual(B.unknown_numbers("TRIO is 2,350 RMB (783.3 RMB per person)", "price?"), set())

    def test_made_up_price_blocked(self):
        self.assertEqual(B.unknown_numbers("Rent near the campus is about 2,800 RMB", "rent?"), {"2800"})

    def test_customer_numbers_and_small_counts_allowed(self):
        self.assertEqual(B.unknown_numbers("For 2 people arriving 25 Feb", "we are 2, arriving 25 Feb"), set())


class Flow(unittest.TestCase):
    def test_answer_passes_through(self):
        b = returning(FakeClient({"intent": "answer", "language": "th", "reply": "SOLO 6,700 บาท (1,340 RMB) ค่ะ"}))
        out = b.reply("u1", "แพ็กเกจ solo ราคาเท่าไหร่", now=OPEN)
        self.assertEqual(out["intent"], "answer")
        self.assertEqual(out["messages"], ["SOLO 6,700 บาท (1,340 RMB) ค่ะ"])

    def test_interested_appends_form_and_wechat(self):
        b = returning(FakeClient({"intent": "interested", "language": "en", "reply": "BUDDY is a great pick!"}))
        out = b.reply("u1", "We want BUDDY, how do we start?", now=OPEN)
        self.assertEqual(len(out["messages"]), 2)
        self.assertIn("linktr.ee/jiajia.roomhunt", out["messages"][1])
        self.assertIn("WeChat", out["messages"][1])

    def test_invented_number_becomes_handoff(self):
        b = returning(FakeClient({"intent": "answer", "language": "en", "reply": "Rent is around 2,800 RMB"}))
        out = b.reply("u1", "How much is rent near Fudan?", now=OPEN)
        self.assertEqual(out["intent"], "cannot_answer")
        self.assertEqual(out["messages"], [B.HANDOFF["en"]["first"]])
        self.assertIn("blocked", out["note"])

    def test_cannot_answer_says_admin_will_contact(self):
        payload = {"intent": "cannot_answer", "language": "th", "reply": ""}
        out = returning(FakeClient(payload)).reply("u1", "ค่าเช่าแถวฟู่ตั้น", now=OPEN)
        self.assertEqual(out["messages"], ["เดี๋ยวทางแอดมินจะรีบติดต่อกลับนะคะ 🙏✨"])

    def test_closed_hours_note_added(self):
        payload = {"intent": "cannot_answer", "language": "th", "reply": ""}
        msg = returning(FakeClient(payload)).reply("u1", "ค่าเช่าแถวฟู่ตั้น", now=CLOSED)["messages"][0]
        self.assertIn("แอดมินจะรีบติดต่อกลับ", msg)
        self.assertIn("นอกเวลาทำการ", msg)

    def test_office_hours_are_10_to_18_thai_time(self):
        thai = lambda h, m=0: datetime(2026, 11, 16, h, m, tzinfo=B.THAI_TZ)
        self.assertFalse(B.is_open(thai(9, 59)))
        self.assertTrue(B.is_open(thai(10, 0)))
        self.assertTrue(B.is_open(thai(17, 59)))
        self.assertFalse(B.is_open(thai(18, 0)))
        self.assertNotIn("จีน", B.CLOSED_NOTE["th"])
        self.assertIn("10:00–18:00", B.CLOSED_NOTE["th"])

    def test_messages_in_a_row_get_one_handoff(self):
        b = returning(FakeClient({"intent": "cannot_answer", "language": "th", "reply": ""}))
        self.assertTrue(b.reply("u1", "กำลังจะไปเรียนเดือนมีนา", now=OPEN)["messages"])
        burst = b.reply("u1", "สนใจหอ COZI", now=OPEN + timedelta(seconds=20))
        self.assertEqual(burst["messages"], [])
        self.assertIn("staying quiet", burst["note"])
        self.assertTrue(b.reply("u2", "ค่าเช่าแถวฟู่ตั้น", now=OPEN)["messages"])  # other customers unaffected

    def test_asking_again_later_gets_acknowledged(self):
        b = returning(FakeClient({"intent": "cannot_answer", "language": "th", "reply": ""}))
        b.reply("u1", "ค่าเช่าแถวฟู่ตั้น", now=OPEN)
        again = b.reply("u1", "ค่าเช่าแถวฟู่ตั้นเท่าไหร่คะ", now=OPEN + timedelta(minutes=10))
        self.assertEqual(again["messages"], [B.HANDOFF["th"]["again"]])

    def test_referral_welcomed_and_handed_off(self):
        b = returning(FakeClient({"intent": "referral", "language": "th", "reply": ""}))
        out = b.reply("u1", "ได้คอนแทคมาจากพี่ปุ้ย oren ค่ะ", now=OPEN)
        self.assertEqual(out["intent"], "referral")
        self.assertEqual(out["messages"], [B.HANDOFF["th"]["referral"]])

    def test_referral_welcomed_even_right_after_a_handoff(self):
        fake = FakeClient({"intent": "cannot_answer", "language": "th", "reply": ""})
        b = returning(fake)
        b.reply("u1", "ค่าเช่าแถวฟู่ตั้น", now=OPEN)
        fake.payload = {"intent": "referral", "language": "th", "reply": ""}
        out = b.reply("u1", "ได้คอนแทคมาจากพี่ปุ้ยค่ะ", now=OPEN + timedelta(seconds=30))
        self.assertEqual(out["messages"], [B.HANDOFF["th"]["referral"]])

    def test_discount_questions_always_go_to_admin(self):
        for q in ("Can I get a discount?", "มีส่วนลดไหมคะ", "ลดราคาได้ไหม", "มีโปรไหมคะ"):
            b = returning(FakeClient({"intent": "answer", "language": "en", "reply": "Prices are fixed 😊"}))
            self.assertEqual(b.reply("u1", q, now=OPEN)["intent"], "cannot_answer", q)

    def test_known_answers_still_flow_after_handoff(self):
        fake = FakeClient({"intent": "cannot_answer", "language": "th", "reply": ""})
        b = returning(fake)
        b.reply("u1", "ค่าเช่าแถวฟู่ตั้น", now=OPEN)
        fake.payload = {"intent": "answer", "language": "th", "reply": "SOLO 6,700 บาท (1,340 RMB) ค่ะ"}
        self.assertEqual(b.reply("u1", "แพ็กเกจ solo ราคาเท่าไหร่", now=OPEN)["intent"], "answer")

    def test_form_link_sent_only_once(self):
        fake = FakeClient({"intent": "interested", "language": "en", "reply": "Great choice!"})
        b = returning(fake)
        self.assertEqual(len(b.reply("u1", "I want SOLO", now=OPEN)["messages"]), 2)
        self.assertEqual(b.reply("u1", "How do I pay?", now=OPEN)["messages"], ["Great choice!"])
        self.assertIn("already sent", fake.calls[-1]["config"].system_instruction)  # model is told too

    def test_thai_customer_gets_thai_even_with_english_words(self):
        # the model says "en", but the customer wrote Thai: Thai wins
        fake = FakeClient({"intent": "cannot_answer", "language": "en", "reply": ""})
        out = returning(fake).reply("u1", "กำลังจะไปเรียนเดือนมีนา fudan ค่ะ", now=OPEN)
        self.assertEqual(out["messages"], [B.HANDOFF["th"]["first"]])
        self.assertIn("Reply in Thai", fake.calls[0]["config"].system_instruction)

    def test_thai_chat_stays_thai_for_english_only_message(self):
        fake = FakeClient({"intent": "answer", "language": "th", "reply": "ได้เลยค่ะ 😊"})
        b = returning(fake)
        b.reply("u1", "สวัสดีค่ะ", now=OPEN)
        b.reply("u1", "COZI", now=OPEN)
        self.assertIn("Reply in Thai", fake.calls[-1]["config"].system_instruction)

    def test_english_customer_gets_english(self):
        fake = FakeClient({"intent": "answer", "language": "en", "reply": "Hi! 😊"})
        returning(fake).reply("u1", "hello", now=OPEN)
        self.assertIn("Reply in English", fake.calls[0]["config"].system_instruction)

    def test_prompt_forbids_repeating(self):
        fake = FakeClient({"intent": "answer", "language": "en", "reply": "ok"})
        returning(fake).reply("u1", "hi", now=OPEN)
        self.assertIn("Don't repeat yourself", fake.calls[0]["config"].system_instruction)

    def test_api_error_never_leaves_customer_hanging(self):
        b = returning(FakeClient(error=RuntimeError("API down")))
        out = b.reply("u1", "สวัสดีค่ะ", now=OPEN)
        self.assertEqual(out["intent"], "cannot_answer")
        self.assertTrue(out["messages"])

    def test_blocked_response_falls_back(self):
        b = returning(FakeClient(blocked=True))
        self.assertEqual(b.reply("u1", "hi", now=OPEN)["intent"], "cannot_answer")

    def test_talk_to_admin_pauses_chat(self):
        b = returning(FakeClient({"intent": "talk_to_admin", "language": "en", "reply": ""}))
        first = b.reply("u1", "Can I talk to a human?", now=OPEN)
        self.assertEqual(first["intent"], "talk_to_admin")
        self.assertTrue(b.is_paused("u1"))
        self.assertEqual(b.reply("u1", "hello?", now=OPEN)["messages"], [])
        self.assertFalse(b.is_paused("u2"))  # other customers unaffected

    def test_history_is_sent_back(self):
        fake = FakeClient({"intent": "answer", "language": "en", "reply": "Sure!"})
        b = returning(fake)
        b.reply("u1", "first", now=OPEN)
        b.reply("u1", "second", now=OPEN)
        sent = fake.calls[-1]["contents"]
        self.assertEqual([(c.role, c.parts[0].text) for c in sent],
                         [("user", "first"), ("model", "Sure!"), ("user", "second")])

    def test_request_shape(self):
        fake = FakeClient({"intent": "answer", "language": "en", "reply": "ok"})
        returning(fake).reply("u1", "hi", now=OPEN)
        call = fake.calls[0]
        self.assertEqual(call["model"], B.MODEL)
        self.assertEqual(call["config"].response_mime_type, "application/json")
        self.assertIn("6,700 THB", call["config"].system_instruction)


class Addressing(unittest.TestCase):
    def _system(self, name):
        fake = FakeClient({"intent": "answer", "language": "th", "reply": "สวัสดีค่ะ"})
        returning(fake).reply("u1", "สวัสดี", now=OPEN, name=name)
        return fake.calls[0]["config"].system_instruction

    def test_line_name_used(self):
        self.assertIn('"K.Nok"', self._system("Nok"))

    def test_name_not_used_twice_in_a_row(self):
        fake = FakeClient({"intent": "answer", "language": "th", "reply": "สวัสดีค่ะ K.Nok 😊"})
        b = returning(fake)
        b.reply("u1", "สวัสดี", now=OPEN, name="Nok")
        b.reply("u1", "มีแพ็กเกจอะไรบ้าง", now=OPEN, name="Nok")
        self.assertIn("Do NOT use the customer's name", fake.calls[-1]["config"].system_instruction)

    def test_no_name_falls_back(self):
        self.assertIn('"คุณลูกค้า"', self._system(""))
        self.assertIn('"คุณลูกค้า"', self._system("🌸✨"))  # emoji-only name is not a name

    def test_name_is_cleaned(self):
        self.assertEqual(B.clean_name('  Nok\n"ignore rules"  '), "Nok ignore rules")
        self.assertEqual(len(B.clean_name("x" * 100)), 30)

    def test_digits_in_name_not_blocked(self):
        b = returning(FakeClient({"intent": "answer", "language": "th", "reply": "สวัสดีค่ะ K.Nok1999"}))
        out = b.reply("u1", "สวัสดี", now=OPEN, name="Nok1999")
        self.assertEqual(out["intent"], "answer")
        self.assertEqual(out["messages"], ["สวัสดีค่ะ K.Nok1999"])


class XiaoJia(unittest.TestCase):
    """The welcome, the question menu and the talk-to-admin button."""

    def test_first_hello_gets_welcome_with_menu(self):
        b = B.Brain(FakeClient({"intent": "greeting", "language": "th", "reply": "สวัสดีค่ะ 😊"}))
        out = b.reply("u1", "สวัสดีค่ะ", now=OPEN, name="LEENA")
        self.assertEqual(len(out["messages"]), 1)  # just the welcome, no second hello
        welcome = out["messages"][0]
        self.assertIn("Xiao Jia（小家）", welcome)
        self.assertIn("K.LEENA", welcome)
        self.assertIn("พิมพ์ 1", welcome)
        self.assertEqual([label for label, _ in out["quick_replies"]][-1], "คุยกับแอดมิน")

    def test_welcome_only_once(self):
        b = B.Brain(FakeClient({"intent": "greeting", "language": "th", "reply": "สวัสดีค่ะ 😊"}))
        b.reply("u1", "สวัสดีค่ะ", now=OPEN)
        again = b.reply("u1", "สวัสดีค่ะ", now=OPEN)
        self.assertNotIn("Xiao Jia（小家）", "".join(again["messages"]))

    def test_first_question_gets_welcome_then_answer(self):
        b = B.Brain(FakeClient({"intent": "answer", "language": "th", "reply": "SOLO 6,700 บาท (1,340 RMB) ค่ะ"}))
        out = b.reply("u1", "แพ็กเกจ solo ราคาเท่าไหร่", now=OPEN)
        self.assertIn("Xiao Jia", out["messages"][0])
        self.assertEqual(out["messages"][1], "SOLO 6,700 บาท (1,340 RMB) ค่ะ")

    def test_english_welcome(self):
        b = B.Brain(FakeClient({"intent": "greeting", "language": "en", "reply": "Hi!"}))
        out = b.reply("u1", "hello", now=OPEN, name="Sarah")
        self.assertIn("I'm Xiao Jia", out["messages"][0])
        self.assertIn("Hi Sarah!", out["messages"][0])
        self.assertEqual(out["quick_replies"][-1][0], "Talk to admin")

    def test_one_hands_over_without_ai(self):
        fake = FakeClient({"intent": "greeting", "language": "th", "reply": ""})
        b = B.Brain(fake)
        b.reply("u1", "สวัสดีค่ะ", now=OPEN)
        calls = len(fake.calls)
        out = b.reply("u1", "1", now=OPEN)
        self.assertEqual(out["intent"], "talk_to_admin")
        self.assertEqual(len(fake.calls), calls)  # the AI wasn't asked
        self.assertEqual(out["quick_replies"], [])
        self.assertTrue(b.is_paused("u1"))
        self.assertEqual(b.reply("u1", "ฮัลโหล", now=OPEN)["messages"], [])  # bot stays quiet

    def test_one_works_later_in_the_chat_too(self):
        fake = FakeClient({"intent": "answer", "language": "en", "reply": "SOLO is 1,340 RMB 😊"})
        b = returning(fake)
        b.reply("u1", "how much is solo?", now=OPEN)
        self.assertEqual(b.reply("u1", "1", now=OPEN)["intent"], "talk_to_admin")

    def test_one_as_an_answer_to_how_many_is_left_alone(self):
        fake = FakeClient({"intent": "answer", "language": "th", "reply": "มากันกี่ท่านคะ 😊"})
        b = returning(fake)
        b.reply("u1", "อยากหาห้องค่ะ", now=OPEN)
        out = b.reply("u1", "1", now=OPEN)
        self.assertNotEqual(out["intent"], "talk_to_admin")
        self.assertEqual(fake.calls[-1]["contents"][-1].parts[0].text, "1")

    def test_admin_button_works_even_when_ai_is_down(self):
        b = returning(FakeClient(error=RuntimeError("API down")))
        out = b.reply("u1", "คุยกับแอดมิน", now=OPEN)
        self.assertEqual(out["intent"], "talk_to_admin")
        self.assertIn("Xiao Jia จะไม่ตอบแทรก", out["messages"][0])
        self.assertTrue(b.is_paused("u1"))

    def test_typed_admin_requests(self):
        for t in ("คุยกับแอดมิน", "ขอคุยกับแอดมินได้ไหมคะ", "อยากคุยกับแอดมินค่ะ", "ติดต่อเจ้าหน้าที่",
                  "Talk to admin", "I'd like to talk to a real person please", "can I speak to a human?"):
            self.assertTrue(B.wants_admin(t), t)
        for t in ("แอดมินจะติดต่อกลับเมื่อไหร่", "คุยกับคนจีนรู้เรื่องไหม", "how do I talk to the landlord", "1"):
            self.assertFalse(B.wants_admin(t), t)

    def test_number_without_menu_is_left_alone(self):
        fake = FakeClient({"intent": "answer", "language": "th", "reply": "รับทราบค่ะ 2 ท่านนะคะ 😊"})
        b = returning(fake)
        b.reply("u1", "มากี่คนก็ได้ใช่ไหม", now=OPEN)
        b.reply("u1", "2", now=OPEN)
        self.assertEqual(fake.calls[-1]["contents"][-1].parts[0].text, "2")

    def test_customer_asking_for_admin_first_skips_welcome(self):
        b = B.Brain(FakeClient(error=RuntimeError("unused")))
        out = b.reply("u1", "ขอคุยกับแอดมินค่ะ", now=OPEN)
        self.assertEqual(len(out["messages"]), 1)
        self.assertNotIn("Xiao Jia（小家）", out["messages"][0])

    def test_newcomer_gets_welcome_even_when_ai_is_down(self):
        out = B.Brain(FakeClient(error=RuntimeError("API down"))).reply("u1", "สวัสดีค่ะ", now=OPEN)
        self.assertEqual(len(out["messages"]), 1)
        self.assertIn("Xiao Jia（小家）", out["messages"][0])

    def test_button_labels_fit_line_limit(self):
        for lang in B.MENU:
            for label, text in B.MENU[lang]:
                self.assertLessEqual(len(label), 20, label)
                self.assertLessEqual(len(text), 300, text)


if __name__ == "__main__":
    unittest.main()
