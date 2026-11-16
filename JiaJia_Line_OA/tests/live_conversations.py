"""Live test: runs real conversations through Gemini (needs GEMINI_API_KEY in .env).

Run:  .venv/bin/python tests/live_conversations.py [rounds]

Each conversation is run `rounds` times (default 3) to check the bot answers consistently.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from bot.brain import Brain  # noqa: E402

logging.getLogger("google_genai.types").setLevel(logging.ERROR)

# (customer message, expected intent(s), text that must appear somewhere in the reply)
CONVERSATIONS = {
    "ราคาภาษาไทย → สนใจ": [
        ("สวัสดีค่ะ แพ็กเกจหาห้องราคาเท่าไหร่คะ", {"answer"}, ["6,700"]),
        ("ถ้ามา 2 คนล่ะคะ", {"answer", "interested"}, ["9,700"]),
        ("สนใจค่ะ เริ่มยังไงคะ", {"interested"}, ["linktr.ee/jiajia.roomhunt", "WeChat"]),
    ],
    "English pricing → add-on → booking": [
        ("Hi! How much is the TRIO package?", {"answer"}, ["2,350"]),
        ("Do you offer deep cleaning?", {"answer", "interested"}, ["450"]),
        ("Great, I want to book the SOLO package. What's next?", {"interested"}, ["linktr.ee/jiajia.roomhunt"]),
    ],
    "คำถามที่ไม่มีในคลัง": [
        ("ค่าเช่าห้องแถวมหาลัยฟู่ตั้นเดือนละเท่าไหร่คะ", {"cannot_answer"}, ["1–1.5", "linktr.ee/jiajia.roomhunt"]),
        ("How much is airport pickup from Pudong?", {"cannot_answer"}, ["admin"]),
        ("ค่าเอเจนต์จ่ายให้ใครคะ", {"answer"}, ["เอเจนต์"]),
    ],
}


def run(rounds: int) -> bool:
    ok = True
    for r in range(1, rounds + 1):
        for name, turns in CONVERSATIONS.items():
            brain = Brain()
            user = f"test-{r}-{name}"
            print(f"\n=== round {r}: {name}")
            for text, intents, must in turns:
                out = brain.reply(user, text)
                joined = "\n".join(out["messages"])
                missing = [m for m in must if m.lower() not in joined.lower()]
                passed = out["intent"] in intents and not missing
                ok &= passed
                print(f"\n👤 {text}\n🤖 [{out['intent']} via {brain.last_model}] {joined}")
                if out["note"]:
                    print(f"   note: {out['note']}")
                if not passed:
                    print(f"   ❌ expected intent {sorted(intents)}, missing {missing}")
    print("\nALL PASSED ✅" if ok else "\nSOME CHECKS FAILED ❌")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run(int(sys.argv[1]) if len(sys.argv) > 1 else 3) else 1)
