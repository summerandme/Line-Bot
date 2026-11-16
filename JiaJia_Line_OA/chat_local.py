"""Chat with the bot in your terminal (needs GEMINI_API_KEY in .env).

Run:  .venv/bin/python chat_local.py
"""

from dotenv import load_dotenv

load_dotenv()

from bot.brain import Brain  # noqa: E402

brain = Brain()
print("พิมพ์ข้อความเหมือนเป็นลูกค้า (Ctrl+C เพื่อออก)\n")
while True:
    try:
        text = input("👤 ")
    except (KeyboardInterrupt, EOFError):
        break
    out = brain.reply("local-user", text)
    print(f"🤖 [{out['intent']}]")
    for m in out["messages"] or ["(บอทเงียบ — แชทนี้ส่งให้แอดมินแล้ว)"]:
        print(m, "\n")
