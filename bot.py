
import os
import re
import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandObject
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv
from rag import KnowledgeBase, build_context_snippets
from openai import OpenAI

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

bot = Bot(
    token=TELEGRAM_BOT_TOKEN,
    default=DefaultBotProperties(parse_mode="HTML")
)
dp = Dispatcher()

# Подготовка базы знаний
KB_PATH = os.path.join(os.path.dirname(__file__), "data", "knowledge.csv")
kb = KnowledgeBase(KB_PATH)

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def load_system_prompt():
    p = os.path.join(os.path.dirname(__file__), "prompt_system_ru.txt")
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

SYSTEM_PROMPT = load_system_prompt()

async def generate_answer(user_question: str) -> str:
    # Находим релевантные записи в CSV
    hits = kb.query(user_question, top_k=RAG_TOP_K)
    context_text = build_context_snippets(hits) if hits else "Контекст из базы знаний не найден."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Вопрос пользователя:\n{user_question}\n\nКонтекст (из базы знаний):\n{context_text}"}
    ]

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.2
        )
        text = resp.choices[0].message.content.strip()

        # Удаляем Markdown-разметку (**жирный**, ## заголовки)
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # убираем ** **
        text = re.sub(r"#+\s*", "", text)  # убираем ### или ####
        text = re.sub(r"_([^_]+)_", r"\1", text)  # убираем подчёркивания
    except Exception as e:
        text = (
            "Извини, сейчас не удалось получить ответ от модели.\n"
            f"Техническая ошибка: {e}"
        )
    return text

@dp.message(Command("start"))
async def cmd_start(m: Message):
    await m.answer(
        "Привет! Я LegalBot — ассистент по вопросам недвижимости.\n"
        "Задай вопрос через команду /ask или открой /help для примеров."
    )

@dp.message(Command("help"))
async def cmd_help(m: Message):
    await m.answer(
        "Как задать вопрос:\n"
        "• /ask Какие документы нужны для продажи квартиры?\n"
        "• /ask Чем отличается аренда и найм?\n\n"
        "Подсказки:\n"
        "— Пиши конкретно и добавляй детали (цель, статус объекта, ипотека и т. д.).\n"
        "— Я всегда добавлю «Правовые основания», если они есть в базе."
    )

@dp.message(Command("ask"))
async def cmd_ask(m: Message, command: CommandObject):
    q = (command.args or "").strip()
    if not q:
        await m.answer("Напиши вопрос после команды /ask. Пример: /ask Какие документы нужны для продажи квартиры?")
        return

    await m.chat.do("typing")
    answer = await generate_answer(q)
    await m.answer(
        f"<b>Вопрос:</b> {q}\n\n"
        f"{answer}\n\n"
        "<i>Ответ носит информационный характер и не является юридической консультацией.</i>"
    )

# Фоллбек: любой текст как вопрос
@dp.message(F.text.len() > 3)
async def any_text(m: Message):
    q = m.text.strip()
    await m.chat.do("typing")
    answer = await generate_answer(q)
    await m.answer(
        f"<b>Вопрос:</b> {q}\n\n"
        f"{answer}\n\n"
        "<i>Ответ носит информационный характер и не является юридической консультацией.</i>"
    )

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
