
import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandObject
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties
from openai import AsyncOpenAI

from config import Config
from services import AnswerService, InteractionLogger
from rag import KnowledgeBase


config = Config.load()

bot = Bot(
    token=config.telegram_bot_token,
    default=DefaultBotProperties(parse_mode="HTML"),
)
dp = Dispatcher()


def setup_services() -> None:
    knowledge_base = KnowledgeBase(str(config.knowledge_base_path))
    openai_client = AsyncOpenAI(api_key=config.openai_api_key)

    answer_service = AnswerService(
        knowledge_base=knowledge_base,
        openai_client=openai_client,
        model=config.openai_model,
        system_prompt=config.system_prompt,
        rag_top_k=config.rag_top_k,
    )

    interaction_logger = InteractionLogger(config.log_path)

    dp[AnswerService] = answer_service
    dp[InteractionLogger] = interaction_logger

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
async def cmd_ask(
    m: Message,
    command: CommandObject,
    answer_service: AnswerService,
    interaction_logger: InteractionLogger,
):
    q = (command.args or "").strip()
    if not q:
        await m.answer("Напиши вопрос после команды /ask. Пример: /ask Какие документы нужны для продажи квартиры?")
        return

    await m.chat.do("typing")
    answer_result = await answer_service.generate_answer(q)
    interaction_logger.log(
        user_id=m.from_user.id,
        username=m.from_user.username,
        question=q,
        answer=answer_result.text,
        top_score=answer_result.top_score,
        model=answer_service.model,
        status=answer_result.status,
    )
    await m.answer(
        f"<b>Вопрос:</b> {q}\n\n"
        f"{answer_result.text}\n\n"
        "<i>Ответ носит информационный характер и не является юридической консультацией.</i>"
    )

# Фоллбек: любой текст как вопрос
@dp.message(F.text.len() > 3)
async def any_text(
    m: Message,
    answer_service: AnswerService,
    interaction_logger: InteractionLogger,
):
    q = m.text.strip()
    await m.chat.do("typing")
    answer_result = await answer_service.generate_answer(q)
    interaction_logger.log(
        user_id=m.from_user.id,
        username=m.from_user.username,
        question=q,
        answer=answer_result.text,
        top_score=answer_result.top_score,
        model=answer_service.model,
        status=answer_result.status,
    )
    await m.answer(
        f"<b>Вопрос:</b> {q}\n\n"
        f"{answer_result.text}\n\n"
        "<i>Ответ носит информационный характер и не является юридической консультацией.</i>"
    )

async def main():
    setup_services()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
