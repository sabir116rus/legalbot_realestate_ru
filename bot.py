
import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, BotCommand, MenuButtonCommands
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

    dp.workflow_data.update(
        answer_service=answer_service,
        interaction_logger=interaction_logger,
    )

@dp.message(Command("start"))
async def cmd_start(m: Message):
    await m.answer(
        "👋 <b>Привет! Я LegalBot</b> — ассистент по вопросам недвижимости.\n\n"
        "🧭 <b>Чем могу помочь:</b>\n"
        "• Покупка и продажа жилья\n"
        "• Аренда и найм\n"
        "• Ипотека, маткапитал, субсидии\n"
        "• Регистрация прав, Росреестр, ЭЦП\n"
        "• Земля, строительство, долевое участие\n\n"
        "📌 <b>Как задать вопрос:</b>\n"
        "• Просто опиши ситуацию или задай вопрос текстом\n"
        "• Загляни в /help за примерами\n\n"
        "<i>Ответы носят информационный характер и не являются юридической консультацией.</i>"
    )

@dp.message(Command("help"))
async def cmd_help(m: Message):
    await m.answer(
        "Как задать вопрос:\n"
        "• Какие документы нужны для продажи квартиры?\n"
        "• Чем отличается аренда и найм?\n\n"
        "Подсказки:\n"
        "— Пиши конкретно и добавляй детали (цель, статус объекта, ипотека и т. д.).\n"
        "— Я всегда добавлю «Правовые основания», если они есть в базе."
    )

# Фоллбек: любой текст как вопрос
@dp.message(F.text & (F.text.len() > 3) & ~F.text.startswith("/"))
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

async def setup_bot_menu() -> None:
    await bot.set_my_commands(
        [
            BotCommand(command="start", description="Начать работу с ботом"),
            BotCommand(command="help", description="Получить подсказки"),
        ]
    )
    await bot.set_chat_menu_button(menu_button=MenuButtonCommands())

async def main():
    setup_services()
    await setup_bot_menu()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
