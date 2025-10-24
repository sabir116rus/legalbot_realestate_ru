
import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    Message,
    BotCommand,
    MenuButtonCommands,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    WebAppInfo,
)
from aiogram.client.default import DefaultBotProperties
from openai import AsyncOpenAI

from config import Config
from services import AnswerService, ConsultationLogger, InteractionLogger
from rag import KnowledgeBase


config = Config.load()

bot = Bot(
    token=config.telegram_bot_token,
    default=DefaultBotProperties(parse_mode="HTML"),
)
dp = Dispatcher(storage=MemoryStorage())

consented_users: set[int] = set()

WELCOME_MESSAGE = (
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


def _user_has_consented(user_id: int | None) -> bool:
    return user_id is not None and user_id in consented_users


async def _ensure_user_consent(message: Message) -> bool:
    if _user_has_consented(message.from_user.id if message.from_user else None):
        return True

    await message.answer(
        "Чтобы продолжить, подтвердите согласие, нажав кнопку «Я даю своё согласие…» в /start."
    )
    return False


class ConsultationForm(StatesGroup):
    full_name = State()
    contact = State()
    request = State()


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
    consultation_logger = ConsultationLogger(config.consultation_log_path)

    dp.workflow_data.update(
        answer_service=answer_service,
        interaction_logger=interaction_logger,
        consultation_logger=consultation_logger,
    )

@dp.message(Command("start"))
async def cmd_start(m: Message):
    user_id = m.from_user.id if m.from_user else None
    if _user_has_consented(user_id):
        await m.answer(WELCOME_MESSAGE)
        return

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="📄 Открыть политику",
                    web_app=WebAppInfo(url=config.privacy_policy_webapp_url),
                )
            ],
            [
                InlineKeyboardButton(
                    text="Я даю своё согласие…",
                    callback_data="consent_yes",
                ),
                InlineKeyboardButton(
                    text="Я не даю своё согласие",
                    callback_data="consent_no",
                ),
            ],
        ]
    )

    await m.answer(config.privacy_policy_message, reply_markup=keyboard)

@dp.message(Command("help"))
async def cmd_help(m: Message):
    if not await _ensure_user_consent(m):
        return

    await m.answer(
        "Как задать вопрос:\n"
        "• Какие документы нужны для продажи квартиры?\n"
        "• Чем отличается аренда и найм?\n\n"
        "Подсказки:\n"
        "— Пиши конкретно и добавляй детали (цель, статус объекта, ипотека и т. д.).\n"
        "— Я всегда добавлю «Правовые основания», если они есть в базе."
    )


@dp.message(Command("consultation"))
async def cmd_consultation(m: Message, state: FSMContext):
    if not await _ensure_user_consent(m):
        return

    await state.set_state(ConsultationForm.full_name)
    await m.answer(
        "📝 <b>Запрос консультации</b>\n"
        "Пожалуйста, укажи своё имя и фамилию."
    )


@dp.message(ConsultationForm.full_name, F.text)
async def consultation_full_name(m: Message, state: FSMContext):
    if not await _ensure_user_consent(m):
        await state.clear()
        return

    await state.update_data(full_name=m.text.strip())
    await state.set_state(ConsultationForm.contact)
    await m.answer("Как с тобой связаться? Оставь телефон, email или ник в мессенджере.")


@dp.message(ConsultationForm.contact, F.text)
async def consultation_contact(m: Message, state: FSMContext):
    if not await _ensure_user_consent(m):
        await state.clear()
        return

    await state.update_data(contact=m.text.strip())
    await state.set_state(ConsultationForm.request)
    await m.answer("Кратко опиши, какая помощь нужна.")


@dp.message(ConsultationForm.request, F.text)
async def consultation_request(
    m: Message,
    state: FSMContext,
    consultation_logger: ConsultationLogger,
):
    if not await _ensure_user_consent(m):
        await state.clear()
        return

    data = await state.get_data()
    await state.clear()

    consultation_logger.log(
        user_id=m.from_user.id,
        username=m.from_user.username,
        full_name=data.get("full_name", ""),
        contact=data.get("contact", ""),
        request=m.text.strip(),
    )

    await m.answer(
        "Спасибо! Заявка на консультацию сохранена. 👌\n"
        "Наш специалист свяжется с тобой по указанным контактам."
    )


# Фоллбек: любой текст как вопрос
@dp.message(F.text & (F.text.len() > 3) & ~F.text.startswith("/"))
async def any_text(
    m: Message,
    answer_service: AnswerService,
    interaction_logger: InteractionLogger,
):
    if not await _ensure_user_consent(m):
        return

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


@dp.callback_query(F.data == "consent_yes")
async def consent_yes(callback: CallbackQuery):
    user_id = callback.from_user.id if callback.from_user else None
    if user_id is not None:
        consented_users.add(user_id)

    await callback.answer("Согласие получено. Спасибо!")
    if callback.message:
        await callback.message.answer(WELCOME_MESSAGE)


@dp.callback_query(F.data == "consent_no")
async def consent_no(callback: CallbackQuery):
    user_id = callback.from_user.id if callback.from_user else None
    if user_id is not None:
        consented_users.discard(user_id)

    await callback.answer("Без согласия мы не можем продолжить работу.")
    if callback.message:
        await callback.message.answer(
            "Жаль, что мы не сможем продолжить. Если передумаешь, вернись в /start."
        )

async def setup_bot_menu() -> None:
    await bot.set_my_commands(
        [
            BotCommand(command="start", description="Начать работу с ботом"),
            BotCommand(command="help", description="Получить подсказки"),
            BotCommand(command="consultation", description="Оставить заявку на консультацию"),
        ]
    )
    await bot.set_chat_menu_button(menu_button=MenuButtonCommands())

async def main():
    setup_services()
    await setup_bot_menu()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
