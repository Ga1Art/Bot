from telegram import Update
from telegram.ext import ContextTypes

from app.bot.keyboards import main_menu_keyboard


HELP_TEXT = (
    "Lead Radar bot is running.\n"
    "Используй кнопки меню ниже.\n\n"
    "Основные действия:\n"
    "- Новые: последние новые лиды\n"
    "- Очередь: рабочая очередь\n"
    "- Горячие: приоритетные A/B лиды\n"
    "- Мои: лиды, взятые тобой в работу\n"
    "- Дедлайны: ближайшие дедлайны\n"
    "- Сводка: краткая статистика\n"
    "- CSV очередь / CSV горячие: выгрузка файлом\n"
    "- Выгрузить в Sheets / Hot в Sheets: синхронизация с Google Sheets\n\n"
    "В карточке лида кнопки `Подходит`, `Не профиль`, `Далеко`, `Бюджет`, "
    "`Дедлайн`, `Дубль` обучают будущую выдачу. `AI-анализ` добавляет оценку "
    "нейронки, если AI включен в .env.\n\n"
    "Команды тоже работают: /new, /queue, /hot, /mine, /deadlines, /summary, "
    "/take <lead_id>, /export, /hotexport, /syncsheets, /synchot"
)


async def start_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(HELP_TEXT, reply_markup=main_menu_keyboard())
