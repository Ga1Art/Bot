from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters

from app.bot.handlers.common import start_handler
from app.bot.handlers.leads import (
    collect_now_handler,
    deadlines_handler,
    export_handler,
    hot_handler,
    hot_export_handler,
    lead_action_handler,
    menu_button_handler,
    mine_handler,
    new_handler,
    queue_handler,
    summary_handler,
    take_handler,
    sync_hot_sheets_handler,
    sync_sheets_handler,
    today_handler,
)
from app.core.config import get_settings


def build_bot_application() -> Application:
    settings = get_settings()
    application = Application.builder().token(settings.telegram_bot_token).build()
    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("today", today_handler))
    application.add_handler(CommandHandler("new", new_handler))
    application.add_handler(CommandHandler("queue", queue_handler))
    application.add_handler(CommandHandler("hot", hot_handler))
    application.add_handler(CommandHandler("mine", mine_handler))
    application.add_handler(CommandHandler("deadlines", deadlines_handler))
    application.add_handler(CommandHandler("summary", summary_handler))
    application.add_handler(CommandHandler("take", take_handler))
    application.add_handler(CommandHandler("export", export_handler))
    application.add_handler(CommandHandler("hotexport", hot_export_handler))
    application.add_handler(CommandHandler("syncsheets", sync_sheets_handler))
    application.add_handler(CommandHandler("synchot", sync_hot_sheets_handler))
    application.add_handler(CommandHandler("collectnow", collect_now_handler))
    application.add_handler(CallbackQueryHandler(lead_action_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, menu_button_handler))
    return application
