from app.bot.app import build_bot_application
from app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    if not settings.telegram_bot_token or settings.telegram_bot_token == "replace_me":
        raise SystemExit("TELEGRAM_BOT_TOKEN is not configured in .env")
    application = build_bot_application()
    application.run_polling()


if __name__ == "__main__":
    main()
