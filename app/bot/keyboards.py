from telegram import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup


BTN_NEW = "Новые"
BTN_QUEUE = "Очередь"
BTN_HOT = "Горячие"
BTN_MINE = "Мои"
BTN_DEADLINES = "Дедлайны"
BTN_SUMMARY = "Сводка"
BTN_EXPORT = "CSV очередь"
BTN_HOT_EXPORT = "CSV горячие"
BTN_SYNC_SHEETS = "Выгрузить в Sheets"
BTN_SYNC_HOT = "Hot в Sheets"
BTN_HELP = "Помощь"


MENU_BUTTONS = {
    BTN_NEW,
    BTN_QUEUE,
    BTN_HOT,
    BTN_MINE,
    BTN_DEADLINES,
    BTN_SUMMARY,
    BTN_EXPORT,
    BTN_HOT_EXPORT,
    BTN_SYNC_SHEETS,
    BTN_SYNC_HOT,
    BTN_HELP,
}


def main_menu_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton(BTN_NEW), KeyboardButton(BTN_QUEUE), KeyboardButton(BTN_HOT)],
            [KeyboardButton(BTN_MINE), KeyboardButton(BTN_DEADLINES), KeyboardButton(BTN_SUMMARY)],
            [KeyboardButton(BTN_EXPORT), KeyboardButton(BTN_HOT_EXPORT)],
            [KeyboardButton(BTN_SYNC_SHEETS), KeyboardButton(BTN_SYNC_HOT)],
            [KeyboardButton(BTN_HELP)],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )


def lead_actions_keyboard(lead_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("В работу", callback_data=f"in_work:{lead_id}"),
                InlineKeyboardButton("Подходит", callback_data=f"accepted:{lead_id}"),
                InlineKeyboardButton("Позже", callback_data=f"new:{lead_id}"),
            ],
            [
                InlineKeyboardButton("AI-анализ", callback_data=f"ai:{lead_id}"),
                InlineKeyboardButton("Не профиль", callback_data=f"reject_not_profile:{lead_id}"),
            ],
            [
                InlineKeyboardButton("Далеко", callback_data=f"reject_far:{lead_id}"),
                InlineKeyboardButton("Бюджет", callback_data=f"reject_budget:{lead_id}"),
                InlineKeyboardButton("Дедлайн", callback_data=f"reject_deadline:{lead_id}"),
            ],
            [
                InlineKeyboardButton("Дубль", callback_data=f"reject_duplicate:{lead_id}"),
                InlineKeyboardButton("Другое", callback_data=f"reject_other:{lead_id}"),
            ]
        ]
    )
