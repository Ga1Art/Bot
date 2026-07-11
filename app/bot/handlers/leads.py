import io
import asyncio
import logging

from telegram import Update
from telegram.error import RetryAfter, TelegramError
from telegram.ext import ContextTypes

from app.bot.handlers.common import HELP_TEXT
from app.bot.keyboards import (
    BTN_DEADLINES,
    BTN_EXPORT,
    BTN_HELP,
    BTN_HOT,
    BTN_HOT_EXPORT,
    BTN_MINE,
    BTN_NEW,
    BTN_QUEUE,
    BTN_SUMMARY,
    BTN_SYNC_HOT,
    BTN_SYNC_SHEETS,
    MENU_BUTTONS,
    lead_actions_keyboard,
    main_menu_keyboard,
)
from app.bot.templates import render_lead_card
from app.core.config import get_settings
from app.db.session import SessionLocal
from app.schemas.lead import LeadRead, LeadStatusUpdate
from app.services.export_service import ExportService
from app.services.google_sheets_service import GoogleSheetsService
from app.services.lead_service import LeadService
from app.services.runner_service import RunnerService

logger = logging.getLogger(__name__)


FEEDBACK_ACTIONS = {
    "accepted": ("in_work", "Менеджер отметил лид как подходящий"),
    "reject_not_profile": ("rejected", "Отклонено: не наш профиль"),
    "reject_far": ("rejected", "Отклонено: слишком далеко"),
    "reject_budget": ("rejected", "Отклонено: бюджет/экономика не подходят"),
    "reject_deadline": ("rejected", "Отклонено: неудобный или слишком близкий дедлайн"),
    "reject_duplicate": ("rejected", "Отклонено: дубль"),
    "reject_other": ("rejected", "Отклонено: другая причина"),
}

TELEGRAM_MESSAGE_SAFE_LIMIT = 3500


def _allowed_chat_ids(raw_chat_id: str) -> set[str]:
    return {item.strip() for item in raw_chat_id.split(",") if item.strip()}


def _is_allowed_chat(current_chat_id: int | str, configured_chat_id: str) -> bool:
    allowed_ids = _allowed_chat_ids(configured_chat_id)
    return not allowed_ids or str(current_chat_id) in allowed_ids


def _shorten_text(value: str, limit: int = 220) -> str:
    compact = " ".join(str(value).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _telegram_actor(update: Update) -> str:
    user = update.effective_user
    if not user:
        return "telegram"
    if user.username:
        return f"@{user.username}"
    full_name = " ".join(part for part in [user.first_name, user.last_name] if part).strip()
    return full_name or "telegram"


def _render_lead_message(lead: LeadRead, include_status: bool = False) -> str:
    text = render_lead_card(
        title=lead.title,
        region=lead.region,
        url=lead.url,
        priority=lead.priority,
        source_name=lead.source_name,
        lead_id=str(lead.id),
        customer_name=lead.customer_name,
        budget_max=lead.budget_max,
        deadline_at=lead.deadline_at.strftime("%d.%m.%Y %H:%M") if lead.deadline_at else None,
        score=lead.relevance_score,
        base_score=lead.base_relevance_score,
        learned_adjustment=lead.learned_score_adjustment,
        learned_reason=lead.learned_reason,
        ai_score=lead.ai_score,
        ai_reason=lead.ai_reason,
        ai_recommended_action=lead.ai_recommended_action,
        is_hot_prospect=lead.is_hot_prospect,
    )
    if include_status:
        text = f"{text}\nСтатус: {lead.status}"
    return text


def _safe_telegram_message(text: str) -> str:
    if len(text) <= TELEGRAM_MESSAGE_SAFE_LIMIT:
        return text
    suffix = "\n\n...карточка сокращена из-за лимита Telegram. Полная информация доступна по ссылке."
    return text[: TELEGRAM_MESSAGE_SAFE_LIMIT - len(suffix)].rstrip() + suffix


async def _reply_lead_cards(update: Update, leads: list[LeadRead], include_status: bool = False) -> int:
    if not update.message:
        return 0

    sent = 0
    failed = 0
    for lead in leads:
        text = _safe_telegram_message(_render_lead_message(lead, include_status=include_status))
        try:
            await update.message.reply_text(text, reply_markup=lead_actions_keyboard(str(lead.id)))
            sent += 1
        except RetryAfter as exc:
            await asyncio.sleep(float(exc.retry_after) + 0.5)
            try:
                await update.message.reply_text(text, reply_markup=lead_actions_keyboard(str(lead.id)))
                sent += 1
            except TelegramError as retry_exc:
                logger.warning("Lead card send failed after retry", extra={"lead_id": str(lead.id), "error": str(retry_exc)})
                failed += 1
        except TelegramError as exc:
            logger.warning("Lead card send failed", extra={"lead_id": str(lead.id), "error": str(exc)})
            failed += 1
        await asyncio.sleep(0.15)

    if failed:
        await update.message.reply_text(
            f"Не удалось отправить карточек: {failed}. Остальные лиды сохранены в базе.",
            reply_markup=main_menu_keyboard(),
        )
    return sent


def _top_dict_lines(title: str, items: dict[str, int], limit: int = 3) -> list[str]:
    if not items:
        return [f"{title}: нет данных"]
    lines = [f"{title}:"]
    for index, (name, count) in enumerate(list(items.items())[:limit], start=1):
        lines.append(f"{index}. {name} — {count}")
    return lines


async def today_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text("Команда /today пока подключена как заглушка.", reply_markup=main_menu_keyboard())


async def new_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    with SessionLocal() as db:
        leads = LeadService(db).latest_collected_new_leads()

    if not update.message:
        return

    if not leads:
        await update.message.reply_text("Новых лидов из последнего сбора сейчас нет.", reply_markup=main_menu_keyboard())
        return

    await update.message.reply_text(
        f"Показываю новые лиды из последнего сбора: {len(leads)} шт.",
        reply_markup=main_menu_keyboard(),
    )
    sent = await _reply_lead_cards(update, leads)
    if sent != len(leads):
        await update.message.reply_text(
            f"Отправлено карточек: {sent} из {len(leads)}.",
            reply_markup=main_menu_keyboard(),
        )


async def queue_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    with SessionLocal() as db:
        leads = LeadService(db).review_queue(limit=10)

    if not update.message:
        return

    if not leads:
        await update.message.reply_text("Очередь пуста.", reply_markup=main_menu_keyboard())
        return

    await _reply_lead_cards(update, leads, include_status=True)


async def hot_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    with SessionLocal() as db:
        leads = LeadService(db).hot_prospects(limit=10)

    if not update.message:
        return

    if not leads:
        await update.message.reply_text(
            "Горячих лидов A/B по приоритетным регионам сейчас нет.",
            reply_markup=main_menu_keyboard(),
        )
        return

    await _reply_lead_cards(update, leads, include_status=True)


async def deadlines_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    with SessionLocal() as db:
        stats = LeadService(db).hot_stats()

    if not update.message:
        return

    if not stats.top_deadlines:
        await update.message.reply_text(
            "По горячим лидам сейчас нет ближайших дедлайнов.",
            reply_markup=main_menu_keyboard(),
        )
        return

    lines = ["Ближайшие дедлайны по горячим лидам:"]
    for index, item in enumerate(stats.top_deadlines[:10], start=1):
        deadline = item.deadline_at.strftime("%d.%m.%Y %H:%M")
        region = item.region or "-"
        lines.append(f"{index}. [{item.priority}] {item.title}")
        lines.append(f"   Дедлайн: {deadline}")
        lines.append(f"   Регион: {region}")
        lines.append(f"   Источник: {item.source_name}")
        lines.append(f"   Скоринг: {item.relevance_score}")
        lines.append(f"   Ссылка: {item.url}")

    await update.message.reply_text("\n".join(lines), reply_markup=main_menu_keyboard())


async def summary_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    with SessionLocal() as db:
        stats = LeadService(db).hot_stats()

    if not update.message:
        return

    if stats.total == 0:
        await update.message.reply_text("Горячих лидов сейчас нет.", reply_markup=main_menu_keyboard())
        return

    lines = [
        "Сводка по горячим лидам:",
        f"Всего: {stats.total}",
        f"Новые: {stats.by_status.get('new', 0)}",
        f"В работе: {stats.by_status.get('in_work', 0)}",
        f"Приоритет A: {stats.by_priority.get('A', 0)}",
        f"Приоритет B: {stats.by_priority.get('B', 0)}",
        "",
    ]
    lines.extend(_top_dict_lines("Топ источники", stats.by_source))
    lines.append("")
    lines.extend(_top_dict_lines("Топ регионы", stats.by_region))
    if stats.top_deadlines:
        lines.append("")
        lines.append("Ближайшие дедлайны:")
        for item in stats.top_deadlines[:3]:
            deadline = item.deadline_at.strftime("%d.%m.%Y %H:%M")
            lines.append(f"- [{item.priority}] {item.title} — {deadline}")

    await update.message.reply_text("\n".join(lines), reply_markup=main_menu_keyboard())


async def mine_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    actor = _telegram_actor(update)
    with SessionLocal() as db:
        leads = LeadService(db).mine_leads(actor=actor, limit=10)

    if not leads:
        await update.message.reply_text(f"У {actor} сейчас нет лидов в работе.", reply_markup=main_menu_keyboard())
        return

    await _reply_lead_cards(update, leads, include_status=True)


async def take_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    if not context.args:
        await update.message.reply_text("Используй: /take <lead_id>", reply_markup=main_menu_keyboard())
        return

    lead_id = context.args[0].strip()
    actor = _telegram_actor(update)

    with SessionLocal() as db:
        service = LeadService(db)
        try:
            lead = service.update_status(
                lead_id=lead_id,
                payload=LeadStatusUpdate(
                    status="in_work",
                    actor=actor,
                    comment=f"Taken in work by {actor} via /take",
                ),
            )
        except Exception:
            await update.message.reply_text(
                "Не удалось взять лид в работу. Проверь `lead_id`.",
                reply_markup=main_menu_keyboard(),
            )
            return

    await update.message.reply_text(
        f"Лид взят в работу: {actor}\n\n{_render_lead_message(lead, include_status=True)}",
        reply_markup=main_menu_keyboard(),
    )


async def export_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    with SessionLocal() as db:
        leads = LeadService(db).review_queue(limit=200)

    await _send_csv_export(
        update=update,
        leads=leads,
        empty_message="Для экспорта пока нет лидов.",
        filename="lead_radar_queue.csv",
        caption="Экспорт текущей очереди лидов",
    )


async def hot_export_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    with SessionLocal() as db:
        leads = LeadService(db).hot_prospects(limit=500)

    await _send_csv_export(
        update=update,
        leads=leads,
        empty_message="Для экспорта hot prospects пока нет лидов.",
        filename="lead_radar_hot_prospects.csv",
        caption="Экспорт hot prospects",
    )


async def _send_csv_export(
    update: Update,
    leads: list[LeadRead],
    empty_message: str,
    filename: str,
    caption: str,
) -> None:
    if not update.message:
        return

    if not leads:
        await update.message.reply_text(empty_message, reply_markup=main_menu_keyboard())
        return

    csv_content = ExportService().leads_to_csv(leads)
    export_file = io.BytesIO(csv_content.encode("utf-8-sig"))
    export_file.name = filename
    await update.message.reply_document(
        document=export_file,
        filename=filename,
        caption=caption,
    )


async def sync_sheets_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    with SessionLocal() as db:
        leads = LeadService(db).review_queue(limit=500)

    await _sync_leads_to_sheets(
        update=update,
        leads=leads,
        empty_message="Для синхронизации пока нет лидов.",
        success_prefix="Синхронизация очереди завершена",
        range_name=get_settings().google_sheets_range,
    )


async def sync_hot_sheets_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    with SessionLocal() as db:
        leads = LeadService(db).hot_prospects(limit=500)

    await _sync_leads_to_sheets(
        update=update,
        leads=leads,
        empty_message="Для синхронизации hot prospects пока нет лидов.",
        success_prefix="Синхронизация hot prospects завершена",
        range_name=get_settings().google_sheets_hot_range,
    )


async def _sync_leads_to_sheets(
    update: Update,
    leads: list[LeadRead],
    empty_message: str,
    success_prefix: str,
    range_name: str,
) -> None:
    if not update.message:
        return

    if not leads:
        await update.message.reply_text(empty_message, reply_markup=main_menu_keyboard())
        return

    service = GoogleSheetsService()
    if not service.is_configured():
        await update.message.reply_text(
            "Google Sheets еще не настроен. Заполни GOOGLE_SHEETS_CREDENTIALS_FILE, "
            "GOOGLE_SHEETS_SPREADSHEET_ID и GOOGLE_SHEETS_RANGE.",
            reply_markup=main_menu_keyboard(),
        )
        return

    synced = service.sync_leads(leads, range_name=range_name)
    await update.message.reply_text(f"{success_prefix}. Строк выгружено: {synced}", reply_markup=main_menu_keyboard())


async def collect_now_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    settings = get_settings()
    current_chat_id = str(update.effective_chat.id)
    if not _is_allowed_chat(current_chat_id, settings.telegram_chat_id):
        await update.message.reply_text(
            "Команда ручного сбора доступна только в основном чате бота.\n"
            f"Текущий chat id: {current_chat_id}\n"
            "Чтобы разрешить этот чат, укажи его в TELEGRAM_CHAT_ID на сервере "
            "и перезапусти bot-контейнер.",
            reply_markup=main_menu_keyboard(),
        )
        return

    await update.message.reply_text(
        "Запускаю ручной сбор лидов. Это может занять пару минут...",
        reply_markup=main_menu_keyboard(),
    )

    runner = RunnerService(include_event_collectors=False)
    active_collectors = [collector.source_name for collector in runner.collectors]
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(runner.run_all),
            timeout=settings.manual_collect_timeout_seconds,
        )
    except asyncio.TimeoutError:
        await update.message.reply_text(
            "Ручной сбор не уложился в лимит времени и был отпущен в фон.\n"
            "Проверь /queue через пару минут. Если новые лиды не появились, посмотри логи bot/collector.",
            reply_markup=main_menu_keyboard(),
        )
        return

    lines = [
        "Ручной сбор завершен.",
        f"Активные источники: {', '.join(active_collectors) or '-'}",
        "",
    ]
    for source_name, found, saved, status in results:
        short_status = _shorten_text(status)
        lines.append(f"{source_name}: найдено {found}, новых {saved}, статус {short_status}")

    await update.message.reply_text("\n".join(lines), reply_markup=main_menu_keyboard())


async def chat_id_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    settings = get_settings()
    current_chat_id = str(update.effective_chat.id)
    configured = settings.telegram_chat_id or "не задан"
    access = "разрешен" if _is_allowed_chat(current_chat_id, settings.telegram_chat_id) else "не разрешен"
    await update.message.reply_text(
        "Информация о текущем чате:\n"
        f"chat id: {current_chat_id}\n"
        f"TELEGRAM_CHAT_ID: {configured}\n"
        f"Доступ к /collectnow: {access}",
        reply_markup=main_menu_keyboard(),
    )


async def menu_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    if text == BTN_NEW:
        await new_handler(update, context)
    elif text == BTN_QUEUE:
        await queue_handler(update, context)
    elif text == BTN_HOT:
        await hot_handler(update, context)
    elif text == BTN_MINE:
        await mine_handler(update, context)
    elif text == BTN_DEADLINES:
        await deadlines_handler(update, context)
    elif text == BTN_SUMMARY:
        await summary_handler(update, context)
    elif text == BTN_EXPORT:
        await export_handler(update, context)
    elif text == BTN_HOT_EXPORT:
        await hot_export_handler(update, context)
    elif text == BTN_SYNC_SHEETS:
        await sync_sheets_handler(update, context)
    elif text == BTN_SYNC_HOT:
        await sync_hot_sheets_handler(update, context)
    elif text == BTN_HELP:
        await update.message.reply_text(HELP_TEXT, reply_markup=main_menu_keyboard())
    elif text in MENU_BUTTONS:
        await update.message.reply_text("Эта кнопка пока не подключена.", reply_markup=main_menu_keyboard())


async def lead_action_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.callback_query:
        return

    query = update.callback_query
    await query.answer()

    try:
        status, lead_id = query.data.split(":", maxsplit=1)
    except (AttributeError, ValueError):
        await query.edit_message_text("Не удалось распознать действие.")
        return

    actor = _telegram_actor(update)

    if status == "ai":
        with SessionLocal() as db:
            service = LeadService(db)
            try:
                lead, message = service.analyze_with_ai(lead_id=lead_id)
            except Exception:
                await query.edit_message_text("Не удалось выполнить AI-анализ лида.")
                return
        await query.edit_message_text(f"{message}\n\n{_render_lead_message(lead, include_status=True)}")
        return

    with SessionLocal() as db:
        service = LeadService(db)
        try:
            if status in FEEDBACK_ACTIONS:
                new_status, comment = FEEDBACK_ACTIONS[status]
                lead = service.record_feedback(
                    lead_id=lead_id,
                    action_type=status,
                    status=new_status,
                    actor=actor,
                    comment=comment,
                )
            else:
                comment = f"Taken in work by {actor} via button" if status == "in_work" else None
                lead = service.update_status(
                    lead_id=lead_id,
                    payload=LeadStatusUpdate(status=status, actor=actor, comment=comment),
                )
        except Exception:
            await query.edit_message_text("Не удалось обновить статус лида.")
            return

    await query.edit_message_text(_render_lead_message(lead, include_status=True))
