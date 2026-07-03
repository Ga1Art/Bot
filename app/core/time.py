from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo


MOSCOW_TZ = ZoneInfo("Europe/Moscow")


def moscow_now_naive() -> datetime:
    return datetime.now(MOSCOW_TZ).replace(tzinfo=None)


def moscow_tomorrow_start_naive() -> datetime:
    tomorrow = datetime.now(MOSCOW_TZ).date() + timedelta(days=1)
    return datetime.combine(tomorrow, time.min)
