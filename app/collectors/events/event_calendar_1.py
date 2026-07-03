from app.collectors.base import BaseCollector


class EventCalendarCollector(BaseCollector):
    source_name = "event_calendar_1"

    def fetch(self) -> list[dict]:
        # Replace this stub with parsing logic for a target event calendar.
        return []
