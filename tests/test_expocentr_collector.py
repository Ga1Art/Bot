from app.collectors.events.expocentr import ExpocentrCollector


def test_parse_html_extracts_expocentr_event() -> None:
    html = """
    <html>
      <body>
        <div>Собственные выставки РЕКЛАМА-2026 33-я международная специализированная выставка. Технологии и услуги для производителей и заказчиков рекламы МВЦ «Крокус Экспо», павильоны 1, зал 4 12+</div>
        <div>27.10.2026—29.10.2026</div>
      </body>
    </html>
    """

    items = ExpocentrCollector().parse_html(html)

    assert len(items) == 1
    assert items[0].title == "РЕКЛАМА-2026"
    assert items[0].customer_name == 'АО "ЭКСПОЦЕНТР"'
    assert items[0].deadline_at is not None
    assert items[0].deadline_at.strftime("%d.%m.%Y") == "29.10.2026"
