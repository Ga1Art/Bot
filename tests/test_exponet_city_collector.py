from app.collectors.events.exponet_city import ExponetCityCollector


def test_parse_html_extracts_exponet_event() -> None:
    html = """
    <html>
      <body>
        <a href="/exhibitions/by-id/printech/printech2026/index.ru.html">Printech - 2026</a>
        <div>16.06</div>
        <div>19.06.2026</div>
        <div>Printech - 2026</div>
        <div>(г. Москва)</div>
        <div>Международная выставка оборудования, технологий и материалов для печатного и рекламного производства</div>
      </body>
    </html>
    """

    items = ExponetCityCollector("moscow", "Москва", "Москва").parse_html(html)

    assert len(items) == 1
    assert items[0].title == "Printech - 2026"
    assert items[0].region == "Москва"
    assert items[0].deadline_at is not None
