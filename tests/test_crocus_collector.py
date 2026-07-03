from app.collectors.events.crocus_expo import CrocusExpoCollector


def test_parse_html_extracts_event() -> None:
    html = """
    <html>
      <body>
        <h1>План выставок 2026</h1>
        <div>Test Expo 2026</div>
        <div>Международная выставка тестовых решений</div>
        <div>21 Января 2026 — 23 Января 2026</div>
        <div>Контактная информация:</div>
        <div>Сайт: www.testexpo.ru</div>
        <div>Организатор: Тест Органайзер</div>
        <div>Краткое описание:</div>
        <div>Сильная B2B выставка с экспонентами и деловой программой.</div>
      </body>
    </html>
    """

    items = CrocusExpoCollector().parse_html(html)

    assert len(items) == 1
    assert items[0].title == "Test Expo 2026"
    assert items[0].customer_name == "Тест Органайзер"
    assert items[0].region == "Московская область"
    assert items[0].url == "https://www.testexpo.ru"
