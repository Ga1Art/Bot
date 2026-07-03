from app.collectors.tenders.eis import EisCollector


def test_parse_html_extracts_tender() -> None:
    html = """
    <html>
      <body>
        <div>№ извещения: 0173100000126000001</div>
        <div>Объект закупки: Изготовление и монтаж выставочного стенда</div>
        <div>Описание объекта закупки: Полный цикл работ по застройке стенда для отраслевой выставки.</div>
        <div>Начальная цена: 1 800 000,00 ₽</div>
        <div>Регион: Москва</div>
        <div>Заказчик: ООО "Тест Заказчик"</div>
        <div>Окончание подачи заявок: 24.06.2026 10:00</div>
        <a href="/epz/order/notice/ea20/view/common-info.html?regNumber=0173100000126000001">0173100000126000001</a>
      </body>
    </html>
    """

    items = EisCollector(keywords=["выставочный стенд"]).parse_html(html, "выставочный стенд")

    assert len(items) == 1
    assert items[0].source_type == "tender"
    assert items[0].external_id == "0173100000126000001"
    assert items[0].title == "Изготовление и монтаж выставочного стенда"
    assert items[0].customer_name == 'ООО "Тест Заказчик"'
    assert items[0].region == "Москва"
    assert str(items[0].budget_max) == "1800000.00"
    assert items[0].deadline_at is not None
    assert "regNumber=0173100000126000001" in items[0].url
