from app.collectors.tenders.b2b_center import B2BCenterCollector
from app.collectors.tenders.bidzaar import BidzaarCollector
from app.collectors.tenders.fabrikant import FabrikantCollector
from app.collectors.tenders.rostender import RostenderCollector
from app.collectors.tenders.synapse import SynapseCollector
from app.core.time import moscow_now_naive, moscow_tomorrow_start_naive


def test_fabrikant_parse_html_extracts_order() -> None:
    html = """
    <html>
      <body>
        <div>
          <div>МПДО № 5545729 -1 Коммерческие закупки</div>
          <div>
            <a href="https://fabrikant.ru/v2/trades/procedure/view/5w-7_vzOkvZqoK3lmeYssg">
              Оказание комплекса услуг по застройке выставочного стенда Ханты-Мансийского автономного округа - Югры
              на Международной промышленной выставке ИННОПРОМ (г. Екатеринбург)
            </a>
            Организатор ФОНД РАЗВИТИЯ ЮГРЫ
            Дата публикации 08.06.2026 15:34
            Дата окончания приёма заявок 19.06.2099 05:00
            14 953 540,00 RUB начальная цена
          </div>
        </div>
      </body>
    </html>
    """

    items = FabrikantCollector(keywords=["выставочный стенд"]).parse_html(html, "выставочный стенд")

    assert len(items) == 1
    assert items[0].source_name == "fabrikant"
    assert items[0].external_id == "5w-7_vzOkvZqoK3lmeYssg"
    assert "застройке выставочного стенда" in items[0].title
    assert items[0].customer_name == "ФОНД РАЗВИТИЯ ЮГРЫ"
    assert items[0].city == "Екатеринбург"
    assert items[0].region == "Свердловская область"
    assert str(items[0].budget_max) == "14953540.00"
    assert items[0].published_at is not None
    assert items[0].deadline_at is not None


def test_b2b_center_parse_html_extracts_order() -> None:
    html = """
    <html>
      <body>
        <table class="search-results">
          <tr>
            <td>Рекламные услуги</td>
            <td>
              <a href="/market/vypolnenie-rabot-po-sozdaniiu-vystavochnogo-stenda/tender-4498502/">
                Предварительный квалификационный отбор № 4498502
                Выполнение работ по созданию выставочного стенда
                Лот №1 Выполнение работ по созданию выставочного стенда
              </a>
            </td>
            <td>АО "ОКБ "АСТРОН"</td>
            <td>23.06.2026 11:29</td>
            <td>01.07.2099 09:00</td>
          </tr>
        </table>
      </body>
    </html>
    """

    items = B2BCenterCollector(keywords=["выставочный стенд"]).parse_html(html, "выставочный стенд")

    assert len(items) == 1
    assert items[0].source_name == "b2b_center"
    assert items[0].external_id == "4498502"
    assert "созданию выставочного стенда" in items[0].title
    assert items[0].customer_name == 'АО "ОКБ "АСТРОН"'
    assert items[0].published_at is not None
    assert items[0].deadline_at is not None


def test_fabrikant_parse_html_skips_finished_and_expired_orders() -> None:
    html = """
    <html>
      <body>
        <div>
          <a href="https://fabrikant.ru/v2/trades/procedure/view/closed-order">
            Оказание услуг по застройке выставочного стенда
          </a>
          Закончено
          Организатор ООО Тест
          Дата публикации 08.06.2026 15:34
          Дата окончания приёма заявок 19.06.2099 05:00
        </div>
        <div>
          <a href="https://fabrikant.ru/v2/trades/procedure/view/expired-order">
            Оказание услуг по застройке выставочного стенда
          </a>
          Организатор ООО Тест
          Дата публикации 08.06.2026 15:34
          Дата окончания приёма заявок 19.06.2026 05:00
        </div>
      </body>
    </html>
    """

    items = FabrikantCollector(keywords=["выставочный стенд"]).parse_html(html, "выставочный стенд")

    assert items == []


def test_b2b_center_parse_html_skips_expired_order() -> None:
    html = """
    <html>
      <body>
        <table class="search-results">
          <tr>
            <td>Рекламные услуги</td>
            <td>
              <a href="/market/vypolnenie-rabot-po-sozdaniiu-vystavochnogo-stenda/tender-4498502/">
                Выполнение работ по созданию выставочного стенда
              </a>
            </td>
            <td>АО "ОКБ "АСТРОН"</td>
            <td>23.06.2026 11:29</td>
            <td>01.07.2025 09:00</td>
          </tr>
        </table>
      </body>
    </html>
    """

    items = B2BCenterCollector(keywords=["выставочный стенд"]).parse_html(html, "выставочный стенд")

    assert items == []


def test_b2b_center_parse_html_skips_today_deadline() -> None:
    today = moscow_now_naive().strftime("%d.%m.%Y 23:59")
    html = f"""
    <html>
      <body>
        <table class="search-results">
          <tr>
            <td>Рекламные услуги</td>
            <td>
              <a href="/market/vypolnenie-rabot-po-sozdaniiu-vystavochnogo-stenda/tender-4498503/">
                Выполнение работ по созданию выставочного стенда
              </a>
            </td>
            <td>АО "ОКБ "АСТРОН"</td>
            <td>23.06.2026 11:29</td>
            <td>{today}</td>
          </tr>
        </table>
      </body>
    </html>
    """

    items = B2BCenterCollector(keywords=["выставочный стенд"]).parse_html(html, "выставочный стенд")

    assert items == []


def test_b2b_center_parse_html_keeps_tomorrow_deadline() -> None:
    tomorrow = moscow_tomorrow_start_naive().strftime("%d.%m.%Y 09:00")
    html = f"""
    <html>
      <body>
        <table class="search-results">
          <tr>
            <td>Рекламные услуги</td>
            <td>
              <a href="/market/vypolnenie-rabot-po-sozdaniiu-vystavochnogo-stenda/tender-4498504/">
                Выполнение работ по созданию выставочного стенда
              </a>
            </td>
            <td>АО "ОКБ "АСТРОН"</td>
            <td>23.06.2026 11:29</td>
            <td>{tomorrow}</td>
          </tr>
        </table>
      </body>
    </html>
    """

    items = B2BCenterCollector(keywords=["выставочный стенд"]).parse_html(html, "выставочный стенд")

    assert len(items) == 1
    assert items[0].external_id == "4498504"


def test_bidzaar_parse_payload_extracts_order() -> None:
    tomorrow = moscow_tomorrow_start_naive().strftime("%Y-%m-%d")
    payload = {
        "items": [
            {
                "id": "019f220e-abaa-749e-9dbc-34cc87991421",
                "number": "348-878",
                "name": "Оказание услуг по оформлению выставочных зон на выездных и спортивных мероприятиях",
                "companyName": "НИКАМЕД",
                "publishDate": "2026-07-02T09:44:31.995436Z",
                "acceptanceEndDate": f"{tomorrow}T09:00:00Z",
                "procedureType": 1,
                "deliveryAddresses": [
                    {
                        "city": "Павловская Слобода",
                        "region": "Московская обл",
                        "comment": "Склад заказчика",
                    }
                ],
            }
        ],
        "totalCount": 1,
    }

    items = BidzaarCollector(keywords=["выставочный стенд"]).parse_payload(payload, "выставочный стенд")

    assert len(items) == 1
    assert items[0].source_name == "bidzaar"
    assert items[0].external_id == "019f220e-abaa-749e-9dbc-34cc87991421"
    assert "оформлению выставочных зон" in items[0].title
    assert items[0].customer_name == "НИКАМЕД"
    assert items[0].city == "Павловская Слобода"
    assert items[0].region == "Московская обл"
    assert items[0].published_at is not None
    assert items[0].deadline_at is not None


def test_bidzaar_parse_payload_skips_sales_and_today_deadline() -> None:
    today = moscow_now_naive().strftime("%Y-%m-%d")
    tomorrow = moscow_tomorrow_start_naive().strftime("%Y-%m-%d")
    payload = {
        "items": [
            {
                "id": "sale-order",
                "number": "349-797",
                "name": "Изготовление выставочного стенда",
                "companyName": "Тест",
                "acceptanceEndDate": f"{tomorrow}T09:00:00Z",
                "procedureType": 2,
                "deliveryAddresses": [{"city": "Москва", "region": "Москва"}],
            },
            {
                "id": "today-order",
                "number": "349-798",
                "name": "Изготовление выставочного стенда",
                "companyName": "Тест",
                "acceptanceEndDate": f"{today}T20:59:00Z",
                "procedureType": 1,
                "deliveryAddresses": [{"city": "Москва", "region": "Москва"}],
            },
        ],
        "totalCount": 2,
    }

    items = BidzaarCollector(keywords=["выставочный стенд"]).parse_payload(payload, "выставочный стенд")

    assert items == []


def test_rostender_parse_html_extracts_order() -> None:
    html = """
    <html>
      <body>
        <article class="tender-row row">
          <span class="tender__number">Тендер №93479901</span>
          <span class="tender__date-start">от 01.07.26</span>
          <a href="/region/moskva/93479901-tender-izgotovlenie-vystavochnogo-stenda">
            Изготовление выставочного стенда для международной выставки
          </a>
          <span>Окончание (МСК) 20.07.2099</span>
          <a href="/region/moskva">Закупки в регионе Москва</a>
          <span>Начальная цена 1 250 000,00 руб.</span>
          <span>Организатор ООО Ромашка</span>
        </article>
      </body>
    </html>
    """

    items = RostenderCollector(keywords=["выставочный стенд"]).parse_html(html, "выставочный стенд")

    assert len(items) == 1
    assert items[0].source_name == "rostender"
    assert items[0].external_id == "93479901"
    assert "Изготовление выставочного стенда" in items[0].title
    assert items[0].region == "Москва"
    assert items[0].city == "Москва"
    assert str(items[0].budget_max) == "1250000.00"
    assert items[0].customer_name == "ООО Ромашка"
    assert items[0].deadline_at is not None


def test_rostender_parse_html_ignores_masked_customer() -> None:
    html = """
    <html>
      <body>
        <article class="tender-row row">
          <span class="tender__number">Тендер №93479904</span>
          <a href="/region/moskva/93479904-tender-izgotovlenie-vystavochnogo-stenda">
            Изготовление выставочного стенда
          </a>
          <span>Окончание (МСК) 20.07.2099</span>
          <a href="/region/moskva">Закупки в регионе Москва</a>
          <span>Организатор ░░░░░░░░░░░ ░░░░░░░░░░░░░</span>
        </article>
      </body>
    </html>
    """

    items = RostenderCollector(keywords=["выставочный стенд"]).parse_html(html, "выставочный стенд")

    assert len(items) == 1
    assert items[0].customer_name is None


def test_rostender_parse_html_skips_closed_and_today_deadline() -> None:
    today = moscow_now_naive().strftime("%d.%m.%Y 23:59")
    html = f"""
    <html>
      <body>
        <article class="tender-row row">
          <a href="/region/moskva/93479902-tender-izgotovlenie-vystavochnogo-stenda">
            Изготовление выставочного стенда
          </a>
          <span>Окончание (МСК) 20.07.2099</span>
          <span>Завершено</span>
        </article>
        <article class="tender-row row">
          <a href="/region/moskva/93479903-tender-izgotovlenie-vystavochnogo-stenda">
            Изготовление выставочного стенда
          </a>
          <span>Окончание (МСК) {today}</span>
        </article>
      </body>
    </html>
    """

    items = RostenderCollector(keywords=["выставочный стенд"]).parse_html(html, "выставочный стенд")

    assert items == []


def test_synapse_parse_html_extracts_order() -> None:
    tomorrow = moscow_tomorrow_start_naive().strftime("%d.%m.%Y")
    html = f"""
    <html>
      <body>
        <div class="sp-tender-block">
          Закупка 2026006104335
          начальная цена 15 950 ₽
          <a class="sp-tb-title" href="/zakupki/aisgzspb/2026006104335--sanktpeterburg-okazanie-uslug-po-izgotovleniyu-mobilnogo">
            Оказание услуг по изготовлению мобильного информационного стенда для нужд СПб ГБУ ДО
          </a>
          заказчик СПБ ГБУ ДО СШОР ПО ПАРУСНОМУ СПОРТУ
          текущая закупка
          площадка АИС ГЗ СПб • способ отбора Закупка
          прием заявок 14:49 · 11.06.2026 — 14:00 · {tomorrow}
          регион Санкт-Петербург
        </div>
      </body>
    </html>
    """

    items = SynapseCollector(keywords=["выставочный стенд"]).parse_html(html, "выставочный стенд")

    assert len(items) == 1
    assert items[0].source_name == "synapse"
    assert items[0].external_id == "2026006104335"
    assert "мобильного информационного стенда" in items[0].title
    assert items[0].customer_name == "СПБ ГБУ ДО СШОР ПО ПАРУСНОМУ СПОРТУ"
    assert items[0].city == "Санкт-Петербург"
    assert str(items[0].budget_max) == "15950"
    assert items[0].deadline_at is not None


def test_synapse_parse_html_skips_closed_and_today_deadline() -> None:
    today = moscow_now_naive().strftime("%d.%m.%Y")
    html = f"""
    <html>
      <body>
        <div class="sp-tender-block">
          Закупка 341-423
          <a class="sp-tb-title" href="/zakupki/bidzaar/341-423--sanktpeterburg-d3-stend">
            3D стенд сход-развала с подъемником
          </a>
          прием заявок завершен
          прием заявок 14:49 · 11.06.2026 — 14:00 · 20.07.2099
        </div>
        <div class="sp-tender-block">
          Закупка 2026006104336
          <a class="sp-tb-title" href="/zakupki/test/2026006104336--sanktpeterburg-informacionniy-stend">
            Изготовление мобильного информационного стенда
          </a>
          текущая закупка
          прием заявок 14:49 · 11.06.2026 — 23:59 · {today}
        </div>
      </body>
    </html>
    """

    items = SynapseCollector(keywords=["выставочный стенд"]).parse_html(html, "выставочный стенд")

    assert items == []
