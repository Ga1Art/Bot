from decimal import Decimal


def render_lead_card(
    title: str,
    region: str | None,
    url: str,
    priority: str,
    source_name: str,
    lead_id: str | None = None,
    customer_name: str | None = None,
    budget_max: Decimal | float | None = None,
    deadline_at: str | None = None,
    score: int | None = None,
    base_score: int | None = None,
    learned_adjustment: int | None = None,
    learned_reason: str | None = None,
    ai_score: int | None = None,
    ai_reason: str | None = None,
    ai_recommended_action: str | None = None,
    is_hot_prospect: bool = False,
) -> str:
    lines = [f"[{priority}] {title}"]
    if lead_id:
        lines.append(f"ID: {lead_id}")
    if is_hot_prospect:
        lines.append("Горячий лид: да")
    lines.extend(
        [
            f"Источник: {source_name}",
            f"Регион: {region or '-'}",
        ]
    )
    if customer_name:
        lines.append(f"Заказчик/организатор: {customer_name}")
    if budget_max is not None:
        lines.append(f"Бюджет: {format_money(budget_max)}")
    if deadline_at:
        lines.append(f"Дедлайн: {deadline_at}")
    if score is not None:
        lines.append(f"Скоринг: {score}")
    if base_score is not None and base_score != score:
        lines.append(f"Базовый скоринг: {base_score}")
    if learned_adjustment:
        sign = "+" if learned_adjustment > 0 else ""
        lines.append(f"Learning: {sign}{learned_adjustment}")
        if learned_reason:
            lines.append(f"Learning-причина: {learned_reason}")
    if ai_score is not None:
        lines.append(f"AI-score: {ai_score}")
        if ai_recommended_action:
            lines.append(f"AI-рекомендация: {ai_recommended_action}")
        if ai_reason:
            lines.append(f"AI: {ai_reason}")
    lines.append(f"Ссылка: {url}")
    return "\n".join(lines)


def format_money(value: Decimal | float) -> str:
    number = float(value)
    formatted = f"{number:,.2f}".replace(",", " ").replace(".", ",")
    if formatted.endswith(",00"):
        formatted = formatted[:-3]
    return f"{formatted} руб."
