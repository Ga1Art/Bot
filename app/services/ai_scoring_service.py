from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import requests

from app.core.config import get_settings
from app.schemas.collector import LeadCreate


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AiLeadAssessment:
    score: int
    reason: str
    recommended_action: str
    tags: list[str]
    risk_tags: list[str]
    model: str
    analyzed_at: datetime


class AiScoringService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def is_enabled(self) -> bool:
        return bool(self.settings.enable_ai_scoring and self._api_key())

    def assess(self, item: LeadCreate, base_score: int) -> AiLeadAssessment | None:
        if not self.is_enabled() or base_score < self.settings.ai_min_base_score:
            return None
        try:
            return self._assess_chat_completions(item=item, base_score=base_score)
        except Exception as exc:
            logger.warning("AI lead assessment failed", extra={"source": item.source_name, "error": str(exc)[:300]})
            return None

    def _assess_chat_completions(self, item: LeadCreate, base_score: int) -> AiLeadAssessment | None:
        payload = {
            "model": self._model(),
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"Ты оцениваешь заявки для компании. Профиль компании: {self.settings.company_profile_text} "
                        "Если лид прямо просит изготовление, монтаж, демонтаж, застройку или дизайн "
                        "выставочного стенда, это сильное соответствие и обычно score 80-100. "
                        "Не занижай оценку из-за отсутствия бюджета, если профиль работ точный. "
                        "Не придумывай факты: площадь, бюджет, сроки, площадку и требования можно "
                        "упоминать только если они явно есть во входных данных. "
                        "Верни только JSON без markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": (
                                "Оцени релевантность лида от 0 до 100. Учитывай профиль работ, "
                                "город/регион, бюджет, дедлайн, качество заказчика и риски. "
                                "Рубрика: 85-100 точный заказ на стенд/застройку/монтаж; "
                                "65-84 профильная рекламная конструкция или оформление зоны; "
                                "40-64 возможно релевантно, но мало деталей; "
                                "0-39 не наш профиль. "
                                "Не завышай оценку отраслевым выставкам, если там нет заказа на стенд."
                            ),
                            "expected_json": {
                                "score": "integer 0..100",
                                "reason": "short Russian explanation, max 300 chars",
                                "recommended_action": "hot|queue|manual_review|hide",
                                "tags": ["profile tags"],
                                "risk_tags": ["risk tags"],
                            },
                            "lead": self._lead_payload(item, base_score),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        session = requests.Session()
        # Some local/server environments contain broken proxy variables.
        # AI calls should behave like collectors and connect directly by default.
        session.trust_env = False
        response = self._post_with_retries(session, payload)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = self._parse_json(content)
        score = max(0, min(100, int(parsed.get("score", 0))))
        source_text = json.dumps(self._lead_payload(item, base_score), ensure_ascii=False).lower()
        return AiLeadAssessment(
            score=score,
            reason=self._sanitize_reason(str(parsed.get("reason") or ""), source_text)[:500],
            recommended_action=str(parsed.get("recommended_action") or "manual_review")[:64],
            tags=self._string_list(parsed.get("tags")),
            risk_tags=self._string_list(parsed.get("risk_tags")),
            model=self._model(),
            analyzed_at=datetime.utcnow(),
        )

    def score_adjustment(self, assessment: AiLeadAssessment | None) -> int:
        if assessment is None:
            return 0
        return self.score_adjustment_from_score(assessment.score)

    def score_adjustment_from_score(self, score: int | None) -> int:
        if score is None:
            return 0
        # Convert 0..100 into a bounded bonus/penalty. AI nudges the score, it does not replace rules.
        centered = max(0, min(100, score)) - 50
        return round(centered * max(0, self.settings.ai_score_weight) / 50)

    def _lead_payload(self, item: LeadCreate, base_score: int) -> dict:
        return {
            "source_name": item.source_name,
            "title": item.title,
            "description": item.description,
            "customer_name": item.customer_name,
            "region": item.region,
            "city": item.city,
            "budget_max": self._safe_decimal(item.budget_max),
            "deadline_at": item.deadline_at.isoformat() if item.deadline_at else None,
            "keywords_matched": item.keywords_matched,
            "base_score": base_score,
            "url": item.url,
        }

    def _post_with_retries(self, session: requests.Session, payload: dict) -> requests.Response:
        url = f"{self._base_url().rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }
        last_response: requests.Response | None = None
        for attempt in range(3):
            response = session.post(url, headers=headers, json=payload, timeout=30)
            last_response = response
            if response.status_code not in {429, 500, 502, 503, 504}:
                return response
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
        return last_response

    def _api_key(self) -> str:
        if self.settings.ai_provider == "gemini":
            return self.settings.gemini_api_key or self.settings.openai_api_key
        return self.settings.openai_api_key

    def _base_url(self) -> str:
        if self.settings.ai_provider == "gemini":
            return self.settings.gemini_base_url
        return self.settings.openai_base_url

    def _model(self) -> str:
        if self.settings.ai_provider == "gemini":
            return self.settings.gemini_model
        return self.settings.openai_model

    def _safe_decimal(self, value: Decimal | None) -> float | None:
        return float(value) if value is not None else None

    def _parse_json(self, content: str) -> dict:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]
        return json.loads(text)

    def _string_list(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item)[:80] for item in value if str(item).strip()][:10]

    def _sanitize_reason(self, reason: str, source_text: str) -> str:
        if not reason:
            return ""
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", reason.replace("\n", " ")) if part.strip()]
        filtered: list[str] = []
        for sentence in sentences:
            lowered = sentence.lower()
            if "площад" in lowered and "площад" not in source_text:
                continue
            if "нестандарт" in lowered and "нестандарт" not in source_text:
                continue
            filtered.append(sentence)
        return " ".join(filtered)
