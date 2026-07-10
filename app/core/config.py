from functools import lru_cache
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _split_csv(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "dev"
    app_host: str = "127.0.0.1"
    app_port: int = 8010
    database_url: str
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    min_target_budget: int = 300000
    min_acceptable_budget: int = 100000
    priority_regions: Annotated[list[str], NoDecode] = ["Москва", "Московская область"]
    digest_hour_morning: int = 9
    digest_hour_evening: int = 16
    manual_collect_timeout_seconds: int = 300
    tender_search_max_pages: int = 2
    bidzaar_search_max_pages: int = 1
    rostender_search_max_pages: int = 1
    synapse_search_max_pages: int = 1
    eis_search_keywords: Annotated[list[str], NoDecode] = [
        "выставочный стенд",
        "изготовление выставочного стенда",
        "монтаж стенда",
        "застройка стенда",
        "оформление стенда",
        "оформление выставочной экспозиции",
        "брендирование стенда",
        "оформление выставочных зон",
        "брендированные элементы",
        "декорирование помещений",
        "оформление мероприятия",
        "экспозиционный стенд",
        "выставочное оборудование",
        "регистрационная стойка",
        "фотозона",
    ]
    rostender_search_keywords: Annotated[list[str], NoDecode] = [
        "выставочный стенд",
        "изготовление выставочного стенда",
        "монтаж выставочного стенда",
        "застройка стенда",
        "оформление стенда",
        "выставочное оборудование",
    ]
    eis_base_url: str = "https://zakupki.gov.ru/epz/order/extendedsearch/results.html"
    scoring_whitelist_keywords: Annotated[list[str], NoDecode] = [
        "выставочный стенд",
        "выставочная застройка",
        "застройка стенда",
        "оформление стенда",
        "оформление мероприятия",
        "экспозиция",
        "регистрационная стойка",
        "фотозона",
        "брендирование",
        "сценические конструкции",
        "навигация",
        "posm",
        "промостенд",
    ]
    scoring_blacklist_keywords: Annotated[list[str], NoDecode] = [
        "вакансия",
        "резюме",
        "обучение",
        "курс",
        "тендерное сопровождение",
        "юридические услуги",
        "бухгалтерские услуги",
        "аудит",
        "консалтинг",
    ]
    scoring_priority_a_threshold: int = 65
    scoring_priority_b_threshold: int = 42
    scoring_whitelist_weight: int = 6
    scoring_whitelist_cap: int = 18
    scoring_blacklist_weight: int = 12
    scoring_blacklist_cap: int = 36
    enable_ai_scoring: bool = False
    ai_provider: str = "gemini"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    ai_score_weight: int = 20
    ai_min_base_score: int = 35
    ai_daily_analysis_limit: int = 50
    ai_analysis_hour: int = 23
    ai_analysis_minute: int = 35
    enable_feedback_learning: bool = True
    feedback_learning_weight: int = 8
    feedback_learning_cap: int = 24
    enable_crocus_collector: bool = False
    enable_exponet_city_collectors: bool = False
    enable_expocentr_collector: bool = False
    enable_eis_collector: bool = False
    enable_b2b_center_collector: bool = True
    enable_bidzaar_collector: bool = True
    enable_fabrikant_collector: bool = True
    enable_rostender_collector: bool = True
    enable_synapse_collector: bool = True
    google_sheets_credentials_file: str = ""
    google_sheets_spreadsheet_id: str = ""
    google_sheets_range: str = "Queue!A1"
    google_sheets_hot_range: str = "Hot!A1"

    @field_validator(
        "priority_regions",
        "eis_search_keywords",
        "rostender_search_keywords",
        "scoring_whitelist_keywords",
        "scoring_blacklist_keywords",
        mode="before",
    )
    @classmethod
    def split_csv_fields(cls, value: str | list[str]) -> list[str]:
        return _split_csv(value)


@lru_cache
def get_settings() -> Settings:
    return Settings()
