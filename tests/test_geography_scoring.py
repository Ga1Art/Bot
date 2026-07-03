from app.normalizers.region import geography_priority_weight, is_target_geography
from app.scoring.engine import explain_score_lead


def test_geography_priority_decreases_with_distance_from_moscow() -> None:
    assert geography_priority_weight("Москва") > geography_priority_weight("Санкт-Петербург")
    assert geography_priority_weight("Санкт-Петербург") > geography_priority_weight("Казань")
    assert geography_priority_weight("Казань") > geography_priority_weight("Екатеринбург")
    assert geography_priority_weight("Новосибирск") == 0


def test_european_regions_are_target_geography() -> None:
    assert is_target_geography("Московская область")
    assert is_target_geography("Краснодарский край")
    assert is_target_geography("Республика Татарстан")
    assert not is_target_geography("Новосибирская область")


def test_score_uses_geography_weight() -> None:
    title = "Застройка стенда на выставке"
    moscow = explain_score_lead(title=title, description="", region="Москва", budget_max=None)
    kazan = explain_score_lead(title=title, description="", region="Республика Татарстан", budget_max=None)
    unknown = explain_score_lead(title=title, description="", region="Новосибирская область", budget_max=None)

    assert moscow.score > kazan.score > unknown.score
