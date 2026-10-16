from energy_forecast.registry import should_promote


def test_must_beat_baseline():
    assert not should_promote(0.06, baseline_mape=0.05, champion_mape=None).promote


def test_first_model_is_promoted_when_it_beats_baseline():
    assert should_promote(0.03, baseline_mape=0.05, champion_mape=None).promote


def test_requires_meaningful_improvement_over_champion():
    assert not should_promote(0.0299, 0.05, champion_mape=0.03, min_improvement=0.01).promote
    assert should_promote(0.029, 0.05, champion_mape=0.03, min_improvement=0.01).promote
