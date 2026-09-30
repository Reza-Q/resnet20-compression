import pytest

from resnet_compression.config import Config, default_pruning_rates


def test_defaults_are_valid():
    Config().validate()


def test_default_pruning_rates_is_one_to_ten_percent():
    assert default_pruning_rates() == [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]


def test_two_instances_do_not_share_the_pruning_rates_list():
    a, b = Config(), Config()
    a.pruning_rates.append(0.99)
    assert 0.99 not in b.pruning_rates


@pytest.mark.parametrize(
    "overrides",
    [
        {"pruning_rates": []},
        {"pruning_rates": [0.0]},
        {"pruning_rates": [1.0]},
        {"pruning_rates": [1.5]},
        {"initial_epochs": 0},
        {"finetune_epochs": 0},
        {"qat_epochs": 0},
        {"prune_iterations": 0},
        {"batch_size": 0},
        {"test_batch_size": 0},
        {"checkpoint_interval": 0.0},
        {"kd_alpha": -0.1},
        {"kd_alpha": 1.1},
        {"qat_backend": "not-a-backend"},
    ],
)
def test_invalid_settings_are_rejected(overrides):
    cfg = Config(**overrides)
    with pytest.raises(ValueError):
        cfg.validate()


def test_scenario_dir_and_results_path_are_distinct_per_scenario(tmp_path):
    cfg = Config(output_dir=str(tmp_path))
    dirs = {cfg.scenario_dir(s) for s in ("A", "B", "C", "D")}
    paths = {cfg.results_path(s) for s in ("A", "B", "C", "D")}
    assert len(dirs) == 4
    assert len(paths) == 4


def test_apply_smoke_test_shrinks_the_schedule():
    cfg = Config().apply_smoke_test()
    assert cfg.smoke_test is True
    assert cfg.initial_epochs == 1
    assert cfg.finetune_epochs == 1
    assert cfg.qat_epochs == 1
    assert cfg.pruning_rates == [0.05, 0.10]
    cfg.validate()


def test_make_dirs_creates_the_full_tree(tmp_path):
    cfg = Config(output_dir=str(tmp_path / "exp"))
    cfg.make_dirs()
    assert cfg.base_dir.is_dir()
    assert cfg.checkpoint_dir.is_dir()
    assert cfg.plots_dir.is_dir()
    for scenario in ("A", "B", "C", "D"):
        assert cfg.scenario_dir(scenario).is_dir()
