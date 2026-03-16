from ifcb_classify.sweep import generate_sweep_runs


def test_single_param():
    runs = generate_sweep_runs({"lr": [0.01, 0.001]})
    assert len(runs) == 2
    assert runs[0].lr == 0.01
    assert runs[1].lr == 0.001


def test_cartesian_product():
    runs = generate_sweep_runs({"lr": [0.01, 0.001], "batch_size": [32, 64]})
    assert len(runs) == 4
    combos = {(r.lr, r.batch_size) for r in runs}
    assert combos == {(0.01, 32), (0.01, 64), (0.001, 32), (0.001, 64)}


def test_single_value_params():
    runs = generate_sweep_runs({"model": ["resnet50"], "lr": [0.01]})
    assert len(runs) == 1
    assert runs[0].model == "resnet50"


def test_empty_params():
    runs = generate_sweep_runs({})
    assert len(runs) == 1
