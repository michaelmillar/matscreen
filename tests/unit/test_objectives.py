from matscreen.screening.objectives import (
    Direction,
    Objective,
    bandgap_objective,
    stability_objective,
    uncertainty_objective,
)


def test_target_range_in_range():
    obj = bandgap_objective(1.0, 1.5)
    assert obj.score(1.2) == 0.0


def test_target_range_below():
    obj = bandgap_objective(1.0, 1.5)
    assert abs(obj.score(0.5) - 0.5) < 1e-10


def test_target_range_above():
    obj = bandgap_objective(1.0, 1.5)
    assert abs(obj.score(2.0) - 0.5) < 1e-10


def test_minimise():
    obj = stability_objective()
    assert obj.score(0.5) == 0.5
    assert obj.score(-1.0) == -1.0


def test_maximise():
    obj = Objective(name="test", direction=Direction.MAXIMISE)
    assert obj.score(5.0) == -5.0


def test_uncertainty_objective():
    obj = uncertainty_objective()
    assert obj.direction == Direction.MINIMISE
    assert obj.weight == 0.5
