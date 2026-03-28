import numpy as np

from matscreen.screening.pareto import crowding_distance, dominates, non_dominated_sort


def test_dominates_clear_case():
    assert dominates(np.array([1.0, 1.0]), np.array([2.0, 2.0]))
    assert not dominates(np.array([2.0, 2.0]), np.array([1.0, 1.0]))


def test_dominates_partial():
    assert dominates(np.array([1.0, 1.0]), np.array([1.0, 2.0]))
    assert not dominates(np.array([1.0, 2.0]), np.array([2.0, 1.0]))


def test_dominates_equal():
    assert not dominates(np.array([1.0, 1.0]), np.array([1.0, 1.0]))


def test_non_dominated_sort_simple():
    costs = np.array([
        [1.0, 3.0],
        [2.0, 2.0],
        [3.0, 1.0],
        [4.0, 4.0],
    ])
    fronts = non_dominated_sort(costs)
    assert len(fronts) == 2
    assert set(fronts[0]) == {0, 1, 2}
    assert fronts[1] == [3]


def test_non_dominated_sort_single_front():
    costs = np.array([
        [1.0, 3.0],
        [2.0, 2.0],
        [3.0, 1.0],
    ])
    fronts = non_dominated_sort(costs)
    assert len(fronts) == 1
    assert set(fronts[0]) == {0, 1, 2}


def test_non_dominated_sort_fully_dominated():
    costs = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ])
    fronts = non_dominated_sort(costs)
    assert len(fronts) == 3
    assert fronts[0] == [0]
    assert fronts[1] == [1]
    assert fronts[2] == [2]


def test_crowding_distance_two_points():
    costs = np.array([[1.0, 2.0], [3.0, 4.0]])
    distances = crowding_distance(costs, [0, 1])
    assert np.all(np.isinf(distances))


def test_crowding_distance_three_points():
    costs = np.array([
        [1.0, 3.0],
        [2.0, 2.0],
        [3.0, 1.0],
    ])
    distances = crowding_distance(costs, [0, 1, 2])
    assert np.isinf(distances[0])
    assert np.isinf(distances[2])
    assert np.isfinite(distances[1])
    assert distances[1] > 0


def test_non_dominated_sort_random(random_cost_matrix):
    fronts = non_dominated_sort(random_cost_matrix)
    all_indices = []
    for front in fronts:
        all_indices.extend(front)
    assert sorted(all_indices) == list(range(20))
