import pandas as pd


def test_a_trajectory_back_at_the_steady_state_has_no_time_labels_on_it():
    from analysis.figures.gnn import labelled_times
    # Away from the steady state at 106 and 112 s, back on it at 120 s, a hair from the 106 s point at 128 s.
    orbit = pd.DataFrame({'pc1': [0.0, 0.0, 1.0, 1.0, 0.01, 1.01], 'pc2': [0.0, 0.0, 0.0, 1.0, 0.0, 0.01]},
                         index=[90.0, 100.0, 106.0, 112.0, 120.0, 128.0])
    assert labelled_times(orbit, (0.0, 0.0)) == [106.0, 112.0]


def test_a_trajectory_far_from_the_steady_state_has_every_time_label():
    from analysis.figures.gnn import ORBIT_LABEL_TIMES_S, labelled_times
    orbit = pd.DataFrame({'pc1': [0.0, 1.0, 0.0, 1.0, 0.5], 'pc2': [0.0, 0.0, 1.0, 1.0, 0.5]},
                         index=[100.0, *ORBIT_LABEL_TIMES_S])
    assert labelled_times(orbit, (0.0, 0.0)) == list(ORBIT_LABEL_TIMES_S)
