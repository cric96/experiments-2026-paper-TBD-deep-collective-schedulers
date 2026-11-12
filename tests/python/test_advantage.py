"""The advantage of the learned methods over the tuned baselines is measured at the same rounds."""
import numpy as np
import pandas as pd

from analysis.campaign import gains
from analysis.figures.advantage import advantage_3d_figure, advantage_at_betas
from experiments.campaign import BASELINES, CORE, RANGE_BASELINES

FINAL = 'legacy-full-range'


def curves(baseline_low: float) -> pd.DataFrame:
    """A learned method at 0.25, 0.5 and 1 round per node and second with gains 20, 40 and 30%; Trickle at 30% and
    the gradual controller at 20% from `baseline_low` to 2 rounds per node and second."""
    learned = [dict(variant=FINAL, algorithm=0, beta=beta, roundsPerNodeSecond=rounds, gain=gain)
               for beta, rounds, gain in ((.2, .25, .2), (.6, .5, .4), (.95, 1., .3))]
    baselines = [dict(variant=RANGE_BASELINES, algorithm=algorithm, beta=beta, roundsPerNodeSecond=rounds, gain=gain)
                 for algorithm, gain in zip(BASELINES, (.3, .2)) for beta, rounds in ((.2, baseline_low), (.95, 2.))]
    return pd.DataFrame(learned + baselines)


def test_the_learned_method_is_compared_with_the_better_baseline_at_its_own_rounds():
    advantage = advantage_at_betas(curves(.2), FINAL, 0, RANGE_BASELINES)
    assert np.allclose(advantage.to_numpy(), [-10, 10, 0]) and list(advantage.index) == [.2, .6, .95]


def test_a_baseline_a_little_slower_than_the_learned_method_is_compared_at_its_slowest():
    assert np.allclose(advantage_at_betas(curves(.28), FINAL, 0, RANGE_BASELINES).to_numpy(), [-10, 10, 0])


def test_no_comparison_where_the_baselines_do_not_reach():
    advantage = advantage_at_betas(curves(.4), FINAL, 0, RANGE_BASELINES)
    assert np.isnan(advantage[.2]) and np.allclose(advantage[[.6, .95]].to_numpy(), [10, 0])


def test_the_3d_figure_renders(tmp_path):
    from test_campaign_figures import FULL_RANGE, synthetic_episodes
    data = gains(synthetic_episodes(FULL_RANGE, RANGE_BASELINES))
    advantage_3d_figure(data, tmp_path, FULL_RANGE, CORE, RANGE_BASELINES)
    assert (tmp_path / 'advantage_3d.pdf').exists()


def test_the_map_writes_the_advantage_in_whole_signed_points():
    from analysis.figures.advantage import cell_text
    assert [cell_text(value) for value in (9.2, -16.1, 0.3, float('nan'))] == ['+9', '−16', '0', '–']


def test_the_map_renders(tmp_path):
    from analysis.figures.advantage import advantage_map_figure
    from test_campaign_figures import FULL_RANGE, synthetic_episodes
    data = gains(synthetic_episodes(FULL_RANGE, RANGE_BASELINES))
    advantage_map_figure(data, tmp_path, FULL_RANGE, CORE, RANGE_BASELINES)
    assert (tmp_path / 'advantage_map.pdf').exists()
