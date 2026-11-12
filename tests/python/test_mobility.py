"""Independent seeds, rather than episodes, are the unit of mobility summaries."""
import pandas as pd
import pytest
from analysis.mobility import METRICS, load_episodes, summarize


def test_summarize_weights_seeds_equally():
    rows = []
    for seed, values in [(1, [0.0, 0.0, 0.0]), (2, [1.0])]:
        for value in values:
            rows.append(dict(algorithm=0, diffusion=1.0, tolerance=5.0, beta=.95,
                             sourceChanges=True, protocol="test", seed=seed, **{metric: value for metric in METRICS}))
    result = summarize(pd.DataFrame(rows)).iloc[0]
    assert result['seeds'] == 2
    assert result['toleratedError'] == .5
    assert result['toleratedError_ci95'] > 0


def test_load_excludes_training_and_retains_censoring(tmp_path):
    path = tmp_path / 'seed2_algo0_D1.0_tol5.0_beta0.95_sourcetrue_episodes.csv'
    pd.DataFrame(dict(training=[True, False], recoverySeconds=[0.0, float('nan')],
                      recoveryCensored=[0, 1])).to_csv(path, index=False)
    result = load_episodes(tmp_path)
    assert len(result) == 1
    assert result.iloc[0].seed == 2
    assert result.iloc[0].recoveryCensored == 1
    assert pd.isna(result.iloc[0].recoverySeconds)


def test_missing_exports_fail_clearly(tmp_path):
    with pytest.raises(ValueError, match='No mobile episode exports'):
        load_episodes(tmp_path)
