"""The mobile beta sweep is paired by seed and must not silently publish incomplete curves."""
import numpy as np
import pandas as pd
import pytest
from analysis.mobility_pareto import prepare, frontier_gains, ADAPTIVE, FIXED_PERIODS, WEIGHTS


@pytest.fixture
def sweep():
    rows=[]
    for algorithm in [*ADAPTIVE, *FIXED_PERIODS]:
        weights=WEIGHTS if algorithm in ADAPTIVE else [.95]
        for beta in weights:
            for seed in range(1,5):
                rate=(.2+beta*.5) if algorithm in ADAPTIVE else 1/FIXED_PERIODS[algorithm]
                for episode in range(10,20):
                    rows.append(dict(algorithm=algorithm,beta=beta,seed=seed,episode=episode,
                        diffusion=1.,tolerance=5.,sourceChanges=True,protocol='dt0.25_train10_length200.0',
                        roundsPerNodeSecond=rate,normalizedError=.02/rate,toleratedError=.01/rate,
                        absoluteError=10/rate,withinTolerance=.5,reachableFraction=1.,
                        recoveryCensored=1.,rewardRate=-1.))
    return pd.DataFrame(rows)


def test_prepare_converts_to_the_existing_paper_units_and_averages_episodes(sweep):
    data=prepare(sweep)
    assert len(data)==124
    point=data[(data.algorithm==7)&(data.seed==1)].iloc[0]
    assert point['ticks']==pytest.approx(4000)
    assert point['error']==pytest.approx(20)
    assert point['tolerated']==pytest.approx(10)
    assert point['outside']==pytest.approx(100)
    assert len(data[data.algorithm.isin(FIXED_PERIODS)])==24


def test_prepare_rejects_an_incomplete_beta_curve(sweep):
    broken=sweep[~((sweep.algorithm==0)&(sweep.beta==.8)&(sweep.seed==3))]
    with pytest.raises(ValueError,match='Incomplete'):
        prepare(broken)


def test_prepare_rejects_mixing_protocols(sweep):
    sweep.loc[0,'diffusion']=4.
    with pytest.raises(ValueError,match='one mobility scenario'):
        prepare(sweep)


def test_frontier_gains_are_paired_by_seed_and_never_extrapolate(sweep):
    data=prepare(sweep)
    gain=frontier_gains(data)
    assert len(gain)==len(ADAPTIVE)*len(WEIGHTS)*3
    assert (gain['seeds']==4).all()
    assert np.isfinite(gain['mean']).all()
    # A learner below the fastest baseline's error is outside the measured reference.
    data.loc[data.algorithm==0,'error']=0.
    gain=frontier_gains(data)
    cells=gain[(gain.algorithm==0)&(gain.gain=='rounds_at_error')]
    assert cells['mean'].isna().all()
    assert (cells['seeds']==0).all()


def test_prepare_rejects_a_partially_written_evaluation_run(sweep):
    broken=sweep[~((sweep.algorithm==0)&(sweep.beta==.8)&(sweep.seed==3)&(sweep.episode==19))]
    with pytest.raises(ValueError,match='Incomplete evaluation episodes'):
        prepare(broken)
