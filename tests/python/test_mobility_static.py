"""Phase accounting and pairing of the static vs. mobile analysis."""
import numpy as np
import pandas as pd
import pytest

from analysis import config
from analysis.mobility_pareto import FIXED_PERIODS
from analysis.mobility_static import (EVALUATION, PHASES, SEEDS, _evaluation_frames, _run_files, connected,
                                      frontier_gains, interval_series, paired_gains, phase_metrics, switch_reaction)

HEADER = ["episode", "training", "time", "diffusion", "tolerance", "beta", "roundsPerNodeSecond",
          "normalizedError", "toleratedError", "withinTolerance", "reachableFraction", "recoverySeconds",
          "recoveryCensored", "rewardRate", "absoluteError"]


def _cumulative_row(episode, t, rate=0.25, error=0.1, within=0.5, absolute=50.0, reachable=1.0):
    """Exported metrics are means per node and per second since the episode start."""
    return dict(episode=episode, training=episode < config.TRAINING_EPISODES,
                time=episode * config.EPISODE_LENGTH_S + t, diffusion=1.0, tolerance=5.0, beta=0.95,
                roundsPerNodeSecond=rate, normalizedError=error, toleratedError=error, withinTolerance=within,
                reachableFraction=reachable, recoverySeconds=np.nan, recoveryCensored=1.0, rewardRate=-1.0,
                absoluteError=absolute)


def _write_run(directory, name, rate=0.25, drop_frame=None, reachable=1.0):
    times = np.arange(0.0, config.EPISODE_LENGTH_S, config.FULL_EXPORT_INTERVAL_S)
    full = [_cumulative_row(episode, t, rate, reachable=reachable) for episode in range(config.N_EPISODES)
            for t in times if (episode, t) != drop_frame]
    episodes = [_cumulative_row(episode, config.EPISODE_LENGTH_S, rate, reachable=reachable)
                for episode in range(config.N_EPISODES)]
    pd.DataFrame(full, columns=HEADER).to_csv(directory / f"{name}_full.csv", index=False)
    pd.DataFrame(episodes, columns=HEADER).to_csv(directory / f"{name}_episodes.csv", index=False)


NAME = "seed1_algo0_D1.0_tol5.0_beta0.95_dt0.25_train10_length200.0_sourcetrue"


def test_the_episode_row_closes_the_frames_and_phases_integrate_constant_rates(tmp_path):
    _write_run(tmp_path, NAME)
    frames = _evaluation_frames(tmp_path / f"{NAME}_episodes.csv", tmp_path / f"{NAME}_full.csv")
    assert set(frames.episode) == set(EVALUATION)
    assert frames.t.max() == config.EPISODE_LENGTH_S
    phases = phase_metrics(frames).set_index("phase")
    for phase in PHASES:
        assert phases.loc[phase.key, "rounds"] == pytest.approx(0.25 * phase.duration * config.N_NODES)
        assert phases.loc[phase.key, "error"] == pytest.approx(0.1 * phase.duration)
        assert phases.loc[phase.key, "outside"] == pytest.approx(0.5 * phase.duration)


def test_a_missing_evaluation_frame_is_rejected(tmp_path):
    _write_run(tmp_path, NAME, drop_frame=(15, 40.0))
    with pytest.raises(ValueError, match="Incomplete evaluation frames"):
        _evaluation_frames(tmp_path / f"{NAME}_episodes.csv", tmp_path / f"{NAME}_full.csv")


def test_interval_series_differentiates_the_cumulative_means(tmp_path):
    _write_run(tmp_path, NAME, rate=0.4)
    frames = _evaluation_frames(tmp_path / f"{NAME}_episodes.csv", tmp_path / f"{NAME}_full.csv")
    series = interval_series(frames)
    assert series.t.min() == config.FULL_EXPORT_INTERVAL_S
    assert series.rate.to_numpy() == pytest.approx(np.full(len(series), 0.4))


def test_duplicated_runs_must_be_identical(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir(), second.mkdir()
    _write_run(first, NAME)
    (second / f"{NAME}_full.csv").symlink_to(first / f"{NAME}_full.csv")
    assert len(_run_files([first, second])) == 1
    (second / f"{NAME}_episodes.csv").write_text("different")
    with pytest.raises(ValueError, match="Two different exports"):
        _run_files([first, second])


def _metrics(policy_error):
    """Fixed periods with error = 1000 / rounds; one adaptive policy with the given error at 5000 rounds."""
    rows = []
    for seed in SEEDS:
        for algorithm, period in FIXED_PERIODS.items():
            rounds = config.N_NODES * config.EPISODE_LENGTH_S / period
            rows.append(dict(algorithm=algorithm, beta=0.95, diffusion=1.0, seed=seed, phase="episode",
                             rounds=rounds, error=1000 / rounds, outside=10.0, absolute=1.0))
        rows.append(dict(algorithm=0, beta=0.95, diffusion=1.0, seed=seed, phase="episode", rounds=5000.0,
                         error=policy_error(seed), outside=10.0, absolute=1.0))
    return pd.DataFrame(rows)


def test_gains_are_paired_by_seed_against_the_same_phase_frontier():
    on_frontier = 1000 / 5000  # the 4 s fixed period
    gains = frontier_gains(_metrics(lambda seed: on_frontier * (1 - 0.1 * seed))).set_index("gain")
    error = gains.loc["error"]
    assert error["mean"] == pytest.approx(0.25)  # mean of 10%, 20%, 30%, 40%
    assert error["seeds"] == len(SEEDS)
    assert gains.loc["outside", "mean"] == pytest.approx(0.0)


def test_paired_gains_are_the_samples_of_the_frontier_gains():
    metrics = _metrics(lambda seed: 1000 / 5000 * (1 - 0.1 * seed))
    samples = paired_gains(metrics).set_index("seed").gain
    assert samples.to_dict() == pytest.approx({seed: 0.1 * seed for seed in SEEDS})
    assert samples.mean() == pytest.approx(frontier_gains(metrics).set_index("gain").loc["error", "mean"])


def test_the_nodes_cut_off_from_the_source_are_removed_from_the_error(tmp_path):
    _write_run(tmp_path, NAME, reachable=0.9)
    frames = _evaluation_frames(tmp_path / f"{NAME}_episodes.csv", tmp_path / f"{NAME}_full.csv")
    phases = phase_metrics(frames).set_index("phase")
    assert phases.loc["tracking", "unreachable"] == pytest.approx(0.1 * PHASES[3].duration)
    assert connected(phases).loc["tracking", "error"] == pytest.approx(0.0, abs=1e-9)  # 0.1 error, all cut off
    series = interval_series(frames)
    assert series.unreachable.to_numpy() == pytest.approx(np.full(len(series), 0.1))


def test_switch_reaction_compares_the_rate_after_and_before_the_switch():
    times = np.arange(2.0, config.EPISODE_LENGTH_S + 1, 2.0)
    rate = np.where((times > config.SOURCE_SWITCH_S) & (times <= config.SOURCE_SWITCH_S + 20), 0.6, 0.3)
    series = pd.DataFrame(dict(t=times, rate=rate, algorithm=0, beta=0.95, diffusion=0.0, seed=1))
    assert switch_reaction(series).ratio.iloc[0] == pytest.approx(2.0)


def test_no_relative_gain_when_the_frontier_has_no_error_left():
    metrics = _metrics(lambda seed: 0.0)
    metrics.loc[metrics.algorithm.isin(FIXED_PERIODS), "error"] = 0.0
    gains = frontier_gains(metrics).set_index("gain")
    assert np.isnan(gains.loc["error", "mean"])


def test_waypoint_and_warm_runs_are_keyed_by_their_speed(tmp_path):
    _write_run(tmp_path, "seed2_algo3_V1.5_tol5.0_beta0.95_dt0.25_train10_length200.0_sourcetrue_waypoint_warm")
    assert set(_run_files([tmp_path])) == {(3, 0.95, 1.5, 2)}


def test_a_training_variant_replaces_only_the_learners_of_the_baseline(tmp_path, monkeypatch):
    from analysis import mobility_variants
    baseline, variants = tmp_path / "baseline", tmp_path / "variants"
    learner = "seed1_algo0_D1.0_tol5.0_beta0.95_dt0.25_train10_length200.0_sourcetrue"
    fixed = "seed1_algo5_D1.0_tol5.0_beta0.95_dt0.25_train10_length200.0_sourcetrue"
    for directory in (baseline, variants / "complete-replay"):
        directory.mkdir(parents=True)
        _write_run(directory, learner, rate=0.25 if directory == baseline else 0.5)
    _write_run(baseline, fixed)
    monkeypatch.setattr(mobility_variants, "DEFAULT_EXPORTS", [baseline])
    monkeypatch.setattr(mobility_variants, "VARIANT_EXPORTS", variants)
    runs = mobility_variants.variant_runs(mobility_variants.VARIANTS[1])
    assert runs[(0, 0.95, 1.0, 1)]["full"].parent == variants / "complete-replay"
    assert runs[(5, 0.95, 1.0, 1)]["full"].parent == baseline


def test_region_series_split_the_rate_and_the_error_between_moving_and_still_nodes(tmp_path):
    from analysis.mobility_regions import allocation, group_series
    rows = []
    for episode in EVALUATION:
        for t in np.arange(0.0, config.EPISODE_LENGTH_S, config.FULL_EXPORT_INTERVAL_S):
            for node, rate in ((0, 1.0), (1, 0.25)):  # node 0 in the region: four times the rounds after the event
                after = max(0.0, t - config.SOURCE_SWITCH_S)
                rounds = 0.25 * t + (rate - 0.25) * after
                region = node == 0 and t >= config.SOURCE_SWITCH_S
                rows.append(dict(episode=episode, time=t, node=node, x=0.0, y=0.0, source=False, rounds=rounds,
                                 absoluteError=10.0 * t, region=region))
    path = tmp_path / "run_positions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    series = group_series(path).set_index(["moving", "time"])
    assert series.loc[(True, 150.0), "rate"] == pytest.approx(1.0)
    assert series.loc[(False, 150.0), "rate"] == pytest.approx(0.25)
    assert series.loc[(True, 60.0), "rate"] == pytest.approx(0.25)
    assert series.loc[(False, 60.0), "error"] == pytest.approx(10.0)
    ratios = allocation(series.reset_index().assign(algorithm=0, level=1.0, seed=1)).set_index("window")
    assert ratios.loc["after", "mean"] == pytest.approx(4.0, rel=0.05)
    assert ratios.loc["before", "mean"] == pytest.approx(1.0)


def test_still_nodes_are_classified_by_whether_their_route_to_the_source_crosses_a_moving_node():
    from analysis.mobility_regions import COMMUNICATION_RANGE_M, route_classes
    step = COMMUNICATION_RANGE_M * 0.8
    frame = pd.DataFrame(dict(  # a chain source - moving - still, a still neighbour of the source, an isolated node
        x=[0.0, step, 2 * step, 0.0, 10 * step], y=[0.0, 0.0, 0.0, step, 10 * step],
        source=[True, False, False, False, False], region=[False, True, False, False, False]))
    assert list(route_classes(frame)) == ["clear", "moving", "through", "clear", "cut off"]
    moving_source = frame.assign(region=[True, False, False, False, False])
    assert list(route_classes(moving_source)) == ["moving", "through", "through", "through", "cut off"]


def test_the_beta_response_divides_the_error_by_the_fixed_reference_of_the_same_seed():
    from analysis.mobility_paper import REFERENCE_FIXED, beta_response
    metrics = pd.DataFrame(dict(
        algorithm=[REFERENCE_FIXED, 0, REFERENCE_FIXED, 0], beta=[0.95] * 4, seed=[1, 1, 2, 2],
        phase=["tracking"] * 4, rounds=[2000.0, 1200.0, 2000.0, 1800.0], error=[10.0, 8.0, 20.0, 30.0]))
    error = beta_response(metrics, "tracking", "error").set_index(["algorithm", "seed"]).value
    assert error[(0, 1)] == pytest.approx(0.8) and error[(0, 2)] == pytest.approx(1.5)
    assert error[(REFERENCE_FIXED, 2)] == pytest.approx(1.0)
    rate = beta_response(metrics, "tracking", "rate").set_index(["algorithm", "seed"]).value
    assert rate[(0, 1)] == pytest.approx(1200.0 / (config.N_NODES * 60.0))


def test_the_paper_metrics_turn_the_mobile_rates_into_episode_totals():
    from analysis.mobility_paper import paper_metrics
    episodes = pd.DataFrame(dict(roundsPerNodeSecond=[0.3, 0.5], withinTolerance=[0.8, 0.6],
                                 normalizedError=[0.1, 0.2], rewardRate=[-0.01, -0.03]))
    metrics = paper_metrics(episodes)
    length = config.EPISODE_LENGTH_S
    assert metrics["ticks"] == pytest.approx(0.4 * config.N_NODES * length)
    assert metrics["unstable"] == pytest.approx(0.3 * length)
    assert metrics["error"] == pytest.approx(0.15 * length)
    assert metrics["reward"] == pytest.approx(-0.02 * length)


def test_the_attention_on_motion_compares_the_attended_neighbour_with_a_random_one():
    from analysis.mobility_paper import attention_on_motion
    step = 30.0
    rows = []
    for t, moved in ((1.0, 0.0), (2.0, 5.0)):  # node 1 moves 5 m between the two seconds
        positions = {0: (0.0, 0.0), 1: (step + moved, 0.0), 2: (0.0, step)}
        for node, (x, y) in positions.items():
            best = positions[0] if node == 1 else positions[1]  # the still nodes look at the moving one
            rows.append(dict(tEp=t, nodeId=node, x=x, y=y, bestNbrX=best[0], bestNbrY=best[1],
                             pc1=1.0 if node == 1 else 0.0))
    shares = attention_on_motion(pd.DataFrame(rows)).set_index("tEp")
    assert list(shares.index) == [2.0]  # the first second has no motion to compare with
    assert shares.loc[2.0, "attended"] == pytest.approx(1.0)
    assert shares.loc[2.0, "chance"] == pytest.approx(0.5)  # one of the two neighbours of nodes 0 and 2 moves
    assert shares.loc[2.0, "pc1_moving"] == pytest.approx(1.0) and shares.loc[2.0, "pc1_still"] == pytest.approx(0.0)
