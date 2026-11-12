import json

import pytest

from experiments.campaign_view import build, view_jobs


def job(key, variant='legacy', algorithm=0, beta=0.95):
    return dict(key=key, variant=variant, algorithm=algorithm, beta=beta)


def test_the_view_drops_variants_sweeps_and_algorithms():
    manifest = [job('a'), job('b', variant='balanced'), job('c', variant='current', beta=0.5),
                job('d', variant='current'), job('e', variant='fixed', algorithm=2)]
    kept = view_jobs([manifest], drop_variants=['balanced'], drop_sweeps=['current'], drop_algorithms=[2])
    assert [j['key'] for j in kept] == ['a', 'd']


def test_a_run_of_two_campaigns_is_an_error():
    with pytest.raises(ValueError):
        view_jobs([[job('a')], [job('a')]])


def campaign(folder, jobs, state='complete'):
    folder.mkdir()
    (folder / 'manifest.json').write_text(json.dumps(jobs))
    for item in jobs:
        run = folder / 'exports' / item['key']
        run.mkdir(parents=True)
        (run / 'status.json').write_text(json.dumps(dict(state=state)))
    return folder


def test_the_view_links_the_runs_and_replaces_only_a_view(tmp_path):
    first = campaign(tmp_path / 'first', [job('a')])
    second = campaign(tmp_path / 'second', [job('b', variant='fixed', algorithm=11)])
    view = tmp_path / 'view'
    build(view, [first, second])
    assert (view / 'exports' / 'b').resolve() == (second / 'exports' / 'b').resolve()
    assert [j['key'] for j in json.loads((view / 'manifest.json').read_text())] == ['a', 'b']
    build(view, [first])
    assert [j['key'] for j in json.loads((view / 'manifest.json').read_text())] == ['a']
    with pytest.raises(ValueError):
        build(first, [second])


def test_the_view_refuses_incomplete_runs(tmp_path):
    with pytest.raises(ValueError):
        build(tmp_path / 'view', [campaign(tmp_path / 'running', [job('a')], state='running')])
