"""A view of several campaigns of one split for the analysis: one folder with their manifests merged and their runs
linked, without the runs that the figures must not show (variants left out, beta sweeps of the controls, algorithms
dropped). The view records its sources and filters in `view.json`.

python -m experiments.campaign_view <view> <campaign>... [--drop-variants balanced] [--drop-sweeps current]
                                    [--drop-algorithms 2]
"""

import argparse
import json
import shutil
from pathlib import Path

from experiments.campaign import FOCUS_BETA

DESCRIPTION = 'view.json'


def view_jobs(manifests, drop_variants=(), drop_sweeps=(), drop_algorithms=()):
    """The jobs of the manifests that the view keeps; a run declared by two campaigns is an error."""
    jobs, keys = [], set()
    for job in (job for manifest in manifests for job in manifest):
        if job['key'] in keys:
            raise ValueError(f'Run declared twice: {job["key"]}')
        keys.add(job['key'])
        dropped = (job['variant'] in drop_variants or job['algorithm'] in drop_algorithms
                   or (job['variant'] in drop_sweeps and job['beta'] != FOCUS_BETA))
        if not dropped:
            jobs.append(job)
    return jobs


def is_complete(home, job):
    status = home / 'exports' / job['key'] / 'status.json'
    return status.exists() and json.loads(status.read_text())['state'] == 'complete'


def build(view, campaigns, drop_variants=(), drop_sweeps=(), drop_algorithms=()):
    """Writes the view of `campaigns` (complete ones only) into `view`, replacing a previous view there."""
    if view.exists():
        if not (view / DESCRIPTION).exists():
            raise ValueError(f'{view} exists and is not a view')
        shutil.rmtree(view)
    manifests = [json.loads((home / 'manifest.json').read_text()) for home in campaigns]
    homes = {job['key']: home for home, manifest in zip(campaigns, manifests) for job in manifest}
    jobs = view_jobs(manifests, drop_variants, drop_sweeps, drop_algorithms)
    incomplete = [job['key'] for job in jobs if not is_complete(homes[job['key']], job)]
    if incomplete:
        raise ValueError(f'{len(incomplete)} incomplete runs, e.g. {incomplete[0]}')
    (view / 'exports').mkdir(parents=True)
    for job in jobs:
        (view / 'exports' / job['key']).symlink_to((homes[job['key']] / 'exports' / job['key']).resolve())
    (view / 'manifest.json').write_text(json.dumps(jobs, indent=1))
    description = dict(campaigns=[str(home) for home in campaigns], dropVariants=list(drop_variants),
                       dropSweeps=list(drop_sweeps), dropAlgorithms=list(drop_algorithms), runs=len(jobs))
    (view / DESCRIPTION).write_text(json.dumps(description, indent=2))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('view', type=Path)
    parser.add_argument('campaigns', type=Path, nargs='+')
    parser.add_argument('--drop-variants', nargs='*', default=[])
    parser.add_argument('--drop-sweeps', nargs='*', default=[], help='Variants kept at the focus beta only')
    parser.add_argument('--drop-algorithms', nargs='*', type=int, default=[])
    args = parser.parse_args()
    jobs = build(args.view, args.campaigns, args.drop_variants, args.drop_sweeps, args.drop_algorithms)
    print(f'{len(jobs)} runs: {args.view}')


if __name__ == '__main__':
    main()
