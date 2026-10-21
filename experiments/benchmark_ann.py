"""Bounded CPU HNSW vs sequential exact assignment pilot; not GPU ANN.

Fixed workload: N=8192, D=14, K=16/64/256, seed=0, ef=4/16/64.
Two warmups and five measured repeats; initial and moved snapshots are separate.
"""

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pcap.assignments_ann import HNSWCPUBackend
from pcap.assignments_exact import assign_exact


def digest(data):
    return hashlib.sha256(data).hexdigest()


def array_hash(data):
    return digest(str((data.shape, data.dtype.str)).encode() + data.tobytes())


def command(args):
    try:
        run = subprocess.run(args, capture_output=True, text=True, timeout=10, cwd=ROOT)
        return dict(returncode=run.returncode, stdout=run.stdout.strip(), stderr=run.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(error=str(exc))


def hardware():
    return dict(cpuinfo=Path('/proc/cpuinfo').read_text(),
                meminfo=Path('/proc/meminfo').read_text(),
                loadavg=Path('/proc/loadavg').read_text().strip(),
                gpu=command(['nvidia-smi', '--query-gpu=name,driver_version,pstate,temperature.gpu,clocks.sm,utilization.gpu',
                             '--format=csv,noheader,nounits']))


def dataset(k):
    rng = np.random.default_rng(0)
    anchors = rng.normal(0, 0.4, (4, 14))
    centers = np.ascontiguousarray(anchors[np.arange(k) % 4] + rng.normal(0, 0.5, (k, 14)), dtype=np.float32)
    points = np.ascontiguousarray(anchors[rng.integers(0, 4, 8192)] + rng.normal(0, 0.8, (8192, 14)), dtype=np.float32)
    moved = np.ascontiguousarray(centers[rng.permutation(k)] + rng.normal(0, 0.3, centers.shape), dtype=np.float32)
    return points, centers, moved


def quality(points, centers, ann, exact):
    # Chunked independent recomputation; no N x K allocation.
    selected = np.empty(len(points), np.float64)
    reference = np.empty(len(points), np.float64)
    for begin in range(0, len(points), 1024):
        end = min(begin + 1024, len(points))
        x = points[begin:end].astype(np.float64)
        delta = x - centers[ann.labels[begin:end]].astype(np.float64)
        selected[begin:end] = np.einsum('ij,ij->i', delta, delta)
        delta = x - centers[exact.labels[begin:end]].astype(np.float64)
        reference[begin:end] = np.einsum('ij,ij->i', delta, delta)
    bound = 64 * np.finfo(np.float64).eps * np.maximum(1.0, selected)
    if not (np.all(np.isfinite(selected)) and np.all(np.abs(selected - ann.squared_distances) <= bound)
            and np.allclose(reference, exact.squared_distances, rtol=2e-14, atol=1e-14)
            and np.all(selected >= reference - bound)):
        raise RuntimeError('Independent distance/quality validation failed')
    excess = np.maximum(selected - reference, 0.0)
    return dict(recall=float(np.mean(ann.labels == exact.labels)),
                tie_aware_agreement=float(np.mean(np.abs(selected - reference) <= bound)),
                mean_excess_squared_distance=float(np.mean(excess)),
                max_excess_squared_distance=float(np.max(excess)),
                objective_gap=float((selected.sum() - reference.sum()) / reference.sum()),
                ann_assignment_sse=float(selected.sum()), exact_assignment_sse=float(reference.sum()))


def measure_ann(points, centers, moved, ef):
    backend = HNSWCPUBackend(seed=0, m=8, ef_construction=100,
                             ann_effort=ef, query_batch_size=4096)
    measured = []
    for phase, snapshot in [('initial', centers), ('refreshed', moved)]:
        start = perf_counter()
        maintenance_start = perf_counter()
        if phase == 'initial':
            backend.prepare(snapshot)
        else:
            backend.refresh(snapshot)
        maintenance = perf_counter() - maintenance_start
        query_start = perf_counter()
        result = backend.assign(points, snapshot, effort=ef)
        query = perf_counter() - query_start
        complete = perf_counter() - start
        measured.append((result, dict(ann_complete_external_s=complete,
                                      maintenance_external_s=maintenance,
                                      ann_query_external_s=query,
                                      ann_query_internal_s=result.query_time_s,
                                      maintenance_internal_s=result.index_update_time_s)))
    ledger = backend.timings
    if ledger['initial_build_count'] != 1 or ledger['rebuild_count'] != 1 or ledger['pending_index_update_s'] != 0:
        raise RuntimeError('Maintenance accounting failed')
    return measured


def measure_exact(points, centers, moved):
    measured = []
    for snapshot in (centers, moved):
        start = perf_counter()
        result = assign_exact(points, snapshot)
        measured.append((result, perf_counter() - start))
    return measured


def summarize(rows):
    output = []
    for k in (16, 64, 256):
        for ef in (4, 16, 64):
            for snapshot in ('initial', 'refreshed'):
                case = [r for r in rows if r['k'] == k and r['ef'] == ef
                        and r['snapshot'] == snapshot and r['phase'] == 'measured']
                if len(case) != 5:
                    continue  # Incomplete cases are preserved only in raw.csv.
                row = dict(k=k, ef=ef, snapshot=snapshot, repeats=5)
                for key in ('exact_external_s', 'ann_complete_external_s', 'ann_query_external_s',
                            'maintenance_external_s', 'ann_query_internal_s', 'maintenance_internal_s'):
                    q25, median, q75 = np.percentile([r[key] for r in case], [25, 50, 75])
                    row[key + '_median'] = float(median)
                    row[key + '_iqr'] = float(q75 - q25)
                row['speedup_with_maintenance'] = row['exact_external_s_median'] / row['ann_complete_external_s_median']
                for key in ('recall', 'tie_aware_agreement', 'mean_excess_squared_distance',
                            'max_excess_squared_distance', 'objective_gap'):
                    row[key + '_min'] = min(r[key] for r in case)
                    row[key + '_max'] = max(r[key] for r in case)
                output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    sources = ['experiments/benchmark_ann.py', 'src/pcap/assignments_ann.py',
               'src/pcap/assignments_exact.py', 'src/pcap/contracts.py']
    for name in sources:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), status='running',
                    git_commit=command(['git', 'rev-parse', 'HEAD']), git_status=command(['git', 'status', '--short']),
                    python=sys.version, executable=sys.executable, platform=platform.platform(),
                    versions={d.metadata['Name']: d.version for d in distributions()}, hardware_before=hardware(),
                    source_sha256={name: digest((ROOT / name).read_bytes()) for name in sources},
                    settings=dict(n=8192, d=14, k=[16, 64, 256], seed=0, ef=[4, 16, 64],
                                  warmups=2, repeats=5, M=8, ef_construction=100, threads=1, query_batch_size=4096,
                                  budget_s=120, dataset='overlapping four-anchor mixture; exact generator archived'),
                    timing_policy='External perf_counter around calls; ANN complete independently wraps prepare/refresh plus assign. Backend object construction excluded. Validation/optional import inside calls included. Internal fields are diagnostic subsets, never added to external complete. Explicit gc.collect outside timing, automatic GC unchanged.',
                    ordering='all cases warmed first; case order reverses each round; exact/ANN order alternates per case/repeat',
                    scope='CPU assignment pilot, one build or rebuild per full query; not full clustering or GPU ANN')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rows = []
    datasets = {k: dataset(k) for k in (16, 64, 256)}
    hashes = {k: [array_hash(x) for x in arrays] for k, arrays in datasets.items()}
    cases = [(k, ef) for k in (16, 64, 256) for ef in (4, 16, 64)]
    try:
        with (output / 'raw.csv').open('x', newline='') as handle:
            writer = None
            for phase, count in [('warmup', 2), ('measured', 5)]:
                for repeat in range(count):
                    for k, ef in (cases if repeat % 2 == 0 else list(reversed(cases))):
                        if perf_counter() - started > 120:
                            raise TimeoutError('Pilot budget reached between pairs')
                        points, centers, moved = datasets[k]
                        gc.collect()
                        order = ['exact', 'ann'] if (repeat + cases.index((k, ef))) % 2 == 0 else ['ann', 'exact']
                        results = {}
                        for method in order:
                            results[method] = (measure_exact(points, centers, moved) if method == 'exact'
                                               else measure_ann(points, centers, moved, ef))
                        if [array_hash(x) for x in datasets[k]] != hashes[k]:
                            raise RuntimeError('Inputs mutated')
                        for i, snapshot in enumerate(('initial', 'refreshed')):
                            ann, timings = results['ann'][i]
                            exact, elapsed = results['exact'][i]
                            row = dict(phase=phase, repeat=repeat, n=8192, d=14, k=k, seed=0, ef=ef,
                                       snapshot=snapshot, order=','.join(order), points_sha256=hashes[k][0],
                                       centroids_sha256=hashes[k][i+1], exact_external_s=elapsed,
                                       exact_internal_s=exact.query_time_s, **timings,
                                       **quality(points, (centers, moved)[i], ann, exact))
                            if writer is None:
                                writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader()
                            writer.writerow(row); handle.flush(); rows.append(row)
            manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        summaries = summarize(rows)
        if summaries:
            with (output / 'summary.csv').open('x', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
        manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_s=perf_counter()-started,
                        hardware_after=hardware(), raw_rows=len(rows), completed_summaries=len(summaries))
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        paths = sorted(p for p in output.rglob('*') if p.is_file())
        lines = [digest(p.read_bytes()) + '  ' + str(p.relative_to(output)) for p in paths]
        (output / 'checksums.sha256').write_text('\n'.join(lines) + '\n')
        for line in lines:
            expected, name = line.split('  ', 1)
            if digest((output / name).read_bytes()) != expected:
                raise RuntimeError('Archive checksum mismatch')
    print(f'Artifacts: {output}; {len(rows)} raw rows; {len(summaries)} summaries; checksums verified')


if __name__ == '__main__':
    main()
