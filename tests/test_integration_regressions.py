"""Regression gates for integration, CLI provenance and synthetic raster flows."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pcap.adaptive_ann_controller import ANNEffortPolicy
from pcap.adaptive_combined import fit_adaptive_minibatch
from pcap.adaptive_controller import BatchPolicy
from pcap.assignments_ann import HNSWCPUBackend
from pcap.assignments_exact import assign_exact, ExactCPUBackend
from pcap.contracts import FitConfig
from pcap.minibatch_ann import fit_minibatch_ann_cpu
from pcap.metrics import compute_sse, compute_silhouette_score, compute_davies_bouldin

ROOT = Path(__file__).resolve().parents[1]


def module(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize('levels', [(64.5, 32), (True,), ('64',), (np.bool_(True),)])
def test_ann_levels_reject_coercion(levels):
    with pytest.raises((TypeError, ValueError)):
        ANNEffortPolicy(levels=levels)


@pytest.mark.parametrize('fit', [fit_minibatch_ann_cpu, fit_adaptive_minibatch])
def test_backend_requires_all_methods(fit):
    points = np.zeros((4, 2), np.float32)
    with pytest.raises(TypeError, match='backend'):
        fit(points, points[:1].copy(), FitConfig(1, 0, 1, 0, 2, ann_effort=64),
            backend=SimpleNamespace(assign=lambda: None))


@pytest.mark.parametrize('mode', ['batch_only', 'ann_only', 'combined'])
def test_real_ann_modes_final_exact_and_counts(mode):
    rng = np.random.default_rng(23)
    points = rng.normal(size=(80, 4)).astype(np.float32)
    initial = points[:4].copy()
    cfg = FitConfig(4, 1, 15, 0, 16, 2, None if mode == 'batch_only' else 64)
    a = fit_adaptive_minibatch(points, initial, cfg, mode=mode)
    b = fit_adaptive_minibatch(points, initial, cfg, mode=mode)
    exact = assign_exact(points, a.centroids)
    np.testing.assert_array_equal(a.labels, exact.labels)
    assert a.sse == pytest.approx(float(exact.squared_distances.sum()), rel=1e-12)
    np.testing.assert_array_equal(a.centroids, b.centroids)
    assert sum(a.metadata['cluster_observation_counts']) == sum(h['batch_size'] for h in a.history)
    assert a.metadata['execution_device'] == 'cpu'
    if mode != 'batch_only':
        assert a.metadata['rebuild_count'] == a.metadata['steps']


def test_real_fixed_ann_explicit_backend_and_final_sse():
    points = np.random.default_rng(3).normal(size=(40, 4)).astype(np.float32)
    result = fit_minibatch_ann_cpu(points, points[:3].copy(), FitConfig(3, 0, 5, 0, 8, 2, 16),
                                   backend=HNSWCPUBackend())
    exact = assign_exact(points, result.centroids)
    np.testing.assert_array_equal(result.labels, exact.labels)
    assert result.sse == pytest.approx(exact.squared_distances.sum())
    assert result.metadata['assignment_points_processed'] == 5*8 + (2*2+3)*40+40


@pytest.mark.parametrize('mode', ['batch_only', 'ann_only', 'combined'])
def test_overflow_rejected_before_backend_execution(mode):
    points = np.zeros((4, 2), np.float32)
    cfg = FitConfig(1, 0, np.uint64(2**63), 0, 1, ann_effort=None if mode == 'batch_only' else 64)
    with pytest.raises(ValueError, match='int64 capacity'):
        fit_adaptive_minibatch(points, points[:1].copy(), cfg, mode=mode)


def test_explicit_policy_floors_and_effort_preserved():
    points = np.zeros((8, 2), np.float32)
    cfg = FitConfig(1, 0, 6, 0, 4, 1, 32)
    bp = BatchPolicy(4, 8, 4, objective_floor=7, quiet_checkpoints=100)
    ap = ANNEffortPolicy(initial_effort=32, objective_floor=9, quiet_checkpoints=100)
    result = fit_adaptive_minibatch(points, points[:1].copy(), cfg, batch_policy=bp, ann_policy=ap)
    assert result.metadata['batch_policy']['objective_floor'] == 7
    assert result.metadata['ann_policy']['objective_floor'] == 9
    assert result.history[-1]['phase'] == 'confirmation'
    assert result.metadata['confirmation_checks'] == 3
    assert result.metadata['termination_reason'] == 'tolerance'
    assert result.history[3]['batch_size'] == 8
    assert result.history[3]['ann_effort'] == 64


def test_actual_changes_and_deterioration_reset_confirmation(monkeypatch):
    import pcap.adaptive_combined as combined
    from dataclasses import replace
    from pcap.adaptive_ann_controller import ANNDecision
    points = np.zeros((8, 2), np.float32)
    calls = 0
    def controlled(policy, state, checkpoint):
        nonlocal calls
        calls += 1
        changed = calls == 3
        deteriorated = calls == 5
        effort = 32 if changed else state.effort
        return replace(state, effort=effort), ANNDecision(state.effort, effort,
                    'decrease' if changed else 'retain', 'test', 0, deteriorated, True)
    monkeypatch.setattr(combined, 'transition_ann', controlled)
    result = combined.fit_adaptive_minibatch(points, points[:1].copy(), FitConfig(1, 0, 7, 0, 4, 1, 64), mode='ann_only')
    assert not result.metadata['confirming_phase_entered']
    assert result.history[2]['stable_checks'] == 0
    assert result.history[4]['stable_checks'] == 0


def test_no_premature_confirmation_at_cap():
    points = np.zeros((8, 2), np.float32)
    cfg = FitConfig(1, 0, 5, 0, 4, 1)
    result = fit_adaptive_minibatch(points, points[:1].copy(), cfg, mode='batch_only',
                                 batch_policy=BatchPolicy(4, 8, 4, quiet_checkpoints=100))
    assert result.metadata['confirmation_checks'] == 2
    assert result.metadata['termination_reason'] == 'max_steps'


def test_metrics_label_range_and_coincident_centroids():
    points = np.array([[-1], [1], [-2], [2]], np.float32)
    labels = np.array([0, 0, 1, 1], np.int32)
    assert np.isinf(compute_davies_bouldin(points, labels))
    with pytest.raises(ValueError, match='nonnegative'):
        compute_sse(points, np.zeros((2, 1), np.float32), np.array([-1, 0, 1, 1], np.int32))


def test_streamed_silhouette_matches_independent_reference():
    points = np.array([[0, 0], [0, 2], [10, 10], [10, 12], [40, 40]], np.float32)
    labels = np.array([0, 0, 1, 1, 2], np.int32)
    distances = np.linalg.norm(points.astype(float)[:, None] - points.astype(float)[None], axis=2)
    reference = []
    for i in range(len(points)):
        own = labels == labels[i]
        if own.sum() == 1:
            reference.append(0)
            continue
        a = distances[i, own].sum() / (own.sum()-1)
        b = min(distances[i, labels == label].mean() for label in np.unique(labels) if label != labels[i])
        reference.append((b-a)/max(a,b))
    assert compute_silhouette_score(points, labels) == pytest.approx(np.mean(reference))


def test_runner_baselines_and_no_gpu_fallback(monkeypatch):
    suite = module('experiments/run_suite.py')
    points = np.zeros((8, 2), np.float32)
    cfg = FitConfig(1, 0, 2, 0, 4)
    result = suite.run_single_method('C0', points, points[:1].copy(), cfg)
    assert result.metadata['config']['batch_size'] is None
    def broken(*args):
        raise RuntimeError('CUDA failure')
    monkeypatch.setitem(sys.modules, 'pcap.minibatch_cuda', SimpleNamespace(fit_minibatch_cuda=broken))
    with pytest.raises(RuntimeError, match='CUDA failure'):
        suite.run_single_method('G1', points, points[:1].copy(), cfg)
    def exact_gpu(points, initial, config):
        assert config.batch_size is None and config.ann_effort is None
        return 'gpu'
    monkeypatch.setitem(sys.modules, 'pcap.kmeans_cuda', SimpleNamespace(fit_cuda=exact_gpu))
    assert suite.run_single_method('G0', points, points[:1].copy(), cfg) == 'gpu'


def test_runner_cpu_pilot_and_plot_entrypoint(tmp_path, monkeypatch):
    suite = module('experiments/run_suite.py')
    csv = suite.main(['--methods', 'C0', 'G2', 'G3', 'G4', 'G5', '--sizes', '24', '--k', '2',
                      '--seeds', '0', '--max-steps', '3', '--repeats', '1', '--output-dir', str(tmp_path)])
    manifest = json.loads(next(tmp_path.glob('manifest*')).read_text())
    assert manifest['complete'] and manifest['total_completed'] == 5
    assert not manifest['failures']
    rows = [json.loads(line) for line in next(tmp_path.glob('raw*')).read_text().splitlines()]
    assert len(rows) == 10
    assert all(row['history'] and row['C0_sha256'] for row in rows)
    plots = module('experiments/plot_results.py')
    monkeypatch.setattr(sys, 'argv', ['plot_results', '--csv', str(csv), '--output-dir', str(tmp_path/'plots')])
    plots.main()
    assert len(list((tmp_path/'plots').glob('*.png'))) == 3
    plots.render_cluster_map(np.array([0, 1], np.int32), np.array([[True, False], [False, True]]),
                             np.array([0, 3]), tmp_path/'map.png')
    assert (tmp_path/'map.png').stat().st_size > 0


def test_failed_suite_records_failure_and_raises(tmp_path, monkeypatch):
    suite = module('experiments/run_suite.py')
    def broken(*args):
        raise RuntimeError('fit failure')
    monkeypatch.setattr(suite, 'run_single_method', broken)
    with pytest.raises(RuntimeError, match='fit failure'):
        suite.main(['--methods', 'C0', '--sizes', '8', '--k', '1', '--seeds', '0', '--output-dir', str(tmp_path)])
    manifest = json.loads(next(tmp_path.glob('manifest*')).read_text())
    assert not manifest['complete'] and manifest['total_completed'] == 0
    assert manifest['failures'][0]['error'] == 'RuntimeError: fit failure'


@pytest.fixture
def safe_scene(tmp_path):
    from test_preprocessing import xml
    import rasterio
    from rasterio.transform import from_origin
    scene = tmp_path/'raw'/'TEST.SAFE'
    scene.mkdir(parents=True)
    (scene/'MTD_MSIL2A.xml').write_text(xml())
    prefix = 'T43ABC_20261001T000000'
    for band in ('B02', 'B03', 'B04', 'B08', 'SCL'):
        resolution = 20 if band == 'SCL' else 10
        folder = scene/'GRANULE'/'L2A_TEST'/'IMG_DATA'/f'R{resolution}m'
        folder.mkdir(parents=True, exist_ok=True)
        # Synthetic GTiff payload using SAFE-like filenames; not real JP2 evidence.
        path = folder/f'{prefix}_{band}_{resolution}m.jp2'
        side = 2 if band == 'SCL' else 4
        values = np.full((side, side), 4, np.uint16) if band == 'SCL' else np.arange(2000, 2016, dtype=np.uint16).reshape(4, 4)
        if band == 'SCL':
            values[0, 0] = 8
        with rasterio.open(path, 'w', driver='GTiff', height=side, width=side, count=1, dtype='uint16',
                           crs='EPSG:32643', transform=from_origin(500000, 4000000, resolution, resolution)) as dst:
            dst.write(values, 1)
    return scene


def test_asset_discovery_rejects_ambiguity_and_mixed_acquisition(safe_scene):
    prepare = module('scripts/prepare_sentinel2_scenes.py')
    bands, scl, xml = prepare.find_scene_assets(safe_scene)
    assert len(bands) == 4
    path = Path(scl)
    different = path.with_name(path.name.replace('20261001', '20261002'))
    path.rename(different)
    with pytest.raises(ValueError, match='share tile'):
        prepare.find_scene_assets(safe_scene)
    different.rename(path)
    duplicate = path.with_name('OTHER_SCL_20m.jp2')
    duplicate.write_bytes(b'')
    with pytest.raises(ValueError, match='exactly one SCL'):
        prepare.find_scene_assets(safe_scene)


def test_synthetic_scene_to_runner_and_cluster_map(safe_scene, tmp_path, monkeypatch):
    prepare = module('scripts/prepare_sentinel2_scenes.py')
    splits = tmp_path/'splits.json'
    splits.write_text(json.dumps({'TEST.SAFE': 'train'}))
    output = tmp_path/'processed'
    monkeypatch.setattr(sys, 'argv', ['prepare', '--raw-dir', str(safe_scene.parent), '--output-dir', str(output),
                        '--crop-size', '4', '--training-scene-id', 'TEST.SAFE', '--split-manifest', str(splits)])
    prepare.main()
    scene = output/'TEST.SAFE'
    X = np.load(scene/'X.npy')
    assert X.shape == (12, 4)
    metadata = json.loads((scene/'metadata.json').read_text())
    assert metadata['split'] == 'train'
    suite = module('experiments/run_suite.py')
    suite.main(['--methods', 'C0', 'G2', 'G5', '--scene-dir', str(scene), '--seeds', '0', '--k', '2',
                '--max-steps', '4', '--repeats', '1', '--output-dir', str(tmp_path/'runs')])
    result = fit_adaptive_minibatch(X, X[:2].copy(), FitConfig(2, 0, 4, 0, 4, 1, 64))
    plots = module('experiments/plot_results.py')
    plots.render_cluster_map(result.labels, np.load(scene/'valid_mask.npy'), np.load(scene/'pixel_indices.npy'), tmp_path/'scene.png')
    assert (tmp_path/'scene.png').exists()


def test_wrong_training_scene_rejected_before_processing(safe_scene, tmp_path, monkeypatch):
    prepare = module('scripts/prepare_sentinel2_scenes.py')
    splits = tmp_path/'splits.json'
    splits.write_text(json.dumps({'TEST.SAFE': 'test'}))
    monkeypatch.setattr(sys, 'argv', ['prepare', '--raw-dir', str(safe_scene.parent), '--training-scene-id', 'TYPO',
                        '--split-manifest', str(splits), '--output-dir', str(tmp_path/'out')])
    with pytest.raises(SystemExit):
        prepare.main()
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('mode', ['fixed', 'ann_only', 'combined'])
def test_approximate_training_never_supplies_final_labels(mode):
    from pcap.contracts import AssignmentResult
    class DeliberatelyPoorANN:
        def prepare(self, centroids, effort=None):
            self.centroids = centroids.copy()
        def refresh(self, centroids):
            self.centroids = centroids.copy()
        def assign(self, points, centroids, effort=None):
            labels = np.zeros(len(points), np.int32)
            delta = points.astype(float) - centroids[0].astype(float)
            return AssignmentResult(labels, np.sum(delta*delta, axis=1), 0)
    points = np.array([[0], [1], [20], [21]], np.float32)
    initial = points[[0, 3]].copy()
    cfg = FitConfig(2, 0, 2, 0, 4, 1, 64)
    result = (fit_minibatch_ann_cpu(points, initial, cfg, backend=DeliberatelyPoorANN()) if mode == 'fixed'
              else fit_adaptive_minibatch(points, initial, cfg, mode=mode, backend=DeliberatelyPoorANN()))
    exact = assign_exact(points, result.centroids)
    assert np.any(result.labels == 1)
    np.testing.assert_array_equal(result.labels, exact.labels)
    assert result.sse == pytest.approx(exact.squared_distances.sum())


def test_effort_mismatch_and_unsupported_default_rejected():
    points = np.zeros((8, 2), np.float32)
    cfg = FitConfig(1, 0, 1, 0, 4, ann_effort=32)
    with pytest.raises(ValueError, match='must match'):
        fit_adaptive_minibatch(points, points[:1].copy(), cfg, ann_policy=ANNEffortPolicy(initial_effort=64))
    cfg = FitConfig(1, 0, 1, 0, 4, ann_effort=7)
    with pytest.raises(ValueError, match='initial_effort'):
        fit_adaptive_minibatch(points, points[:1].copy(), cfg)


def test_asset_discovery_requires_product_xml_and_matching_product(safe_scene):
    prepare = module('scripts/prepare_sentinel2_scenes.py')
    xml = safe_scene/'MTD_MSIL2A.xml'
    original = xml.read_text()
    xml.write_text(original.replace('TEST.SAFE', 'OTHER.SAFE'))
    with pytest.raises(ValueError, match='product ID'):
        prepare.find_scene_assets(safe_scene)
    xml.write_text(original)
    xml.rename(safe_scene/'MTD_TL.xml')
    with pytest.raises(ValueError, match='MTD_MSIL2A'):
        prepare.find_scene_assets(safe_scene)


def test_scene_hash_validation_precedes_benchmark_outputs(tmp_path):
    suite = module('experiments/run_suite.py')
    scene = tmp_path/'scene'
    scene.mkdir()
    np.save(scene/'X.npy', np.zeros((4, 4), np.float32))
    (scene/'metadata.json').write_text(json.dumps({'X_sha256': 'wrong'}))
    with pytest.raises(ValueError, match='hash mismatch'):
        suite.main(['--methods', 'C0', '--scene-dir', str(scene), '--k', '1', '--output-dir', str(tmp_path/'out')])
    assert not (tmp_path/'out').exists()


def test_confirmation_quality_gate_resets_counter_without_unfreezing(monkeypatch):
    import pcap.adaptive_combined as combined
    from pcap.contracts import AssignmentResult
    real_assign = combined.assign_exact
    calls = 0
    def controlled(points, centroids):
        nonlocal calls
        calls += 1
        result = real_assign(points, centroids)
        # Initial probe + four checkpoints: first confirming checkpoint is bad.
        if calls == 5:
            return AssignmentResult(result.labels, np.full(len(points), .1), 0)
        return result
    monkeypatch.setattr(combined, 'assign_exact', controlled)
    points = np.zeros((8, 2), np.float32)
    result = combined.fit_adaptive_minibatch(points, points[:1].copy(), FitConfig(1, 0, 7, 0, 4, 1),
                    mode='batch_only', batch_policy=BatchPolicy(4, 8, 4, quiet_checkpoints=100))
    assert result.history[3]['phase'] == 'confirmation'
    assert result.history[3]['confirmation_checks'] == 0
    assert result.metadata['confirmation_checks'] == 3
    assert result.metadata['steps'] == 7
    assert all(h['batch_size'] == 8 for h in result.history[3:])
    assert result.sse == 0


def test_saved_models_scene_render_and_checksums(safe_scene,tmp_path,monkeypatch):
    import hashlib
    prepare=module('scripts/prepare_sentinel2_scenes.py')
    splits=tmp_path/'splits.json'
    splits.write_text(json.dumps({'TEST.SAFE':'train'}))
    processed=tmp_path/'prepared'
    argv=['prepare','--raw-dir',str(safe_scene.parent),'--output-dir',str(processed),
          '--crop-size','4','--training-scene-id','TEST.SAFE','--split-manifest',str(splits)]
    monkeypatch.setattr(sys,'argv',argv)
    prepare.main()
    with pytest.raises(FileExistsError):
        prepare.main()
    suite=module('experiments/run_suite.py')
    out=tmp_path/'runs'
    suite.main(['--methods','C0','--scene-dir',str(processed/'TEST.SAFE'),'--k','2',
                '--seeds','0','--repeats','1','--max-steps','2','--save-models','--output-dir',str(out)])
    model=next(out.glob('models_*/*'))
    render=module('scripts/render_scene.py')
    images=render.render(processed/'TEST.SAFE',model,tmp_path/'images')
    import rasterio
    with rasterio.open(images/'clusters.tif') as src:
        assert src.crs.to_epsg()==32643 and src.nodata==-1
        assert np.sum(src.read(1)>=0)==12
    assert (images/'true_color.png').exists() and (images/'false_color_nir.png').exists()
    for base, inventory in [(images,images/'checksums.sha256'),(out,next(out.glob('checksums*')))]:
        for line in inventory.read_text().splitlines():
            digest,path=line.split('  ',1)
            assert hashlib.sha256((base/path).read_bytes()).hexdigest()==digest
    with pytest.raises(FileExistsError):
        render.render(processed/'TEST.SAFE',model,images)


def test_suite_budget_records_explicit_failure(tmp_path):
    suite=module('experiments/run_suite.py')
    with pytest.raises(TimeoutError):
        suite.main(['--methods','C0','--sizes','8','--k','1','--seeds','0',
                    '--budget-seconds','1e-12','--output-dir',str(tmp_path)])
    manifest=json.loads(next(tmp_path.glob('manifest*')).read_text())
    assert not manifest['complete'] and manifest['failures']


def test_summary_uses_median_iqr_and_matched_work(tmp_path):
    import pandas as pd
    report=module('experiments/summarize_suite.py')
    rows=[]
    for method,times in [('C0',[1.,2.,9.]),('G0',[.5,1.,2.])]:
        for time in times:
            rows.append(dict(method=method,implementation='cpu_exact' if method=='C0' else 'cuda_exact',
                n=8,d=4,k=2,seed=0,wall_time_s=time,final_exact_sse=10.,steps=3,
                termination_reason='max_steps',convergence_satisfied=False,points_processed=24))
    csv=tmp_path/'data.csv';pd.DataFrame(rows).to_csv(csv,index=False)
    out=tmp_path/'report';table=report.summarize(csv,out)
    assert table.loc[table.method=='C0','median_fit_s'].iloc[0]==2.
    assert table.loc[table.method=='C0','iqr_fit_s'].iloc[0]==4.
    speed=pd.read_csv(out/'exact_speedups.csv')
    assert speed.speedup.iloc[0]==2 and speed.matched_work.iloc[0]
    rows[-1]['steps']=4
    pd.DataFrame(rows).to_csv(csv,index=False)
    report.summarize(csv,tmp_path/'unmatched')
    speed=pd.read_csv(tmp_path/'unmatched'/'exact_speedups.csv')
    assert not speed.matched_work.iloc[0] and pd.isna(speed.speedup.iloc[0])


def test_history_plots_and_no_figure_overwrite(tmp_path,monkeypatch):
    plots=module('experiments/plot_results.py')
    raw=tmp_path/'raw.jsonl'
    row=dict(phase='measured',repeat=0,method='G5',implementation='cpu_adaptive_combined',
             n=8,d=4,k=2,seed=0,history=[dict(step=1,batch_size=4,ann_effort=64),dict(step=2,batch_size=2,ann_effort=32)])
    raw.write_text(json.dumps(row)+'\n')
    plots.plot_histories(raw,tmp_path)
    assert (tmp_path/'history_n8_d4_k2_seed0_G5.png').exists()
    csv=tmp_path/'empty.csv';csv.write_text('method\n')
    monkeypatch.setattr(sys,'argv',['plots','--csv',str(csv),'--output-dir',str(tmp_path)])
    with pytest.raises(FileExistsError):
        plots.main()


def test_multiscene_summary_never_pools_equal_shapes(tmp_path):
    import pandas as pd
    summary=module('experiments/summarize_suite.py')
    rows=[]
    for dataset in ('train','test'):
        for method in ('C0','G0'):
            rows.append(dict(dataset_id=dataset,case_id=dataset,session_id='one',method=method,
                implementation=method,n=8,d=4,k=2,seed=0,wall_time_s=1.,final_exact_sse=10.,
                steps=2,termination_reason='max_steps',convergence_satisfied=False,points_processed=16))
    file=tmp_path/'data.csv';pd.DataFrame(rows).to_csv(file,index=False)
    result=summary.summarize(file,tmp_path/'summaries')
    assert len(result)==4
    assert all(result.repeats==1)


def test_declarative_prep_reuses_frozen_statistics_and_rejects_fake_holdout(safe_scene,tmp_path):
    prepare=module('scripts/prepare_study.py')
    cfg={'products':{'a':{'path':str(safe_scene),'product_id':'TEST.SAFE'}},
         'crops':[dict(id='train',product='a',window=[0,0,4,4],split='train')]}
    file=tmp_path/'study.json';file.write_text(json.dumps(cfg))
    out=prepare.prepare(file,tmp_path/'prepared')
    saved=(out/'standardization_stats.json').read_bytes()
    cfg['crops'][0].update(id='spatial',split='spatial_test')
    file.write_text(json.dumps(cfg))
    new=prepare.prepare(file,tmp_path/'frozen',stats_path=out/'standardization_stats.json')
    assert (new/'standardization_stats.json').read_bytes()==saved
    cfg['crops'].append(dict(id='test',product='a',window=[0,0,4,4],split='test'))
    file.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='independent'):
        prepare.prepare(file,tmp_path/'bad',stats_path=out/'standardization_stats.json')


def test_quality_runner_records_target_status_and_checksums(tmp_path):
    from types import SimpleNamespace
    import importlib
    # Load through the script's normal import path; no CUDA requested.
    runner=module('experiments/benchmark_quality.py')
    scene=tmp_path/'scene';scene.mkdir();X=np.array([[0],[1],[20],[21]],np.float32)
    np.save(scene/'X.npy',X)
    import hashlib
    (scene/'metadata.json').write_text(json.dumps(dict(X_sha256=hashlib.sha256(X.tobytes()).hexdigest())))
    out=runner.main(['--scene-dir',str(scene),'--output-dir',str(tmp_path/'study'),'--methods','C0','C1',
        '--k','2','--seeds','0','--max-steps','10','--reference-steps','30','--warmups','1','--repeats','1','--evaluation-interval','1'])
    m=json.loads((out/'manifest.json').read_text());assert m['complete'] and m['completed_measured_fits']==2
    raw=[json.loads(s) for s in (out/'raw.jsonl').read_text().splitlines()]
    assert all(r['metadata']['quality_history'] for r in raw)
    for line in (out/'checksums.sha256').read_text().splitlines():
        expected,name=line.split('  ',1);assert hashlib.sha256((out/name).read_bytes()).hexdigest()==expected


def test_quality_runner_applies_and_records_frozen_policy_json(tmp_path):
    runner=module('experiments/benchmark_quality.py')
    scene=tmp_path/'scene';scene.mkdir();X=np.array([[0],[1],[20],[21]],np.float32)
    np.save(scene/'X.npy',X)
    import hashlib
    (scene/'metadata.json').write_text(json.dumps(dict(X_sha256=hashlib.sha256(X.tobytes()).hexdigest())))
    policy=tmp_path/'policy.json'
    policy.write_text(json.dumps({'batch':{'minimum':2,'maximum':4,'quiet_checkpoints':2}}))
    out=runner.main(['--scene-dir',str(scene),'--output-dir',str(tmp_path/'study-policy'),
        '--methods','C0','G3','--k','2','--seeds','0','--max-steps','2','--reference-steps','30',
        '--batch-size','2','--warmups','1','--repeats','1','--evaluation-interval','1',
        '--probe-interval','1','--policy-json',str(policy)])
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['policy_settings']['batch']['minimum']==2
    records=[json.loads(line) for line in (out/'raw.jsonl').read_text().splitlines()]
    adaptive=next(row for row in records if row['method']=='G3')
    assert adaptive['metadata']['batch_policy']['minimum']==2
    assert adaptive['metadata']['batch_policy']['maximum']==4


def test_quality_runner_rejects_ambiguous_policy_effort(tmp_path):
    runner=module('experiments/benchmark_quality.py')
    policy=tmp_path/'policy.json';policy.write_text(json.dumps({'ann_cuda':{'levels':[16,8,4,1],'initial_effort':16}}))
    with pytest.raises(SystemExit):
        runner.main(['--scene-dir',str(tmp_path/'missing-scene'),'--output-dir',str(tmp_path/'out'),
                     '--policy-json',str(policy),'--ann-effort','4'])
    assert not (tmp_path/'out').exists()


def test_gpu_variant_names_preserve_cpu_prototypes():
    suite=module('experiments/run_suite.py')
    assert suite.METHODS['G2']=='cpu_minibatch_ann'
    assert suite.METHODS['G4']=='cpu_adaptive_ann'
    assert suite.METHODS['G5']=='cpu_adaptive_combined'
    assert all(suite.METHODS[m].startswith('cuda_') for m in ('G2CUDA','G4CUDA','G5CUDA'))


def test_policy_cli_validation_precedes_output_creation(tmp_path):
    suite=module('experiments/run_suite.py')
    bad=tmp_path/'bad.json';bad.write_text(json.dumps({'ann_cuda':{'levels':[8,4],'initial_effort':3}}))
    with pytest.raises(ValueError):
        suite.main(['--methods','C0','--sizes','8','--k','1','--seeds','0','--policy-json',str(bad),'--output-dir',str(tmp_path/'out')])
    assert not (tmp_path/'out').exists()
