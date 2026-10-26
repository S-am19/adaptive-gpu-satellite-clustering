import numpy as np
import pytest
from pcap.controlled_fit import fit_controlled
from pcap.contracts import FitConfig
from pcap.adaptive_controller import BatchPolicy
from pcap.assignments_exact import assign_exact
from pcap.kmeans_cpu import fit_cpu


def test_full_reference_matches_legacy_and_quality_cadence():
    X=np.random.default_rng(1).normal(size=(31,4)).astype(np.float32);C=X[:3].copy()
    cfg=FitConfig(3,0,7,0)
    a=fit_controlled(X,C,cfg,mode='full',evaluation_interval=2)
    b=fit_cpu(X,C,cfg)
    np.testing.assert_allclose(a.centroids,b.centroids,rtol=1e-6,atol=1e-7)
    assert a.sse==pytest.approx(b.sse,rel=1e-12)
    assert a.metadata['quality_history'][0]['step']==0
    assert a.metadata['quality_history'][-1]['step']==a.metadata['steps']
    assert all(q['step']%2==0 or q is a.metadata['quality_history'][-1] for q in a.metadata['quality_history'])


def test_target_exit_is_not_convergence_and_zero_target():
    X=np.ones((8,2),np.float32)
    r=fit_controlled(X,X[:1].copy(),FitConfig(1,0,10,0,2,1),evaluation_interval=2,target_sse=0)
    assert r.metadata['steps']==0 and r.metadata['termination_reason']=='quality_target'
    assert r.metadata['target_satisfied'] and not r.metadata['convergence_satisfied']
    assert r.metadata['points_processed']==0 and r.sse==0
    assert r.metadata['time_to_target_s']>=r.timings['quality_evaluation_s']


def test_target_unreached_and_sampling_unchanged_by_quality_monitor():
    X=np.random.default_rng(2).normal(size=(63,4)).astype(np.float32);cfg=FitConfig(3,9,7,0,8,2)
    a=fit_controlled(X,X[:3].copy(),cfg,evaluation_interval=1,target_sse=0)
    b=fit_controlled(X,X[:3].copy(),cfg)
    np.testing.assert_array_equal(a.centroids,b.centroids)
    assert not a.metadata['target_satisfied'] and a.metadata['time_to_target_s'] is None
    assert a.metadata['termination_reason']=='max_steps'
    assert a.sse==pytest.approx(assign_exact(X,a.centroids).squared_distances.sum())


def test_persistent_variable_counts_and_frozen_confirmation():
    X=np.ones((64,4),np.float32);p=BatchPolicy.for_run(64,16,minimum=2,maximum=32)
    r=fit_controlled(X,X[:1].copy(),FitConfig(1,0,15,0,16,1),mode='batch_only',batch_policy=p)
    assert [h['batch_size'] for h in r.history]==[16]*3+[8]*3+[4]*3+[2]*3+[32]*3
    assert sum(r.metadata['cluster_observation_counts'])==186==r.metadata['points_processed']
    assert r.metadata['termination_reason']=='tolerance' and r.metadata['confirmation_checks']==3
    assert all(h['adaptation_frozen'] for h in r.history[12:])


@pytest.mark.parametrize('kwargs',[dict(target_sse=1),dict(evaluation_interval=0),dict(device='fake'),dict(mode='fake'),dict(target_sse=-1,evaluation_interval=1)])
def test_invalid_quality_settings(kwargs):
    X=np.ones((4,2),np.float32)
    with pytest.raises((ValueError,TypeError)):
        fit_controlled(X,X[:1].copy(),FitConfig(1,0,2,0,1),**kwargs)
