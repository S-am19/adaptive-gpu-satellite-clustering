"""Real-device projection ANN quality, exact budget and index lifecycle tests."""
import numpy as np
import pytest
from numba import cuda
from pcap.assignments_ann_cuda import ProjectionCUDAIndex
from pcap.assignments_exact import assign_exact


@pytest.mark.parametrize('d',[4,14])
@pytest.mark.parametrize('k',[1,4,16,64,256])
def test_projection_effort_quality_and_counts(k,d):
    rng=np.random.default_rng(31)
    C=rng.normal(size=(k,d)).astype(np.float32)
    X=rng.normal(size=(127,d)).astype(np.float32)
    backend=ProjectionCUDAIndex(seed=3,exact_threshold=0,query_batch_size=23)
    backend.prepare(C)
    exact=assign_exact(X,C)
    previous=None
    for effort in sorted({1,min(4,k),min(16,k),k}):
        before=backend.timings['centroid_distance_evaluations']
        result=backend.assign(X,C,effort)
        delta=X.astype(float)-C[result.labels].astype(float)
        np.testing.assert_allclose(result.squared_distances,np.sum(delta*delta,axis=1),rtol=1e-13,atol=1e-13)
        assert backend.timings['centroid_distance_evaluations']-before==len(X)*effort
        assert result.labels.dtype==np.int32 and result.labels.flags.c_contiguous
        assert np.all(result.squared_distances>=exact.squared_distances-1e-12)
        if previous is not None: assert np.all(result.squared_distances<=previous+1e-12)
        previous=result.squared_distances
    np.testing.assert_array_equal(result.labels,exact.labels)
    if k>=64:
        low=backend.assign(X,C,1)
        assert np.any(low.squared_distances>exact.squared_distances+1e-8)


def test_small_k_is_explicit_exact_and_refresh_reproducible():
    X=np.array([[1,0],[-1,0],[0,0],[2,0]],np.float32)
    C=np.array([[0,0],[0,0],[2,0]],np.float32)
    a=ProjectionCUDAIndex(seed=7);a.prepare(C)
    r=a.assign(X,C,1)
    assert a.search_mode(1)=='exact_small_k'
    np.testing.assert_array_equal(r.labels,assign_exact(X,C).labels)
    assert r.index_update_time_s>0
    assert a.assign(X,C,1).index_update_time_s==0
    moved=C.copy();moved[0]=[4,0]
    with pytest.raises(ValueError,match='stale'):a.assign(X,moved)
    a.refresh(moved)
    b=ProjectionCUDAIndex(seed=7);b.prepare(moved)
    np.testing.assert_array_equal(a.assign(X,moved).labels,b.assign(X,moved).labels)
    assert a.timings['rebuild_count']==1


def test_resident_stale_version_and_invalid_inputs():
    C=np.zeros((16,4),np.float32);X=np.ones((3,4),np.float32)
    backend=ProjectionCUDAIndex(exact_threshold=0)
    backend.prepare(C)
    dx=cuda.to_device(X);ix=cuda.to_device(np.arange(3,dtype=np.int32))
    labels=cuda.device_array(3,np.int32);dist=cuda.device_array(3,np.float64)
    with pytest.raises(ValueError,match='stale'):
        backend.assign_device(dx,ix,labels,dist,version=1)
    backend.refresh_device(cuda.to_device(C),1)
    backend.assign_device(dx,ix,labels,dist,version=1,effort=1)
    np.testing.assert_array_equal(labels.copy_to_host(),np.zeros(3,np.int32))
    with pytest.raises(ValueError):backend.assign(X,C,0)
    with pytest.raises(TypeError):backend.assign(X.astype(float),C)
    with pytest.raises(ValueError):backend.prepare(np.full((4,4),np.nan,np.float32))


@pytest.mark.parametrize('mode',['full','fixed','batch_only','ann_only','combined'])
def test_gpu_controlled_matches_cpu_at_exhaustive_budget(mode):
    from pcap.controlled_fit import fit_controlled
    from pcap.contracts import FitConfig
    X=np.random.default_rng(27).normal(size=(67,4)).astype(np.float32);C=X[:4].copy()
    use_ann=mode in ('ann_only','combined')
    cfg=FitConfig(4,3,7,0,None if mode=='full' else 8,2,4 if use_ann else None)
    gpu=fit_controlled(X,C,cfg,device='cuda',mode=mode,use_ann=use_ann,evaluation_interval=2)
    # Force full-budget projection ANN while preserving GPU controller state.
    cpu_cfg=FitConfig(4,3,7,0,None if mode=='full' else 8,2)
    cpu=fit_controlled(X,C,cpu_cfg,mode='batch_only' if mode=='combined' else 'fixed' if mode=='ann_only' else mode,evaluation_interval=2)
    np.testing.assert_allclose(gpu.centroids,cpu.centroids,rtol=2e-6,atol=1e-7)
    np.testing.assert_array_equal(gpu.labels,assign_exact(X,gpu.centroids).labels)
    assert gpu.sse==pytest.approx(assign_exact(X,gpu.centroids).squared_distances.sum(),rel=1e-12)
    assert gpu.metadata['execution_device']=='cuda'
    assert gpu.metadata['points_processed']==sum(h['batch_size'] for h in gpu.history)


def test_real_approximate_updates_independent_numpy_reference():
    from pcap.controlled_fit import fit_controlled
    from pcap.contracts import FitConfig
    X=np.random.default_rng(7).normal(size=(80,4)).astype(np.float32);C=X[:16].copy()
    cfg=FitConfig(16,3,4,0,19,100,4)
    result=fit_controlled(X,C,cfg,device='cuda',use_ann=True,exact_small_k=0)
    direction=np.random.default_rng(3).normal(size=4);direction/=np.linalg.norm(direction)
    rng=np.random.default_rng(3);sums=np.zeros(C.shape,float);counts=np.zeros(16,np.int64)
    for _ in range(4):
        batch=X[rng.choice(len(X),19,replace=False)].astype(float)
        projections=np.sum(C.astype(float)*direction,axis=1)
        selected=np.argsort(abs(np.sum(batch*direction,axis=1)[:,None]-projections),axis=1,kind='stable')[:,:4]
        distances=np.sum((batch[:,None]-C[selected].astype(float))**2,axis=2)
        labels=selected[np.arange(19),distances.argmin(axis=1)]
        for j in range(16):
            values=batch[labels==j]
            if len(values):sums[j]+=values.sum(axis=0);counts[j]+=len(values);C[j]=sums[j]/counts[j]
    np.testing.assert_allclose(result.centroids,C,rtol=2e-6,atol=1e-7)
    np.testing.assert_array_equal(result.metadata['cluster_observation_counts'],counts)
    assert result.metadata['index_timings']['centroid_distance_evaluations']==4*19*4
    assert result.metadata['index_timings']['rebuild_count']==4
    assert result.metadata['search_mode']=='approximate_projection_window'


@pytest.mark.parametrize('limit,reason',[(5,'max_steps'),(6,'tolerance')])
def test_gpu_joint_confirmation_and_quality_target(limit,reason):
    from pcap.controlled_fit import fit_controlled
    from pcap.contracts import FitConfig
    from pcap.adaptive_controller import BatchPolicy
    from pcap.adaptive_ann_controller import ANNEffortPolicy
    X=np.ones((8,4),np.float32)
    p=BatchPolicy(2,8,2,quiet_checkpoints=100)
    a=ANNEffortPolicy(levels=(8,4,1),initial_effort=4,quiet_checkpoints=100)
    r=fit_controlled(X,X[:1].copy(),FitConfig(1,0,limit,0,2,1,4),device='cuda',mode='combined',use_ann=True,batch_policy=p,ann_policy=a)
    assert r.metadata['termination_reason']==reason and r.metadata['confirmation_checks']==limit-3
    assert all(h['batch_size']==8 and h['ann_effort']==8 and h['adaptation_frozen'] for h in r.history[3:])
    target=fit_controlled(X,X[:1].copy(),FitConfig(1,0,10,0,2),device='cuda',evaluation_interval=1,target_sse=0)
    assert target.metadata['termination_reason']=='quality_target' and target.metadata['steps']==0
