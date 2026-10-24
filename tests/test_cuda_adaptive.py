import numpy as np
import pytest

from pcap.adaptive_controller import BatchPolicy
from pcap.adaptive_minibatch import fit_adaptive_minibatch_cpu
from pcap.adaptive_cuda import fit_adaptive_minibatch_cuda
from pcap.assignments_exact import assign_exact
from pcap.contracts import FitConfig


@pytest.mark.parametrize('n,d,k,b', [(37,4,3,7), (91,14,4,13), (1,1,1,1)])
def test_cuda_adaptive_matches_cpu(n,d,k,b):
    X = np.random.default_rng(13).normal(size=(n,d)).astype(np.float32)
    cfg = FitConfig(k, 9, 8, 0, b, 1)
    policy = BatchPolicy.for_run(n,b,minimum=1,maximum=n)
    cpu = fit_adaptive_minibatch_cpu(X,X[:k].copy(),cfg,policy=policy)
    gpu = fit_adaptive_minibatch_cuda(X,X[:k].copy(),cfg,policy=policy)
    np.testing.assert_allclose(gpu.centroids,cpu.centroids,rtol=2e-6,atol=1e-7)
    np.testing.assert_array_equal(gpu.labels,assign_exact(X,gpu.centroids).labels)
    assert gpu.sse == pytest.approx(assign_exact(X,gpu.centroids).squared_distances.sum(),rel=1e-12)
    assert gpu.metadata['points_processed'] == sum(r['batch_size'] for r in gpu.history)
    assert gpu.metadata['cluster_observation_counts'] == cpu.metadata['cluster_observation_counts']
    assert gpu.metadata['execution_device'] == 'cuda'


@pytest.mark.parametrize('limit,reason', [(14,'max_steps'),(15,'tolerance')])
def test_cuda_changes_and_frozen_confirmation(limit,reason):
    X = np.ones((64,4),np.float32)
    p = BatchPolicy.for_run(64,16,minimum=2,maximum=32)
    result = fit_adaptive_minibatch_cuda(X,X[:1].copy(),FitConfig(1,0,limit,0,16,1),policy=p)
    assert [r['batch_size'] for r in result.history] == [16]*3+[8]*3+[4]*3+[2]*3+[32]*(limit-12)
    assert result.metadata['termination_reason'] == reason
    assert result.metadata['controller_calls'] == 12
    assert sum(result.metadata['cluster_observation_counts']) == result.metadata['points_processed']
    assert result.sse == 0
