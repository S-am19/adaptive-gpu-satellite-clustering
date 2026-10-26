from pathlib import Path
import subprocess
import numpy as np
import pytest
from pcap.contracts import FitConfig
from pcap.kmeans_cpu import fit_cpu
from pcap.kmeans_openmp import fit_openmp
from pcap.assignments_exact import assign_exact


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    out=tmp_path_factory.mktemp('openmp')/'baseline'
    source=Path(__file__).resolve().parents[1]/'native/kmeans_openmp.cpp'
    subprocess.run(['g++','-O3','-std=c++17','-fopenmp','-ffp-contract=off',str(source),'-o',str(out)],check=True,timeout=30)
    return out


@pytest.mark.parametrize('threads',[1,2,4])
@pytest.mark.parametrize('n,d,k',[(17,1,1),(257,4,4),(63,14,7)])
def test_openmp_matches_cpu(native,threads,n,d,k):
    X=np.random.default_rng(17).normal(size=(n,d)).astype(np.float32);C=X[:k].copy();cfg=FitConfig(k,0,12,0)
    a=fit_openmp(X,C,cfg,threads=threads,executable=native);b=fit_cpu(X,C,cfg)
    np.testing.assert_allclose(a.centroids,b.centroids,rtol=2e-6,atol=1e-7)
    np.testing.assert_array_equal(a.labels,assign_exact(X,a.centroids).labels)
    assert a.sse==pytest.approx(assign_exact(X,a.centroids).squared_distances.sum(),rel=1e-12)
    assert a.metadata['threads_actual']<=threads
    assert a.timings['native_fit_s']<=a.timings['process_s']<=a.timings['wall_s']
    assert all(y['sse']<=x['sse']+1e-10 for x,y in zip(a.history,a.history[1:]))


def test_openmp_ties_empty_and_runtime_limit(native,monkeypatch):
    X=np.ones((10,4),np.float32);C=np.array([[1]*4,[1]*4,[100]*4],np.float32)
    monkeypatch.setenv('OMP_THREAD_LIMIT','1')
    r=fit_openmp(X,C,FitConfig(3,0,5,0),threads=4,executable=native)
    np.testing.assert_array_equal(r.centroids,C)
    np.testing.assert_array_equal(r.labels,np.zeros(10,np.int32))
    assert r.metadata['steps']==1 and r.metadata['threads_actual']==1 and r.sse==0


def test_openmp_explicit_missing_executable_and_invalid_config(tmp_path):
    X=np.ones((4,2),np.float32)
    with pytest.raises(FileNotFoundError):fit_openmp(X,X[:1].copy(),FitConfig(1,0,2,0),executable=tmp_path/'absent')
    with pytest.raises(ValueError):fit_openmp(X,X[:1].copy(),FitConfig(1,0,2,0,batch_size=1))
