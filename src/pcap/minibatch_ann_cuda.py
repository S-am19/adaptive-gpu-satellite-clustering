"""GPU fixed ANN / ANN-adaptive / joint-adaptive entry points."""
from .controlled_fit import fit_controlled


def fit_minibatch_ann_cuda(X,C0,config,**kwargs):
    return fit_controlled(X,C0,config,device='cuda',mode='fixed',use_ann=True,**kwargs)


def fit_adaptive_ann_cuda(X,C0,config,*,mode='combined',**kwargs):
    if mode not in ('ann_only','combined'):
        raise ValueError('mode must be ann_only or combined')
    return fit_controlled(X,C0,config,device='cuda',mode=mode,use_ann=True,**kwargs)
