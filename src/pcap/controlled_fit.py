"""Instrumented persistent fits for controlled GPU ANN ablations/time-to-quality.

Legacy fit functions are unchanged. This execution path explicitly includes
checkpoint quality evaluation overhead; do not pool its timings with legacy fits.
"""
from dataclasses import asdict, replace
from time import perf_counter

import numpy as np

from .adaptive_controller import BatchPolicy, Checkpoint, initial_state, transition
from .adaptive_ann_controller import ANNEffortPolicy, initial_ann_state, transition_ann
from .assignments_exact import _assign, _accumulate
from .contracts import FitConfig, RunResult, validate_points_centroids, validate_integer
from .minibatch import _movement, _sample_batch_indices


class _CPUState:
    def __init__(self, X, C0, capacity, seed, use_ann):
        self.X, self.C = X, C0.copy()
        self.sums, self.counts = np.zeros(C0.shape, np.float64), np.zeros(len(C0), np.int64)
        self.index = None
        self.index_s = 0.
        if use_ann:
            from .assignments_ann import HNSWCPUBackend
            self.index = HNSWCPUBackend(seed=seed)
            start = perf_counter(); self.index.prepare(self.C); self.index_s += perf_counter()-start

    def assign(self, indices, effort):
        self.batch = np.ascontiguousarray(self.X[indices])
        if self.index:
            r = self.index.assign(self.batch, self.C, effort)
            self.labels = r.labels
            return float(np.sum(r.squared_distances, dtype=np.float64))
        self.labels, distances = _assign(self.batch, self.C)
        return float(distances.sum())

    def update(self, full):
        sums, counts = _accumulate(self.batch, self.labels, len(self.C))
        if full:
            self.sums.fill(0); self.counts.fill(0)
        self.sums += sums; self.counts += counts
        before = self.C.copy()
        active = counts > 0
        self.C[active] = (self.sums[active] / self.counts[active, None]).astype(np.float32)
        if not np.isfinite(self.C).all() or not np.isfinite(self.sums).all():
            raise FloatingPointError('Nonfinite CPU centroid statistics')
        movement = _movement(self.C, before)
        if self.index:
            start = perf_counter(); self.index.refresh(self.C); self.index_s += perf_counter()-start
        return movement

    def probe(self, indices, previous):
        labels, distances = _assign(np.ascontiguousarray(self.X[indices]), self.C)
        return labels, float(distances.sum()), _movement(self.C, previous)

    def checkpoint_copy(self):
        return self.C.copy()

    def objective(self):
        _, distances = _assign(self.X, self.C)
        return float(distances.sum())

    def final(self):
        labels, distances = _assign(self.X, self.C)
        return self.C.copy(), labels, self.counts.copy(), float(distances.sum())


class _GPUState:
    def __init__(self, X, C0, capacity, seed, use_ann):
        from numba import cuda
        from .adaptive_cuda import _DeviceUpdates
        self.cuda = cuda
        self.device = _DeviceUpdates(X, C0, capacity, min(len(X), 1024))
        self.index, self.index_s, self.version = None, 0., 0
        if use_ann:
            from .assignments_ann_cuda import ProjectionCUDAIndex
            self.index = ProjectionCUDAIndex(seed=seed)
            start=perf_counter(); self.index.prepare(C0); self.index_s+=perf_counter()-start
        self.initial = True

    def assign(self, indices, effort):
        device = self.device
        if not self.index:
            return float(device.assign(indices).squared_distances[0])
        device.b = len(indices)
        device.active_indices = device.indices[:device.b]
        device.active_labels = device.labels[:device.b]
        device.active_distances = device.distances[:device.b]
        device.active_indices.copy_to_device(np.asarray(indices, np.int32))
        self.index.assign_device(device.X, device.active_indices, device.active_labels,
            device.active_distances, version=self.version, effort=effort)
        return device.reduce(device.active_distances)

    def update(self, full):
        if full:
            from .kmeans_cuda import _reset_stats_kernel, _blocks, _THREADS
            _reset_stats_kernel[_blocks(self.device.C.size), _THREADS](self.device.sums, self.device.counts)
        movement = self.device.update()
        if self.index:
            self.version += 1
            start=perf_counter(); self.index.refresh_device(self.device.C, self.version); self.index_s+=perf_counter()-start
        return movement

    def probe(self, indices, previous):
        if self.initial:
            self.device.set_probe(indices)
            self.initial=False
        result=self.device.probe()
        return result.labels, float(result.squared_distances.sum()), self.device.checkpoint_movement

    def checkpoint_copy(self):
        return None  # Device checkpoint centroids are copied by probe().

    def objective(self):
        from .minibatch_cuda import _assign_kernel, _blocks, _THREADS
        d=self.device
        _assign_kernel[_blocks(len(d.X)), _THREADS](d.X,d.C,d.final_labels,d.final_distances)
        return d.reduce(d.final_distances)

    def final(self):
        C, result, counts, sse=self.device.final()
        return C,result.labels,counts,sse


def fit_controlled(X, C0, config, *, device='cpu', mode='fixed', use_ann=False,
                   batch_policy=None, ann_policy=None, evaluation_interval=None,
                   target_sse=None, exact_small_k=8):
    """New instrumented fit; same FitConfig/RunResult, no shared-contract change.

    mode: full (Lloyd), fixed Mini-Batch, batch_only, ann_only, combined.
    GPU ANN effort counts projection candidates; CPU ANN effort is HNSW ef.
    Exact full-data quality is measured initially, periodically and at exit.
    A target exit is quality_target, never mislabeled heuristic convergence.
    """
    wall=perf_counter()
    if not isinstance(config,FitConfig): raise TypeError('config must be FitConfig')
    validate_points_centroids(X,C0)
    n,d=X.shape; k=len(C0)
    if config.k!=k or k>n: raise ValueError('C0/config.k must agree, with k <= N')
    if device not in ('cpu','cuda') or mode not in ('full','fixed','batch_only','ann_only','combined'):
        raise ValueError('Unknown device or update mode')
    if mode=='full' and (use_ann or config.batch_size is not None or config.ann_effort is not None): raise ValueError('Full Lloyd requires exact assignment and batch_size=ann_effort=None')
    adaptive_batch=mode in ('batch_only','combined')
    adaptive_ann=mode in ('ann_only','combined')
    if adaptive_ann and not use_ann: raise ValueError('ANN adaptation requires an ANN backend')
    if not use_ann and config.ann_effort is not None: raise ValueError('Exact assignment requires ann_effort=None')
    if mode!='full' and (config.batch_size is None or config.batch_size>n): raise ValueError('batch_size required in 1..N')
    interval=int(config.probe_interval)
    if evaluation_interval is not None: validate_integer('evaluation_interval',evaluation_interval)
    if target_sse is not None:
        if isinstance(target_sse,(bool,np.bool_)) or not np.isscalar(target_sse) or not np.isfinite(target_sse) or target_sse<0:
            raise ValueError('target_sse must be finite and nonnegative')
        if evaluation_interval is None: raise ValueError('Target requires full-data evaluation schedule')
    validate_integer('exact_small_k',exact_small_k,0)
    supplied_batch=batch_policy is not None; supplied_ann=ann_policy is not None
    if adaptive_batch:
        batch_policy=batch_policy or BatchPolicy.for_run(n,config.batch_size)
        if not isinstance(batch_policy,BatchPolicy) or batch_policy.initial_batch!=config.batch_size or batch_policy.maximum>n:
            raise ValueError('Batch policy must match configuration and dataset')
    elif batch_policy is not None: raise ValueError('Batch policy supplied without batch adaptation')
    maximum=int(batch_policy.maximum) if adaptive_batch else (n if mode=='full' else int(config.batch_size))
    if int(config.max_steps)*maximum>np.iinfo(np.int64).max: raise ValueError('Cumulative observations exceed int64 capacity')
    effort=config.ann_effort if config.ann_effort is not None else (min(k,16) if device=='cuda' else 64)
    if adaptive_ann:
        if ann_policy is None:
            levels=tuple(sorted({k,min(k,16),max(1,min(k,16)//2),1,int(effort)},reverse=True)) if device=='cuda' else (64,32,16,8)
            ann_policy=ANNEffortPolicy(levels=levels,initial_effort=effort)
        if not isinstance(ann_policy,ANNEffortPolicy) or ann_policy.initial_effort!=effort:
            raise ValueError('ANN policy must match initial effort')
    elif ann_policy is not None: raise ValueError('ANN policy supplied without ANN adaptation')
    start=perf_counter()
    state=(_GPUState if device=='cuda' else _CPUState)(X,C0,maximum,int(config.seed),use_ann)
    setup_s=perf_counter()-start
    if device=='cuda' and state.index: state.index.exact_threshold=int(exact_small_k)
    rng=np.random.default_rng(config.seed)
    probe_rng=np.random.default_rng(np.random.SeedSequence([config.seed,1]))
    indices=np.sort(probe_rng.choice(n,min(n,1024),replace=False))
    values=X[indices].astype(np.float64)
    scale=max(float(np.sqrt(np.mean(np.sum((values-values.mean(axis=0))**2,axis=1)))),1e-12)
    start=perf_counter();previous_labels,probe_sse,_=state.probe(indices,C0);monitor_s=perf_counter()-start
    previous_C=state.checkpoint_copy();q=probe_sse/len(indices)
    if adaptive_batch and not supplied_batch: batch_policy=replace(batch_policy,objective_floor=1e-12*scale*scale)
    if adaptive_ann and not supplied_ann: ann_policy=replace(ann_policy,objective_floor=1e-12*scale*scale)
    b_state=initial_state(batch_policy,q) if adaptive_batch else None
    a_state=initial_ann_state(ann_policy,q) if adaptive_ann else None
    history=[];quality=[];observations=0;stable_count=confirm_count=0
    confirming=converged=target_met=False;confirmation_q=None;reason='max_steps'
    sampling_s=assignment_s=update_s=controller_s=quality_s=0.
    first_target_s=None;quality_points=0;probe_checks=0;step=0
    def evaluate(step):
        nonlocal quality_s,quality_points,target_met,first_target_s
        start=perf_counter();objective=state.objective();quality_s+=perf_counter()-start;quality_points+=n
        elapsed=perf_counter()-wall
        hit=target_sse is not None and objective<=target_sse
        if hit and first_target_s is None: first_target_s=elapsed
        target_met=target_met or hit
        quality.append(dict(step=step,elapsed_s=elapsed,full_data_sse=objective,points_processed=observations,target_satisfied=hit))
        return hit
    if evaluation_interval is not None and evaluate(0): reason='quality_target'
    if not target_met:
        for step in range(1,int(config.max_steps)+1):
            batch=int(b_state.batch_size) if b_state else (n if mode=='full' else int(config.batch_size))
            effort_now=int(a_state.effort) if a_state else (effort if use_ann else None)
            phase='confirmation' if confirming else 'adapting'
            start=perf_counter();sample=_sample_batch_indices(rng,n,batch);sampling_s+=perf_counter()-start
            start=perf_counter();batch_sse=state.assign(sample,effort_now);assignment_s+=perf_counter()-start
            start=perf_counter();movement=state.update(mode=='full');update_s+=perf_counter()-start
            observations+=batch
            entry=dict(step=step,phase=phase,batch_size=batch,ann_effort=effort_now,
                       points_processed=observations,batch_sse_before_update=batch_sse,
                       centroid_displacement=movement,convergence_checked=False)
            if mode=='full': converged=movement<=config.tolerance
            elif step%interval==0:
                start=perf_counter();labels,probe_sse,displacement=state.probe(indices,previous_C)
                changes=float(np.mean(labels!=previous_labels));previous_labels=labels;previous_C=state.checkpoint_copy()
                checkpoint=Checkpoint(probe_sse/len(indices),displacement/scale,changes);probe_checks+=1
                monitor_s+=perf_counter()-start
                stable=displacement<=config.tolerance and changes==0
                start=perf_counter();decisions={}
                if mode=='fixed':
                    stable_count=stable_count+1 if stable else 0;converged=stable_count>=3
                elif confirming:
                    policies=[p for p in (batch_policy,ann_policy) if p is not None]
                    allowed=min(confirmation_q+p.deterioration*max(confirmation_q,p.objective_floor) for p in policies)
                    confirm_count=confirm_count+1 if stable and checkpoint.objective<=allowed else 0
                    converged=confirm_count>=3
                else:
                    entry.update(batch_state_before=asdict(b_state) if b_state else None,ann_state_before=asdict(a_state) if a_state else None)
                    if adaptive_batch:
                        b_state,decision=transition(batch_policy,b_state,checkpoint);decisions['batch']=dict(asdict(decision),changed=decision.changed)
                    if adaptive_ann:
                        a_state,decision=transition_ann(ann_policy,a_state,checkpoint);decisions['ann']=dict(asdict(decision),changed=decision.changed)
                    entry.update(batch_state_after=asdict(b_state) if b_state else None,ann_state_after=asdict(a_state) if a_state else None)
                    safe=all(not v['changed'] and not v['deteriorated'] for v in decisions.values())
                    stable_count=stable_count+1 if stable and safe else 0
                    if stable_count>=3:
                        confirming=True;confirmation_q=checkpoint.objective;confirm_count=0
                        if b_state: b_state=replace(b_state,batch_size=int(batch_policy.maximum))
                        if a_state: a_state=replace(a_state,effort=int(ann_policy.maximum))
                        entry['confirmation_entered']=True
                entry.update(convergence_checked=True,checkpoint_displacement=displacement,
                             probe_sse=probe_sse,normalized_checkpoint_displacement=checkpoint.movement,probe_label_change_fraction=changes,
                             controller_decisions=decisions,stable_checks=stable_count,
                             confirmation_checks=confirm_count,adaptation_frozen=phase=='confirmation',
                             next_batch_size=int(b_state.batch_size) if b_state else batch,
                             next_ann_effort=int(a_state.effort) if a_state else effort_now)
                controller_s+=perf_counter()-start
            history.append(entry)
            # Always check quality on a terminal update, even off cadence.
            terminal=converged or step==int(config.max_steps)
            if evaluation_interval is not None and (step%evaluation_interval==0 or terminal):
                if evaluate(step): reason='quality_target';break
            if converged: reason='tolerance';break
    start=perf_counter();C,labels,counts,final_sse=state.final();final_s=perf_counter()-start
    ledger=state.index.timings if state.index else {}
    metadata=dict(algorithm=f'controlled_{device}_{mode}_{"ann" if use_ann else "exact"}',
        execution_device=device,config=asdict(config),mode=mode,final_exact_sse=final_sse,
        steps=step,termination_reason=reason,convergence_satisfied=converged,
        target_sse=target_sse,target_satisfied=target_met,first_observed_target_s=first_target_s,
        quality_history=quality,quality_evaluation_points=quality_points,
        points_processed=observations,cluster_observation_counts=counts.tolist(),
        cumulative_statistics=mode!='full',probe_indices=indices.tolist(),probe_checks=probe_checks,
        probe_role='overlapping training diagnostic; exact monitoring',probe_scale=scale,initial_probe_objective=q,
        confirmation_checks=confirm_count,confirming_phase_entered=confirming,
        batch_policy=asdict(batch_policy) if batch_policy else None,
        ann_policy=asdict(ann_policy) if ann_policy else None,index_timings=ledger,
        ann_effort_units='projection candidate count' if device=='cuda' and use_ann else 'HNSW ef' if use_ann else None,
        search_mode=state.index.search_mode(int(a_state.effort) if a_state else effort) if device=='cuda' and use_ann else 'HNSW CPU' if use_ann else 'exact',
        initial_search_mode=state.index.search_mode(effort) if device=='cuda' and use_ann else 'HNSW CPU' if use_ann else 'exact',
        final_batch_state=asdict(b_state) if b_state else None,final_ann_state=asdict(a_state) if a_state else None,
        timing_policy='instrumented persistent fit; includes setup/index/transfers and full-data quality checkpoints; do not pool with legacy fits',
        final_exact_evaluation=True)
    if device=='cuda': metadata.update(gpu=str(state.cuda.get_current_device().name),estimated_device_buffer_bytes=state.device.estimated_bytes,index_build_device='cpu' if use_ann else None)
    if target_met and final_sse>target_sse:
        raise FloatingPointError('Final exact SSE violates the recorded quality target')
    exact_points=(probe_checks+1)*len(indices)+quality_points+n
    metadata['assignment_points_processed']=observations+exact_points
    metadata['centroid_distance_evaluations']=ledger.get('centroid_distance_evaluations',0)+exact_points*k if device=='cuda' and use_ann else (observations+exact_points)*k if not use_ann else None
    if device=='cuda' and use_ann:
        metadata['estimated_device_buffer_bytes']+=state.index.device_index_bytes
    timings=dict(wall_s=perf_counter()-wall,setup_s=setup_s,sampling_s=sampling_s,
        batch_assignment_s=assignment_s,accumulation_update_s=update_s,monitoring_s=monitor_s,
        controller_s=controller_s,quality_evaluation_s=quality_s,final_assignment_s=final_s,
        index_maintenance_s=state.index_s)
    timings['compute_s']=timings['wall_s']-setup_s
    metadata['time_to_target_s']=timings['wall_s'] if target_met else None
    if quality and quality[-1]['step']==step:
        quality[-1].update(elapsed_s=timings['wall_s'],full_data_sse=final_sse,final_output=True)
    metadata['timing_scope']='index_maintenance_s overlaps setup/update; other named work components are subsets of wall; compute excludes setup'
    return RunResult(C,labels,history,timings,metadata)
