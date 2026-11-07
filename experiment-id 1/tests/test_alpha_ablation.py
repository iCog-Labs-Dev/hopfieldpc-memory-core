"""Alpha isolation, independent replay, positive controls, and epoch resume."""
import copy
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import admission_ablation as admission
import alpha_ablation as study
from alpha_memory import (CachedOccupancyBank, ResidualEventWriter, apply_bank_batch,
                          audit_alpha_log, decode_vectors, make_alpha_bank)
from mnist_online_memory import make_bank, train_step, validate_config
from online_memory_artifacts import EventWriter, diagnostic_mode, read_events
from test_admission_ablation import parent_fixture, fixture_model, toy_bank, toy_model


@pytest.fixture(autouse=True)
def deterministic_cpu(monkeypatch):
    torch.set_num_threads(1)
    identity = lambda: {'commit': 'fixture', 'source_sha256': 'fixture'}
    monkeypatch.setattr(admission, 'source_identity', identity)
    monkeypatch.setattr(study, 'source_identity', identity)
    monkeypatch.setattr(study, 'archive_source', lambda root, source: None)


@pytest.mark.parametrize('alpha',[.03,.1,.3])
def test_cached_occupancy_preserves_every_original_policy_operation(alpha):
    cfg = validate_config({'num_classes':3,'max_memories_per_class':3,'ema_alpha':alpha,'merge_distance_threshold':.5})
    original, cached = make_bank(cfg,7), make_alpha_bank(cfg,7,alpha)
    for b in (original,cached): b.initialize_center(torch.zeros(7))
    generator = torch.Generator().manual_seed(11)
    for step in range(90):
        label = step%3
        z = torch.randn(7,generator=generator)+torch.eye(3,7)[label]*3
        p = original.plan_candidate(z,label,bootstrap=not original.ready)
        q = cached.plan_candidate(z,label,bootstrap=not cached.ready)
        assert (p.action,p.reason,p.slot,p.distance)==(q.action,q.reason,q.slot,q.distance)
        if p.is_write:
            original.commit(p,example_id=step,step=step,epoch=2)
            cached.commit(q,example_id=step,step=step,epoch=2)
        admission.require_equal(original.state_dict(),cached.state_dict())
        assert original.occupancy()==cached.occupancy()
    counts = cached.occupancy();counts[0]=999
    assert cached.occupancy()[0]!=999
    assert not list(cached.parameters())


def test_cache_is_invalidated_by_loading_same_size_different_class_counts():
    cfg=validate_config({'num_classes':2,'max_memories_per_class':3})
    a,b=make_alpha_bank(cfg,2,.1),make_bank(cfg,2)
    for bank in (a,b):
        bank.initialize_center(torch.zeros(2))
        for c,v in enumerate(([1.,0.],[-1.,0.])):
            bank.commit(bank.plan_candidate(torch.tensor(v),c,bootstrap=True),example_id=c,step=0,epoch=1)
    a.commit(a.plan_candidate(torch.tensor([0.,1.]),0),example_id=2,step=1,epoch=2)
    b.commit(b.plan_candidate(torch.tensor([0.,1.]),1),example_id=2,step=1,epoch=2)
    assert a.occupancy()==[2,1]
    a.load_state_dict(b.state_dict())
    assert a.occupancy()==[1,2]


def test_shared_step_has_identical_cnn_gradients_rng_and_control_bank(tmp_path):
    cfg,initial=toy_bank()
    cfg=validate_config(cfg|{'ema_alpha':.1})
    banks={a:make_alpha_bank(cfg,2,a) for a in [.03,.1,.3]}
    for b in banks.values(): b.load_state_dict(initial.state_dict())
    reference=make_bank(cfg,2);reference.load_state_dict(initial.state_dict())
    model,other=toy_model(),toy_model()
    first,second=torch.optim.Adam(model.parameters(),lr=.001),torch.optim.Adam(other.parameters(),lr=.001)
    x,y=torch.tensor([[3.,0.],[0.,3.],[3.,0.],[0.,3.]]),torch.tensor([0,1,1,0])
    for step in range(4):
        rng=torch.get_rng_state().clone()
        loss,stats=study.shared_step(model,first,banks,x,y,torch.arange(4),cfg=cfg,writers={},
                     epoch=2,batch=step,step=step,seed=0,run_id='fixture',costs={},bank_costs={a:{} for a in banks})
        assert torch.equal(rng,torch.get_rng_state())
        old=train_step(other,reference,second,x,y,torch.arange(4),cfg=cfg,epoch=2,batch=step,step=step,
                       seed=0,run_id='fixture',admission_policy='all_direct_correct')
        assert loss==old['loss'] and stats[.1]['counts']==old['counts']
        admission.require_equal(model.state_dict(),other.state_dict())
        admission.require_equal(first.state_dict(),second.state_dict())
        admission.require_equal(banks[.1].state_dict(),reference.state_dict())
        for p,q in zip(model.parameters(),other.parameters()):assert torch.equal(p.grad,q.grad)
    assert len({b.values.data_ptr() for b in banks.values()})==3
    assert len({b.center.data_ptr() for b in banks.values()})==3
    assert not torch.equal(banks[.03].values,banks[.3].values)


def test_independent_audit_checks_threshold_boundary_and_detects_corrupt_vectors(tmp_path):
    candidate=torch.tensor([.8,.6])
    threshold=float(1-F.normalize(torch.tensor([1.,0.]),dim=-1)@F.normalize(candidate,dim=-1))
    cfg=validate_config({'num_classes':2,'merge_distance_threshold':threshold})
    bank=make_alpha_bank(cfg,2,.3);bank.initialize_center(torch.zeros(2))
    writer=EventWriter(tmp_path,3,2);writer.event(dict(type='center',center=[0.,0.]))
    model=toy_model()
    with diagnostic_mode(model):
        apply_bank_batch(model,bank,torch.tensor([[1.,0.],[-1.,0.]]),torch.tensor([0,1]),
                         torch.tensor([0,1]),torch.tensor([0,1]),cfg=cfg,epoch=2,batch=0,step=0,seed=0,run_id='fixture',writer=writer)
        apply_bank_batch(model,bank,candidate[None],torch.tensor([0]),torch.tensor([0]),torch.tensor([2]),
                         cfg=cfg,epoch=2,batch=1,step=1,seed=0,run_id='fixture',writer=writer)
    cursor=writer.flush();targets=torch.tensor([0,1,0]);ids=torch.arange(3)
    audit=audit_alpha_log(tmp_path,cursor,cfg,bank,ids,targets)
    assert audit['replay_exact'] and audit['merge']==1 and audit['native_boundary_checks']>0
    path=tmp_path/cursor['vector_files'][0]
    with np.load(path) as data: damaged={k:data[k].copy() for k in data.files}
    damaged['new'][0,0]+=1
    with path.open('wb') as stream:np.savez_compressed(stream,**damaged)
    with pytest.raises(AssertionError):audit_alpha_log(tmp_path,cursor,cfg,bank,ids,targets)


def test_residual_codec_preserves_non_ema_records_and_detects_bit_corruption(tmp_path):
    writer=ResidualEventWriter(tmp_path,10,10)
    writer.write(dict(entry_id=0,alpha=.3,candidate=torch.tensor([1.,2.]),old=None,new=torch.tensor([1.,2.])))
    writer.write(dict(entry_id=0,alpha=.3,candidate=torch.tensor([3.,4.]),
                      old=torch.tensor([1.5,2.]),new=torch.tensor([99.,-7.])))
    cursor=writer.flush();events=list(read_events(tmp_path,cursor))
    with np.load(tmp_path/cursor['vector_files'][0]) as chunk: data={k:chunk[k].copy() for k in chunk.files}
    _,old,new=decode_vectors(events[0],data,np.zeros(2,dtype=np.float32))
    assert old is None and np.array_equal(new,np.array([1.,2.],dtype=np.float32))
    _,old,actual=decode_vectors(events[1],data,new)
    assert np.array_equal(old,np.array([1.5,2.],dtype=np.float32))
    assert np.array_equal(actual,np.array([99.,-7.],dtype=np.float32))
    assert np.any(data['old_xor'][1]) and np.any(data['new_xor'][1])
    data['new_xor'][1,0]^=np.uint32(1)
    with pytest.raises(ValueError,match='checksum'):decode_vectors(events[1],data,new)


@pytest.mark.parametrize('alpha',[.03,.1,.3])
def test_residual_and_dense_logs_recover_the_same_vectors_and_bank(tmp_path,alpha):
    cfg=validate_config({'num_classes':2,'merge_distance_threshold':2})
    banks=[make_alpha_bank(cfg,2,alpha) for _ in range(2)]
    for b in banks:b.initialize_center(torch.zeros(2))
    writers=[EventWriter(tmp_path/'dense',7,3),ResidualEventWriter(tmp_path/'residual',7,3)]
    for writer in writers:writer.event(dict(type='center',center=[0.,0.]))
    model=toy_model();targets=torch.tensor([0,1]*4)
    for batch in range(4):
        x=torch.tensor([[3.015123+batch*.1,.012345],[.012345,3.015123+batch*.1]])
        with diagnostic_mode(model):
            for bank,writer in zip(banks,writers):
                apply_bank_batch(model,bank,x,torch.tensor([0,1]),torch.tensor([0,1]),torch.arange(batch*2,batch*2+2),
                     cfg=cfg,epoch=2,batch=batch,step=batch,seed=0,run_id='fixture',writer=writer)
    admission.require_equal(banks[0].state_dict(),banks[1].state_dict())
    cursors=[w.flush() for w in writers]
    for name,cursor,bank in zip(['dense','residual'],cursors,banks):
        audit=audit_alpha_log(tmp_path/name,cursor,cfg,bank,torch.arange(8),targets)
        assert audit['replay_exact'] and audit['examples']==8
    dense=[e for e in read_events(tmp_path/'dense',cursors[0]) if e['type']=='write']
    residual=[e for e in read_events(tmp_path/'residual',cursors[1]) if e['type']=='write']
    previous={}
    for a,b in zip(dense,residual):
        with np.load(tmp_path/'dense'/a['vectors']['file']) as z:full={k:z[k].copy() for k in z.files}
        with np.load(tmp_path/'residual'/b['vectors']['file']) as z:encoded={k:z[k].copy() for k in z.files}
        candidate,old,new=decode_vectors(b,encoded,previous.get(b['entry_id'],np.zeros(2,dtype=np.float32)))
        row=a['vectors']['row']
        assert np.array_equal(candidate,full['candidate'][row]) and np.array_equal(new,full['new'][row])
        if old is not None:assert np.array_equal(old,full['old'][row])
        assert not np.any(encoded['old_xor']) and not np.any(encoded['new_xor'])
        previous[b['entry_id']]=new.copy()


def reference_fixture(tmp_path,monkeypatch):
    parent=parent_fixture(tmp_path,monkeypatch)
    worker=admission.run_ablation(parent,output=tmp_path/'step1worker')
    reference=admission.aggregate_runs([worker],output=tmp_path/'step1')
    monkeypatch.setattr(study,'build_baseline_net',fixture_model)
    monkeypatch.setattr(study,'load_mnist_tensors',admission.load_mnist_tensors)
    return reference


def test_synthetic_alpha_study_reproduces_control_and_audits_all_three_banks(tmp_path,monkeypatch):
    reference=reference_fixture(tmp_path,monkeypatch)
    output=study.run_study(reference,output=tmp_path/'alpha')
    result=json.loads((output/'results.json').read_text())
    assert result['manifest']['protected_artifacts_unchanged']
    assert result['manifest']['shared_actual_cnn_training'] and not result['manifest']['backbone_feature_cache_used']
    assert not result['manifest']['test_set_opened']
    for name,audit in result['seeds'][0]['audits'].items():
        assert audit['replay_exact'] and audit['examples']==80
    assert result['seeds'][0]['exact_alpha_0p1_every_epoch']
    assert result['selection']['alpha'] in [.03,.1,.3]
    assert (output/'report.md').is_file() and (output/'validation_corruptions.png').is_file()
    combined=study.aggregate_studies([output],output=tmp_path/'combined')
    pooled=json.loads((combined/'results.json').read_text())
    assert pooled['selection']==result['selection']
    assert pooled['manifest']['selection_scope'].startswith('All reported seeds')
    with pytest.raises(ValueError,match='Duplicate'):
        study.aggregate_studies([output,output],output=tmp_path/'duplicate')
    assert not (tmp_path/'duplicate').exists()


def test_shared_alpha_checkpoint_resume_is_exact(tmp_path,monkeypatch):
    reference=reference_fixture(tmp_path,monkeypatch)
    full=study.run_study(reference,output=tmp_path/'full')
    resumed=study.run_study(reference,output=tmp_path/'resumed',stop_after_epoch=2)
    assert json.loads((resumed/'manifest.json').read_text())['status']=='incomplete'
    study.run_study(output=resumed,resume=True)
    a=torch.load(full/'seed_0/checkpoints/epoch_3.pt',weights_only=False)
    b=torch.load(resumed/'seed_0/checkpoints/epoch_3.pt',weights_only=False)
    for key in ['model','optimizer','banks','history','shuffle_rng','split_ids','optimizer_step']:
        admission.require_equal(a[key],b[key],key)
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])
    monkeypatch.setattr(study,'source_identity',lambda:{'commit':'changed','source_sha256':'changed'})
    with pytest.raises(ValueError,match='source'):study.run_study(output=resumed,resume=True)


def test_disk_reserve_failure_keeps_a_resumable_run(tmp_path,monkeypatch):
    from types import SimpleNamespace
    reference=reference_fixture(tmp_path,monkeypatch)
    output=tmp_path/'disk_limited'
    with monkeypatch.context() as patch:
        patch.setattr(study.shutil,'disk_usage',lambda path:SimpleNamespace(free=0))
        with pytest.raises(OSError,match='reserve'):study.run_study(reference,output=output)
    assert json.loads((output/'manifest.json').read_text())['status']=='failed'
    study.run_study(output=output,resume=True)
    assert json.loads((output/'manifest.json').read_text())['status']=='complete'


@pytest.mark.parametrize('values',[[],[.1],[.03,.03,.1],[0,.1],[True,.1],[float('nan'),.1],[float('inf'),.1],[1.1,.1],[.03,.3]])
def test_invalid_alpha_grids_fail(values):
    with pytest.raises(ValueError,match='Alphas'):study.validate_alphas(values)


def test_alpha_selection_prioritizes_clean_correct_counts_over_noise():
    def result(a,correct,auc):
        return {'method':study.alpha_name(a),'kind':'gaussian','level':0,'accuracy':correct/100,'both_correct':correct,'helped':0}
    seed={'validation_results':[result(.03,98,.1),result(.1,97,.99),result(.3,98,.5)],
          'robust_auc':{study.alpha_name(a)+':'+k:v for a,v in [(.03,.1),(.1,.99),(.3,.5)] for k in ['gaussian','salt_pepper','occlusion']}}
    assert study.select_alpha([seed],[.03,.1,.3])['alpha']==.3
    for row in seed['validation_results']:row['accuracy']=None
    assert study.select_alpha([seed],[.03,.1,.3])['alpha'] is None
