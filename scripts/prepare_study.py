"""Declarative genuine-SAFE crop preparation with frozen train-only normalization."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from prepare_sentinel2_scenes import find_scene_assets
from pcap.preprocessing import load_sentinel2,fit_standardization,Standardization


def prepare(config_path,output_dir,*,stats_path=None):
    config_path,output_dir=Path(config_path),Path(output_dir)
    config=json.loads(config_path.read_text())
    crops=config.get('crops',[]);products=config.get('products',{})
    if not crops or not products:raise ValueError('products and crops are required')
    ids=[c['id'] for c in crops]
    if len(set(ids))!=len(ids) or any(not x or Path(x).name!=x or x in ('.','..') for x in ids):raise ValueError('Unique safe crop IDs required')
    if any(c['split'] not in ('train','validation','test','spatial_test') for c in crops):raise ValueError('Unknown dataset split')
    assets={}
    for key,entry in products.items():
        directory=Path(entry['path']).expanduser()
        paths,scl,xml=find_scene_assets(directory)
        from pcap.preprocessing import read_product_metadata
        product=read_product_metadata(xml).product_id
        if product!=entry['product_id']:raise ValueError('Expected product ID mismatch')
        assets[key]=(paths,scl,xml,product)
    # Independent validation/test must not share a product with any other split.
    product_splits={}
    for crop in crops:
        if crop['product'] not in assets:raise ValueError('Unknown crop product')
        product=assets[crop['product']][3]
        product_splits.setdefault(product,set()).add(crop['split'])
    if any(len(splits)>1 and bool(splits&{'validation','test'}) for splits in product_splits.values()):
        raise ValueError('Validation/test require independent product identities; use spatial_test for same-product diagnostics')
    frozen=Standardization.from_json(Path(stats_path).read_text()) if stats_path else None
    fit_ids=config.get('fit_statistics_on',[c['id'] for c in crops if c['split']=='train'])
    if not frozen and (not fit_ids or set(fit_ids)-set(ids)):raise ValueError('Training crop IDs required')
    if any(c['split']!='train' for c in crops if c['id'] in fit_ids):raise ValueError('Never fit statistics on held-out crops')
    if output_dir.exists():raise FileExistsError('Choose a new output directory')
    loaded={}
    for crop in crops:
        paths,scl,xml,product=assets[crop['product']]
        window=crop['window']
        limit=config.get('max_crop_pixels',2048*2048)
        scene=load_sentinel2(paths,scl,xml,window=window,max_crop_pixels=limit,
            sample_size=crop.get('sample_size'),seed=crop.get('seed',0),exclude_snow=config.get('exclude_snow',False))
        loaded[crop['id']]=scene
    if frozen is None:
        frozen=fit_standardization([loaded[i].X for i in fit_ids],training_ids=fit_ids,split='train')
    output_dir.mkdir(parents=True,exist_ok=False)
    stats_text=frozen.to_json();(output_dir/'standardization_stats.json').write_text(stats_text)
    for crop in crops:
        scene=loaded[crop['id']].standardized(frozen)
        scene.metadata.update(scene_id=assets[crop['product']][3],crop_id=crop['id'],split=crop['split'],
            standardization_source='frozen_file' if stats_path else 'declared_training_crops',
            standardization_sha256=hashlib.sha256(stats_text.encode()).hexdigest(),
            valid_mask_sha256=hashlib.sha256(scene.valid_mask.tobytes()).hexdigest(),
            study_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
        folder=output_dir/crop['id'];folder.mkdir()
        for file,array in [('X',scene.X),('valid_mask',scene.valid_mask),('pixel_indices',scene.pixel_indices)]:np.save(folder/(file+'.npy'),array)
        (folder/'metadata.json').write_text(json.dumps(scene.metadata,indent=2))
    (output_dir/'study.json').write_text(json.dumps(dict(config=config,frozen_statistics=bool(stats_path),statistics_training_ids=frozen.training_ids),indent=2))
    artifacts=sorted(p for p in output_dir.rglob('*') if p.is_file())
    (output_dir/'checksums.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(output_dir)}\n' for p in artifacts))
    print('Prepared',len(crops),'real SAFE crops:',output_dir)
    return output_dir


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--stats-json',type=Path,help='Apply existing training statistics without refitting')
    a=p.parse_args();prepare(a.config,a.output_dir,stats_path=a.stats_json)
