"""Recompute manuscript analyses from atlas objects and TCGA expression inputs."""
from pathlib import Path
import argparse, csv, hashlib, json, os, shutil, subprocess, sys, time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True,help='Directory containing HNSCC_integrated_atlas.h5ad, HPVnegHNSCCAtlas.rds and tcga/')
    p.add_argument('--work-dir',type=Path,required=True,help='New directory outside this package')
    p.add_argument('--rscript',default='Rscript')
    a=p.parse_args();work=a.work_dir.resolve();raw=a.input_dir.resolve()
    if work==ROOT or ROOT in work.parents:raise ValueError('Work directory must be outside this package')
    for row in csv.DictReader((ROOT/'raw_data/input_manifest.csv').open()):
        f=raw/row['expected_path'].removeprefix('raw_data/')
        if not f.is_file():raise FileNotFoundError(f)
        if sha(f)!=row['sha256']:raise ValueError(f'Input checksum mismatch: {f.name}')
    work.mkdir(parents=True,exist_ok=False)
    (work/'logs').mkdir();(work/'outputs').mkdir();(work/'pipeline').mkdir()
    (work/'raw_data').symlink_to(raw,target_is_directory=True)
    env=os.environ.copy()
    for k in ['SCHNSCC_H5AD','SCHNSCC_OUT']:env.pop(k,None)
    env.update({'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','VECLIB_MAXIMUM_THREADS':'4','NUMBA_NUM_THREADS':'4'})
    env.setdefault('MPLCONFIGDIR',str(work/'runtime_cache/matplotlib'))
    env.setdefault('NUMBA_CACHE_DIR',str(work/'runtime_cache/numba'))
    names=['analyze_vascular.py','analyze_mechanisms.R','analyze_immune.py','analyze_neighbors.py','analyze_tcga.py']
    status={'mode':'full upstream recomputation','stages':{}}
    def save():(work/'run_status.json').write_text(json.dumps(status,indent=2))
    for name in names:
        shutil.copy2(ROOT/'pipeline'/name,work/'pipeline'/name)
        cmd=[a.rscript if name.endswith('.R') else sys.executable,str(work/'pipeline'/name)]
        if name.endswith('.R'):cmd += [str(raw/'HPVnegHNSCCAtlas.rds'),str(work/'outputs/jain_review')]
        status['stages'][name]={'started':time.time(),'state':'running'};save()
        with (work/'logs'/f'{name}.log').open('w') as log:
            result=subprocess.run(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
        status['stages'][name].update({'finished':time.time(),'exit_code':result.returncode,'state':'completed' if result.returncode==0 else 'failed'});save()
        if result.returncode:raise SystemExit(f'{name} failed. Inspect its log.')
    subprocess.run([sys.executable,str(ROOT/'pipeline/run_figures.py'),'--analysis-root',str(work/'outputs'),'--output',str(work/'manuscript_figures')],env=env,check=True)
if __name__=='__main__':main()
