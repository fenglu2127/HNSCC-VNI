"""Generate only the 18 panels embedded in the submission manuscript."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse, csv, gzip, importlib, json, tempfile
import pandas as pd
import matplotlib.pyplot as plt
import common

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True,help='New output directory; must not exist')
    p.add_argument('--analysis-root',type=Path,help='Fresh upstream outputs; omit to use bundled reference intermediates')
    args=p.parse_args()
    out=args.output.resolve()
    if out==ROOT or ROOT in out.parents:
        raise ValueError('Write test outputs outside the submission package')
    out.mkdir(parents=True,exist_ok=False)
    common.FIG=str(out)
    with tempfile.TemporaryDirectory(prefix='hnscc_modules_') as tmp:
        for item in json.loads((ROOT/'docs/plot_inputs.json').read_text()):
            if args.analysis_root:
                matches=list(args.analysis_root.rglob(item['logical_name']))
                if len(matches)!=1:
                    raise ValueError(f"Expected one fresh input for {item['logical_name']}, found {len(matches)}")
                common._INDEX[item['logical_name']]=str(matches[0])
                continue
            files=[ROOT/f for f in item['files']]
            if len(files)==1:
                common._INDEX[item['logical_name']]=str(files[0])
            else:
                merged=Path(tmp)/item['logical_name']
                with merged.open('w') as dst:
                    for i,f in enumerate(files):
                        with gzip.open(f,'rt') as src:
                            if i:next(src)
                            for line in src:dst.write(line)
                common._INDEX[item['logical_name']]=str(merged)
        modules={i:importlib.import_module(f'figure_{i}') for i in range(1,9)}
        for m in modules.values():m.OUT=out;m.OUTPUT=out
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'figure.dpi':200,'savefig.dpi':200,'axes.spines.top':False,'axes.spines.right':False})
        if args.analysis_root:
            import anndata as ad
            atlas=ad.read_h5ad(args.analysis_root/'vascular_ecosystem_vn_scored.h5ad',backed='r')
            try:
                df=atlas.obs.copy();xy=atlas.obsm['X_umap'].copy()
            finally:atlas.file.close()
        else:
            df=pd.read_csv(ROOT/'outputs/source_data/Figure1_coordinates_scores.csv.gz',index_col=0)
            xy=df[['UMAP1','UMAP2']].to_numpy()
        modules[1].make_figure(df,xy)
        for fun in ['figure2A_mean_scores','figure2B_loadings','figure2C_correlation']:getattr(modules[2],fun)()
        modules[3].INPUT=common.D('DE_VNhigh_vs_VNlow_EC.csv')
        modules[3].main()
        modules[4].figure4_ec_subtype()
        modules[5].Figure5A()
        modules[5].Figure5B()
        modules[5].main()
        plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'figure.dpi':150,'savefig.dpi':300,'pdf.fonttype':42})
        modules[6].make_figures(pd.read_csv(common.D('five_axis_vni_survival_data.csv')),pd.read_csv(common.D('cox_results.csv')))
        plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':11,'figure.dpi':200,'savefig.dpi':300})
        for fun in ['make_figure_7a','make_figure_7b','make_figure_7c']:getattr(modules[7],fun)()
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'figure.dpi':200,'savefig.dpi':200})
        modules[8].Figure8A()
        modules[8].Figure8B()
        modules[8].main()
    manifest=list(csv.DictReader((ROOT/'docs/figure_manifest.csv').open()))
    expected={r['panel']+'.png' for r in manifest}
    assert {f.name for f in out.glob('*.png')}==expected
    if not args.analysis_root:
        for name in ['Figure3_plotted_genes.csv','Figure5C_statistics.csv','Figure8C_statistics.csv']:
            pd.testing.assert_frame_equal(pd.read_csv(ROOT/'outputs/source_data'/name),pd.read_csv(out/name),check_exact=False,rtol=1e-10,atol=1e-12)
    print('Generated exactly 18 PNG panels and their PDFs. Fresh-run comparisons are recorded separately.')

if __name__=='__main__':main()
