# HNSCC vascular normalization index

Analysis code, figure source data and marker definitions for a study of the vascular ecosystem in head and neck squamous cell carcinoma.

## Contents

- `pipeline/`: upstream analysis and figure-generation scripts.
- `outputs/figures/`: Figures 1, 2A–C, 3, 4, 5A–C, 6A–C, 7A–C and 8A–C.
- `outputs/source_data/`: panel-level values, coordinates and statistical results.
- `outputs/intermediates/`: processed inputs for figure generation. The mechanism-module matrix is stored in six row-wise gzip CSV parts and joined by the plotting runner.
- `supplementary/`: module markers, gene membership and index definitions in CSV and Excel formats.
- `docs/figure_manifest.csv`: mapping between panels, scripts and source data.
- `docs/environment_and_inputs.json`: computational environment and input checksums.
- `raw_data/input_manifest.csv`: upstream input identities, checksums and provenance.

## Generate figures

Use Python 3.12 and install the dependencies:

```bash
python -m pip install -r requirements-python.txt
bash run_all.sh --output /path/to/figures
```

The output directory must not already exist and must be outside this repository. This route uses bundled processed inputs and produces 18 PNG panels, matching PDFs and panel statistics. Image rendering can vary with fonts and software versions.

## Run upstream analyses

Install the Python dependencies and the R packages listed in `requirements-R.txt`. Provide the five files listed in `raw_data/input_manifest.csv`: two atlas objects and three files in a `tcga/` subdirectory.

```bash
bash run_complete.sh --input-dir /path/to/input_data --work-dir /path/to/analysis
```

The work directory must not already exist and must be outside this repository. The runner verifies input SHA-256 values, calculates vascular scores, mechanism modules, immune subtypes, embedding-neighbor associations and TCGA survival models, then generates figures. Logs and diagnostic outputs are saved in the work directory. This route recalculates scores rather than loading bundled score caches.

## Methods and marker definitions

The five-axis VNI is the equal-weight mean of direction-aligned standardized VM, adjusted PC, inverted HDA, IVI and adjusted EBI scores. PC and EBI include subtraction before standardization. Table S1 lists 7 VNI constituent sets, 26 mechanism modules and 7 immune-function sets (40 sets and 323 gene memberships). Table S2 defines 6 composite indices. Marker lists specify intended gene sets; detected genes depend on the input dataset.

Figure 2A shows unadjusted module means. Mechanism indices in Figure 7 are distinct from the five-axis VNI. UMAP neighbors represent similarity in an embedding, not physical tissue proximity. Cell-level tests do not establish patient-level effects, causality or clinical utility.

## Data availability

Figure source data and processed plotting inputs are included. Complete upstream analyses require approximately 16 GiB of additional processed data; these files are not included. The Seurat atlas source is https://doi.org/10.5281/zenodo.17880437. The AnnData object is a derived atlas export; its conversion script is unavailable. Exact TCGA input checksums are documented, but the original download manifest is unavailable. A persistent download location for the exact AnnData and TCGA inputs is not currently specified, which limits independent upstream reproduction.

The workflow starts from integrated atlas objects and TCGA FPKM, not sequencing reads. Data-use permissions are governed by the original providers. No software licence is specified in this repository.
