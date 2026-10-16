# Upstream inputs

`input_manifest.csv` lists the five input files, sizes, checksums and provenance. Place the two atlas objects in the input directory and the expression, clinical and identifier-map files in its `tcga/` subdirectory. Pass that directory to `run_complete.sh --input-dir`.

Inputs are processed atlas objects and TCGA FPKM/clinical tables, not raw sequencing reads. They are not bundled here. The Seurat atlas source is https://doi.org/10.5281/zenodo.17880437; RDS MD5: `f56f59390de9e59624b3633ffb00033d`.

The AnnData conversion script and exact TCGA download manifest are unavailable. A persistent download location for these exact inputs is not specified. Bundled data under `outputs/` support figure generation but do not replace upstream expression inputs.
