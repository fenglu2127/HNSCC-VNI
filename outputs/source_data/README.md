# Figure source data

Files are mapped in `docs/figure_manifest.csv`. CSVs use UTF-8, headers and empty fields for missing values; `.csv.gz` is gzip-compressed CSV. Gene symbols and cell/patient identifiers are retained as join keys. Scores are dimensionless computational quantities, not direct measurements of perfusion, vessel coverage or tissue distance.

| Files | Interpretation |
|---|---|
| Figure1_coordinates_scores | Cell ID, ecosystem UMAP1/2 and raw/adjusted score columns used in Figure 1; 75,394 rows |
| Figure2A_module_means | Mean unadjusted module scores by endothelial VNI group; not the adjusted contributions |
| Figure2B_weights | Signed/adjusted z-score component labels and their equal weights |
| Figure2C_correlations | Pearson correlations of adjusted PC/EBI, raw VM/HDA/IVI and VNI across the ecosystem |
| Figure3_plotted_genes | Retained gene list, log fold change, Wilcoxon statistics/adjusted P values and display-only functional category |
| Figure4_subtype_percent | Endothelial subtype percentages within each VNI group |
| Figure5A_cell_neighbor_fractions | EC IDs, VNI state and T-cell fractions among embedding neighbors; any-neighbor indicator for Fisher's test |
| Figure5B_* | Immune-subtype percentages in neighbor-VNI quartiles |
| Figure5C_statistics | Retained immune module comparison sample sizes, means and Mann–Whitney P values |
| Figure6_survival_data | Patient-level scores; OS_time_months in months, OS_event 1=death/0=censored; blank clinical values are missing |
| Figure6_cox_results | Cox estimates, hazard ratios and confidence intervals; model/covariate labels retained |
| Figure7A_correlations | Spearman correlations of five mechanism indices across the complete atlas |
| Figure7B_celltype_indices | Mean mechanism indices by RDS annotation, with cell counts |
| Figure7C_module_means | Module means in VNI endothelial IDs versus all atlas cells; AddModuleScore units |
| Figure8_statistics | Endothelial module tests and T-cell-context tests; includes hypoxia results, which are not displayed in panel 8C |
| Figure8C_statistics | Modules displayed in panel 8C |

The corresponding processed cell-level plotting inputs are bundled under `outputs/intermediates/`. Figure-level summaries alone do not replace upstream expression data or permit recalculation of every test. Statistical tests treat cells as observations where specified by the existing scripts; cell-level inference does not account for donor-level dependence. Mechanism scores are hypothesis-generating, not validated clinical signatures.
