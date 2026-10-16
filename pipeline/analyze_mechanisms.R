#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(data.table)
  library(dplyr)
  library(ggplot2)
})

# --- package-relative paths (this script lives in <pkg>/pipeline/) ---
initial.options <- commandArgs(trailingOnly = FALSE)
script.path <- sub("--file=", "", initial.options[grep("--file=", initial.options)])
pkg <- dirname(dirname(normalizePath(script.path)))
args <- commandArgs(trailingOnly = TRUE)
rds_path <- ifelse(length(args) >= 1, args[[1]], file.path(pkg, "raw_data", "HPVnegHNSCCAtlas.rds"))
out_dir  <- ifelse(length(args) >= 2, args[[2]], file.path(pkg, "outputs", "jain_review"))
fig_dir  <- file.path(out_dir, "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

message("Reading atlas: ", rds_path)
obj <- readRDS(rds_path)
DefaultAssay(obj) <- if ("RNA" %in% names(obj@assays)) "RNA" else DefaultAssay(obj)
meta <- obj@meta.data
message("Loaded object size GB: ", round(as.numeric(object.size(obj)) / 1024^3, 2))

guess_col <- function(patterns) {
  hits <- colnames(meta)[Reduce(`|`, lapply(patterns, function(p) grepl(p, colnames(meta), ignore.case = TRUE)))]
  if (length(hits) == 0) return(NA_character_)
  hits[[1]]
}
celltype_col <- guess_col(c("celltype","cell.type","annotation","label","ident","compartment","classification"))
sample_col   <- guess_col(c("patient","sample","donor","orig.ident"))
dataset_col  <- guess_col(c("dataset","cohort","study","source"))
write.csv(data.frame(column = colnames(meta)), file.path(out_dir, "metadata_columns.csv"), row.names = FALSE)
message("celltype_col=", celltype_col, "; sample_col=", sample_col, "; dataset_col=", dataset_col)
if (!is.na(celltype_col)) message("Cell type values: ", paste(head(unique(meta[[celltype_col]]), 20), collapse=", "))

score_sets <- list(
  EC_maturation_phalanx        = c("PECAM1","VWF","CDH5","CLDN5","ESAM","KLF2","KLF4","TEK","ENG"),
  EC_tip_sprouting             = c("KDR","DLL4","ESM1","ANGPT2","APLN","CXCR4","PDGFB"),
  EC_stalk_Notch               = c("NOTCH1","HEY1","HEY2","JAG1","NRP1","FLT1"),
  pathological_TEC_embryonic   = c("FBN2","EMILIN2","LOX","SERPINE1","COL15A1","IGFBP7"),
  abnormal_angiogenesis        = c("VEGFA","ANGPT2","FGF2","PDGFB","APLN","ADM","MMP14"),
  hypoxia_acidosis             = c("HIF1A","CA9","SLC2A1","LDHA","ALDOA","ENO1","ADM"),
  EC_antigen_presentation      = c("HLA-A","HLA-B","HLA-C","B2M","TAP1","TAP2","NLRC5"),
  EC_leukocyte_adhesion        = c("ICAM1","ICAM2","VCAM1","SELE","SELPLG","CD99","JAM2","JAM3"),
  EC_death_ligand_barrier      = c("FASLG","PDCD1LG2","CD274","LGALS9","TNFSF10"),
  CXCL9_10_Tcell_recruitment   = c("CXCL9","CXCL10","CXCL11","CXCR3","IFNG","STAT1","IRF1"),
  Tcell_cytotoxic_effector     = c("CD8A","CD8B","GZMB","PRF1","NKG7","GNLY","IFNG"),
  Tcell_dysfunction_Treg       = c("PDCD1","HAVCR2","LAG3","TIGIT","TOX","CXCL13","FOXP3","IL2RA","CTLA4"),
  pericyte_coverage            = c("RGS5","PDGFRB","CSPG4","MCAM","ACTA2","MYH11","TAGLN","NOTCH3","DES"),
  ANG1_TIE2_stabilization      = c("ANGPT1","TEK","TIE1","FOXO1","KLF2","KLF4"),
  ANG2_destabilization         = c("ANGPT2","TEK","VEGFA","KDR","MMP14"),
  PDGFB_PDGFRB_axis            = c("PDGFB","PDGFRB","RGS5","CSPG4","MCAM"),
  basement_membrane_integrity  = c("COL4A1","COL4A2","HSPG2","LAMA4","LAMB1","NID1","NID2","TIMP1"),
  ECM_remodeling_MMP_LOX       = c("MMP2","MMP9","MMP14","LOX","LOXL2","TIMP1","TIMP2","FN1"),
  CAF_mechanical_compression   = c("FAP","ACTA2","TAGLN","MYL9","COL1A1","COL1A2","COL3A1","FN1","POSTN","LOX","LOXL2"),
  YAP_TAZ_mechanotransduction  = c("YAP1","WWTR1","CTGF","CYR61","ANKRD1","AMOTL2"),
  glycolysis_lactate           = c("SLC2A1","HK2","PFKP","ALDOA","ENO1","PKM","LDHA","SLC16A1","SLC16A3"),
  adenosine_immunosuppression  = c("ENTPD1","NT5E","ADORA2A","ADORA2B","CD38"),
  inflammatory_NFkB            = c("NFKB1","RELA","IL6","CXCL8","CCL2","ICAM1","VCAM1"),
  SPP1_TAM_remodeling          = c("SPP1","APOE","TREM2","MMP9","CTSB","CSTB","LGALS3"),
  IL1B_inflammatory_myeloid    = c("IL1B","CXCL8","S100A8","S100A9","NLRP3","LYZ","LST1"),
  mregDC_antigen_presentation  = c("CCR7","LAMP3","FSCN1","CCL19","CCL22","CD80","CD83","HLA-DRA","HLA-DPA1")
)

present_genes <- function(gs) intersect(gs, rownames(obj))

message("Scoring ", length(score_sets), " review-derived modules...")
for (nm in names(score_sets)) {
  genes <- present_genes(score_sets[[nm]])
  if (length(genes) < 2) {
    message("  Skip ", nm, ": only ", length(genes), " genes present")
    next
  }
  message("  Scoring: ", nm, " (", length(genes), "/", length(score_sets[[nm]]), " genes)")
  obj <- AddModuleScore(obj, features = list(genes), name = paste0(nm, "_"))
  new_col <- tail(colnames(obj@meta.data), 1)
  colnames(obj@meta.data)[colnames(obj@meta.data) == new_col] <- paste0(nm, "_score")
}
message("Module scoring complete.")

z    <- function(x) as.numeric(scale(x))
safe <- function(col) if (col %in% colnames(obj@meta.data)) obj@meta.data[[col]] else rep(NA_real_, nrow(obj@meta.data))
mean_index <- function(cols, signs = rep(1, length(cols))) {
  mat <- do.call(cbind, Map(function(col, s) s * z(safe(col)), cols, signs))
  rowMeans(mat, na.rm = TRUE)
}

obj$review_normalization_index <- mean_index(
  c("EC_maturation_phalanx_score","EC_antigen_presentation_score","EC_leukocyte_adhesion_score",
    "pericyte_coverage_score","basement_membrane_integrity_score",
    "hypoxia_acidosis_score","abnormal_angiogenesis_score","ANG2_destabilization_score"),
  c(1, 1, 1, 1, 1, -1, -1, -1))

obj$endothelial_immune_barrier_index <- mean_index(
  c("EC_death_ligand_barrier_score","Tcell_dysfunction_Treg_score","hypoxia_acidosis_score",
    "adenosine_immunosuppression_score","EC_antigen_presentation_score","EC_leukocyte_adhesion_score"),
  c(1, 1, 1, 1, -1, -1))

obj$immune_gate_reopening_index <- mean_index(
  c("EC_antigen_presentation_score","EC_leukocyte_adhesion_score",
    "CXCL9_10_Tcell_recruitment_score","Tcell_cytotoxic_effector_score",
    "Tcell_dysfunction_Treg_score"),
  c(1, 1, 1, 1, -1))

obj$stromal_compression_index <- mean_index(
  c("CAF_mechanical_compression_score","ECM_remodeling_MMP_LOX_score",
    "YAP_TAZ_mechanotransduction_score","pericyte_coverage_score"),
  c(1, 1, 1, -1))

obj$metabolic_immunosuppression_index <- mean_index(
  c("hypoxia_acidosis_score","glycolysis_lactate_score",
    "adenosine_immunosuppression_score","Tcell_cytotoxic_effector_score"),
  c(1, 1, 1, -1))

message("Composite indices computed.")

score_cols <- grep("_score$|_index$", colnames(obj@meta.data), value = TRUE)
fwrite(as.data.table(obj@meta.data[, score_cols, drop = FALSE], keep.rownames = "cell"),
       file.path(out_dir, "cell_review_module_scores.csv"))
message("Saved: cell_review_module_scores.csv")

summarise_by <- function(col, filename) {
  if (is.na(col) || !(col %in% colnames(obj@meta.data))) return(NULL)
  tab <- obj@meta.data %>%
    mutate(.group = .data[[col]]) %>%
    group_by(.group) %>%
    summarise(n = n(), across(all_of(score_cols), ~mean(.x, na.rm = TRUE)), .groups = "drop")
  write.csv(tab, file.path(out_dir, filename), row.names = FALSE)
  tab
}
celltype_summary <- summarise_by(celltype_col, "scores_by_celltype.csv")
sample_summary   <- summarise_by(sample_col,   "scores_by_sample.csv")
dataset_summary  <- summarise_by(dataset_col,  "scores_by_dataset.csv")

axis_cols <- intersect(c(
  "review_normalization_index","endothelial_immune_barrier_index","immune_gate_reopening_index",
  "stromal_compression_index","metabolic_immunosuppression_index",
  "EC_tip_sprouting_score","EC_maturation_phalanx_score","pathological_TEC_embryonic_score",
  "CXCL9_10_Tcell_recruitment_score","Tcell_cytotoxic_effector_score","Tcell_dysfunction_Treg_score",
  "SPP1_TAM_remodeling_score","IL1B_inflammatory_myeloid_score",
  "mregDC_antigen_presentation_score","CAF_mechanical_compression_score"
), colnames(obj@meta.data))

cor_mat <- cor(obj@meta.data[, axis_cols, drop = FALSE], method = "spearman", use = "pairwise.complete.obs")
write.csv(cor_mat, file.path(out_dir, "review_axis_spearman_correlation.csv"))
message("Saved: review_axis_spearman_correlation.csv")

# Ecosystem state clustering (sample level)
if (!is.null(sample_summary)) {
  sample_axes <- intersect(axis_cols, colnames(sample_summary))
  if (length(sample_axes) >= 4 && nrow(sample_summary) >= 4) {
    mat  <- as.matrix(sample_summary[, sample_axes, drop = FALSE])
    rownames(mat) <- sample_summary$.group
    keep <- rowSums(!is.finite(mat)) == 0
    k    <- min(4, sum(keep))
    if (k >= 2) {
      set.seed(1234)
      km <- kmeans(scale(mat[keep, , drop = FALSE]), centers = k, nstart = 50)
      clusters <- data.frame(sample = rownames(mat)[keep],
                             review_ecosystem_state = paste0("JainEco", km$cluster))
      write.csv(clusters, file.path(out_dir, "sample_review_ecosystem_states.csv"), row.names = FALSE)
      message("Saved: sample_review_ecosystem_states.csv  (", k, " clusters)")
    }
  }
}

# Correlation heatmap
pdf(file.path(fig_dir, "review_axis_correlation_heatmap.pdf"), width = 9, height = 8)
par(mar = c(10, 10, 3, 1))
image(t(cor_mat[nrow(cor_mat):1, ]), axes = FALSE,
      col = colorRampPalette(c("#e94560","white","#0f3460"))(101), zlim = c(-1, 1))
axis(1, at = seq(0, 1, length.out = ncol(cor_mat)), labels = colnames(cor_mat), las = 2, cex.axis = 0.62)
axis(2, at = seq(0, 1, length.out = nrow(cor_mat)), labels = rev(rownames(cor_mat)), las = 2, cex.axis = 0.62)
title("Cell 2026 vascular-normalization review axes · Spearman correlation")
dev.off()
message("Saved: review_axis_correlation_heatmap.pdf")

# Scatter: normalization vs immune gate (sample level)
if (!is.null(sample_summary) &&
    all(c("review_normalization_index","immune_gate_reopening_index",
          "endothelial_immune_barrier_index") %in% colnames(sample_summary))) {
  p <- ggplot(sample_summary,
              aes(x = review_normalization_index, y = immune_gate_reopening_index,
                  color = endothelial_immune_barrier_index, size = n)) +
    geom_point(alpha = 0.85) +
    scale_color_gradient2(low = "#0f3460", mid = "grey90", high = "#e94560") +
    theme_classic(base_size = 12) +
    labs(x = "Review-derived vascular normalization index",
         y = "Immune gate reopening index",
         color = "Endothelial\nimmune barrier", size = "Cells")
  ggsave(file.path(fig_dir, "sample_normalization_vs_immune_gate.pdf"), p, width = 6.4, height = 4.8)
  message("Saved: sample_normalization_vs_immune_gate.pdf")
}

message("\nAll done.")
message("Outputs in: ", out_dir)
message("Figures in: ", fig_dir)
