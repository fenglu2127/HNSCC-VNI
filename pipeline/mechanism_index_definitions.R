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

