library(UCell)
library(GEOquery)
library(Seurat)
library(SeuratObject)
library(SingleCellExperiment)
set.seed(123)

#Load in data for all patients
gb2 <- readRDS('/Users/elikond/Downloads/integrated_npc50_k12_res6.rds')
gb9 <- readRDS('/Users/elikond/Downloads/gb9_sct_int_pc30_res_0.6_k_param_80.rds')
gb13 <- readRDS('/Users/elikond/Downloads/gb13_chi3l1_integrated.rds')
wcr8 <- readRDS('/Users/elikond/Downloads/w8_int_pc_50_kparam_40,res_0.5.rds')

#Get proneural and mesenchymal signatures
proneural <- read.csv('/Users/elikond/Downloads/Proneural.csv')
mesenchymal <- read.csv('/Users/elikond/Downloads/Mesenchymal.csv')
proneural <- as.data.frame(proneural)
mesenchymal <- as.data.frame(mesenchymal)

signatures <- list(Mesenchymal = c(mesenchymal[,1]), Proneural = c(proneural[,1]))

mesen_pron <- function(sig, patient_data){
  seurat.object <- AddModuleScore_UCell(patient_data, features = signatures, name = NULL,
                                        ncores = 4)
  normalized.seurat.object <- NormalizeData(seurat.object)
  variable.seurat.object <- FindVariableFeatures(normalized.seurat.object, selection.method = "vst", nfeatures = 500)
  scaled.seurat.object <- ScaleData(variable.seurat.object)
  pca.seurat.object <- RunPCA(scaled.seurat.object, features = scaled.seurat.object@assays$RNA@var.features,
                          npcs = 20)
  umap.seurat.object <- RunUMAP(pca.seurat.object, reduction = "pca", dims = 1:20, seed.use = 123)
  FeaturePlot(umap.seurat.object, reduction = "umap", features = names(signatures), ncol = 3,
              order = T)
  
}

mesen_pron(signatures, gb9)