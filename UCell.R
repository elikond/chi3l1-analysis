library(UCell)
library(GEOquery)
library(Seurat)
library(SeuratObject)
library(SingleCellExperiment)
set.seed(123)

gb9 <- readRDS('/Users/elikond/Downloads/gb9_sct_int_pc30_res_0.6_k_param_80.rds')
proneural <- read.csv('/Users/elikond/Downloads/Proneural.csv')
mesenchymal <- read.csv('/Users/elikond/Downloads/Mesenchymal.csv')
proneural <- as.data.frame(proneural)
mesenchymal <- as.data.frame(mesenchymal)

signatures <- list(Mesenchymal = c(mesenchymal[,1]), Proneural = c(proneural[,1]))

#seurat.object <- CreateSeuratObject(counts = gb9, project = "test1")
my.seurat.object <- AddModuleScore_UCell(gb9, features = signatures, name = NULL,
                                      ncores = 4)
head(my.seurat.object@meta.data)

normalized.seurat.object <- NormalizeData(my.seurat.object)
seurat.object <- FindVariableFeatures(normalized.seurat.object, selection.method = "vst", nfeatures = 500)

seurat.object <- ScaleData(seurat.object)
seurat.object <- RunPCA(seurat.object, features = seurat.object@assays$RNA@var.features,
                        npcs = 20)
seurat.object <- RunUMAP(seurat.object, reduction = "pca", dims = 1:20, seed.use = 123)

FeaturePlot(seurat.object, reduction = "umap", features = names(signatures), ncol = 3,
            order = T)

cell_df <- data.frame(Cells(seurat.object))
Idents(object = seurat.object)
levels(x = seurat.object)
subset(x = seurat.object, idents = 0)

mesenchymal_proneural <- seurat.object@meta.data[,9:10]
temp_col <- sub('GEX:', '', rownames(mesenchymal_proneural))
final_names <- sub('x', '.1', temp_col)
row.names(mesenchymal_proneural) <- final_names
write.csv(mesenchymal_proneural, '~/Downloads/mesenchymal_proneural.csv')