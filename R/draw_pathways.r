library(glue)
library(org.Mm.eg.db)
library(clusterProfiler)
library(pathview)
library(dplyr)
library(magrittr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source(file.path("get_kegg_id.r"))

COMPARE <- c("84Q_15Q")  # , "84Q+IGF1_15Q", "84Q+IGF1_84Q"
OUTDIR <- file.path("../../plots/enrichments/KEGG/pathways")
intKEGG <- read.csv("../../data/kegg_path.csv")$kegg.id
file_dir <- glue("../../tables/S1_DEGs")

drawPath<- function(file_path, output_dir, prefix, mapIDs, cp = COMPARE){
  for (comp in cp){
    test <- read.table(glue("{file_path}/DE_{comp}.tsv"), sep='\t', header = T)
    row.names(test) <- substr(row.names(test), 1, 18)
    
    NewID <- bitr(row.names(test), 
                  fromType="ENSEMBL", 
                  toType=c("ENSEMBL", "ENTREZID"), 
                  OrgDb="org.Mm.eg.db", 
                  drop=F)
    addedID <- merge(test, NewID, by.x = 0, by.y = "ENSEMBL")
    names(addedID)[names(addedID) == "log2FoldChange"] <- "logFC"
    
    dataFC <- as.data.frame(addedID$logFC)
    colnames(dataFC) <- c("logFC")
    dataFC$ENTREZID <- addedID$ENTREZID
    
    dataFC <- dataFC %>% group_by(ENTREZID) %>% summarise(logFC = sum(logFC))
    dataFC <- as.data.frame(dataFC)
    dataFC <- dataFC[!is.na(dataFC$ENTREZID),]
    row.names(dataFC) <- dataFC$ENTREZID
    for (mapID in mapIDs){
      queryData <- data.frame(dataFC)
      queryData <- queryData[row.names(queryData) %in% p2e[[mapID]], ] # out table
      
      print(queryData[, 2, drop=FALSE])
      if (!file.exists(glue("{output_dir}/{mapID}"))){
        dir.create(file.path(output_dir, mapID))
      }
      
      lim <- 1.5
      write.table(queryData, file = file.path(output_dir, glue("{mapID}/data_{comp}.tsv")), sep='\t')
      pv.out <- pathview(gene.data = queryData[, 2, drop=FALSE], pathway.id = substr(mapID, 4, 8),
                         species = "mmu", kegg.dir = file.path(output_dir, mapID), 
                         same.layer = F, node.sum = "mean",
                         out.suffix = glue("kegg_{comp}"), kegg.native = T, limit = c(-lim, lim))
    }
  }
}

drawPath(file_dir, OUTDIR, "DESeq", intKEGG)
