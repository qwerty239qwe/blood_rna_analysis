library(rrcov)


rm(list = ls())

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

countTable <- read.table("../data/mg_counts.tsv", sep='\t', header = TRUE)
countsData <- as.matrix(countTable[,2:dim(countTable)[2]])
cpm <- log2(countsData + 1)

tpmTable <- read.delim("../data/mg_t_TPMs.tsv", sep='\t', header = TRUE)
tpmData <- as.matrix(tpmTable[,2:dim(tpmTable)[2]])
cpm <- log2(tpmData + 1)

pca2 <- PcaHubert(t(cpm), k=6)  
plot(pca2)

pc <- PcaGrid(t(cpm), 6)
plot(pc)