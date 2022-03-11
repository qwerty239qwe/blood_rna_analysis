library(rrcov)


rm(list = ls())

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

countTable <- read.table("../data/mg_counts_2203.tsv", sep='\t', header = TRUE)
countsData <- as.matrix(countTable[,2:dim(countTable)[2]])
cpm <- log2(countsData + 1)

pca2 <- PcaHubert(t(cpm), k=6)  
plot(pca2)

pc <- PcaGrid(t(cpm), 6)
plot(pc)

rleTable <- read.table("../data/mg_RLE.tsv", sep='\t', header = TRUE)
rleData <- as.matrix(rleTable[,2:dim(rleTable)[2]])
cpm <- log2(rleData + 1)

pca2 <- PcaHubert(t(cpm), k=6)  
plot(pca2)

pc <- PcaGrid(t(cpm), 6)
plot(pc)

tpmTable <- read.delim("../data/mg_tpm_2203.tsv", sep='\t', header = TRUE)
tpmData <- as.matrix(tpmTable[,2:dim(tpmTable)[2]])
cpm <- log2(tpmData + 1)

pca2 <- PcaHubert(t(cpm), k=6)  
plot(pca2)

pc <- PcaGrid(t(cpm), 6)
plot(pc)