library(MASS)
library(class)
library(cluster)
library(impute)
library(Hmisc)
library(WGCNA)
library(DESeq2)
library(stringr)

quantile_normalisation <- function(df){
  df_rank <- apply(df,2,rank,ties.method="min")
  df_sorted <- data.frame(apply(df, 2, sort))
  df_mean <- apply(df_sorted, 1, mean)
  
  index_to_mean <- function(my_index, my_mean){
    return(my_mean[my_index])
  }
  
  df_final <- apply(df_rank, 2, index_to_mean, my_mean=df_mean)
  rownames(df_final) <- rownames(df)
  return(df_final)
}

options(stringsAsFactors = FALSE)
enableWGCNAThreads()

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

outliers <- c("R031", "R040", "R046", "R050", "R054")
type <- "signed"

sampleInfo <- read.table("../data/all_info/labels.csv", sep='\t', header = TRUE)
sampleInfo <- sampleInfo[-match(outliers, sampleInfo$sample),]
countTable <- read.table("../data/mg_counts.tsv", sep='\t', header = TRUE)
countsData <- as.matrix(countTable[,2:dim(countTable)[2]])

colnames(countsData) <- str_match(colnames(countsData), "(.*)A")[,2]
rownames(countsData) <- countTable$gene

countsData <- countsData[, -match(outliers, colnames(countsData))]

sampleInfo$group <- factor(sampleInfo$group)
sampleInfo$batch <- factor(sampleInfo$batch)
dds <- DESeqDataSetFromMatrix(countsData[, sampleInfo$sample], 
                              colData = sampleInfo, 
                              design = ~batch + group)
vsd <- vst(dds)
vsdData <- log2(vsd@assays@data@listData[[1]] + 1)

vsdData <- t(quantile_normalisation(vsdData))

gsg = goodSamplesGenes(vsdData, verbose = 3)
gsg$allOK

if (!gsg$allOK){
  # Optionally, print the gene and sample names that were removed:
  if (sum(!gsg$goodGenes)>0) 
    printFlush(paste("Removing genes:", 
                     paste(names(vsdData)[!gsg$goodGenes], collapse = ",")));
  if (sum(!gsg$goodSamples)>0) 
    printFlush(paste("Removing samples:", 
                     paste(rownames(vsdData)[!gsg$goodSamples], collapse = ",")));
  # Remove the offending genes and samples from the data:
  vsdData = vsdData[gsg$goodSamples, gsg$goodGenes]
}

sampleTree = hclust(dist(vsdData), method = "average")
plot(sampleTree, main = "Sample clustering to detect outliers", sub="", xlab="")

powers1 <- c(seq(1, 10, by=1), seq(12, 20, by=2))
sft <- pickSoftThreshold(vsdData, powerVector = powers1, networkType = type)
RpowerTable <- sft[[2]]
 
cex1 = 0.7
par(mfrow = c(1,2))
plot(RpowerTable[,1], -sign(RpowerTable[,3])*RpowerTable[,2], xlab = "soft threshold (power)", ylab = "scale free topology model fit, signes R^2", type = "n")
text(RpowerTable[,1], -sign(RpowerTable[,3])*RpowerTable[,2], labels = powers1, cex = cex1, col = "red")
abline(h = 0.95, col = "red")
plot(RpowerTable[,1], RpowerTable[,5], xlab = "soft threshold (power)", ylab = "mean connectivity", type = "n")
text(RpowerTable[,1], RpowerTable[,5], labels = powers1, cex = cex1, col = "red")

sft$powerEstimate

power = 4
par(mfrow=c(1, 1))
k.data <- softConnectivity(vsdData, power = power, type = type) -1
scaleFreePlot(k.data, main = paste("data set I, power=", power), truncated = F)

kCut <- 3601
kRank <- rank(-k.data)
vardata <- apply(t(vsdData), 2, var)
restK <- kRank <= kCut & vardata >0 & vardata > 0 
ADJdata <- adjacency(datExpr = t(vsdData)[,restK], power = betal)
dissTOM <- TOMdist(ADJdata)
hierTOM <- hclust(as.dist(dissTOM), method = "average")
par(mfrow = c(1,1))
plot(hierTOM, labels = F, main = "dendrogram, 3600 mast connected in data set I")


net = blockwiseModules(vsdData, power = power, maxBlockSize = 40000,
                       TOMType = type, minModuleSize = 30,
                       reassignThreshold = 0, mergeCutHeight = 0.25,
                       numericLabels = TRUE, pamRespectsDendro = FALSE,
                       saveTOMs=TRUE, corType = "pearson", 
                       maxPOutliers=1, loadTOMs=TRUE,
                       saveTOMFileBase = paste0("wgcna", ".tom"),
                       verbose = 3)


















# not used
A = adjacency(t(vsdData), type = "signed")

k = as.numeric(apply(A, 2, sum)) - 1
# standardized connectivity
Z.k = scale(k)

# Designate samples as outlying if their Z.k value is below the threshold
thresholdZ.k = -2.5  # often -2.5

# the color vector indicates outlyingness (red)
outlierColor = ifelse(Z.k < thresholdZ.k, "red", "black")

# calculate the cluster tree using flahsClust or hclust
sampleTree = flashClust(as.dist(1 - A), method = "average")
# Convert traits to a color representation: where red indicates high
# values
traitColors = data.frame(numbers2colors(datTraits, signed = FALSE))
dimnames(traitColors)[[2]] = paste(names(datTraits), "C", sep = "")
datColors = data.frame(outlierC = outlierColor, traitColors)
# Plot the sample dendrogram and the colors underneath.
plotDendroAndColors(sampleTree, groupLabels = names(datColors), colors = datColors, 
                    main = "Sample dendrogram and trait heatmap")