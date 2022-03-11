library(ggplot2)
library(scales)
library(glue)
library(dplyr)
library(grid) # grob
library(ggrepel)

topDir <- "F:/ntuh/1_DEG/DESeq_rmsv_2"
COMPR <- c("G2_G1", "G2_G3", "G3_G1")

# col names
X <- c("baseMean", "logCPM", "AveExpr")
Y <- c("log2FoldChange", "logFC", "logFC")
P <- c("padj", "FDR", "adj.P.Val")
XLABELS <- c("mean of normalized counts", "logCPM", "average log2-expression values")
YLABELS <- c("log2 fold change", "log2 fold change", "log2 fold change")
PLABELS <- c("p-value", "p-value", "p-value")

OUTDIR <- "F:/ntuh"

plotMAs <- function(comp, degDir, x, y, p, xlab, ylab, yThres=log2(1.5), pThres=0.1, outdir=OUTDIR){
  t <- read.table(file.path(degDir, glue("DE_{comp}.tsv")), sep='\t')
  t <- t[complete.cases(t),]
  t$de <- 0
  t$de[(t[, y] > yThres)&(t[, p] < pThres)] <- 1
  t$de[(t[, y] < (-yThres))&(t[, p] < pThres)] <- (-1)
  t$de <- as.factor(t$de)
  deLevels <- levels(t$de)
  color_values <- c()
  color_labels <- c()
  if (-1 %in% deLevels){
    color_values <- c(color_values, "#0466c8")
    color_labels <- c(color_labels, "Down regulated")
  }
  color_values <- c(color_values, "#999999")
  color_labels <- c(color_labels, "Not DE genes")
  if ( 1 %in% deLevels){
    color_values <- c(color_values, "#e63946")
    color_labels <- c(color_labels, "Up regulated")
  }
  
  degname <- strsplit(degDir, "/")
  degname <- degname[[1]][length(degname[[1]])]
  
  p <- ggplot(t, aes_string(x=x, y=y, color="de")) + 
    geom_point(size=1) +
    scale_color_manual(name = "DE genes", values=color_values, labels=color_labels) + 
    geom_hline(yintercept=0, size=1.5, alpha=0.4) +
    labs(x = xlab, y = ylab, title = paste(strsplit(comp, "_")[[1]], collapse = " vs "))
  if (grepl("DESeq", degname)){
    p + scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                      labels = trans_format("log10", math_format(10^.x)))
  }
  ggsave(file.path(OUTDIR, glue("{degname}_{comp}.jpeg")), width = 9, height = 9, units = "in", device="jpeg")
}

reverselog_trans <- function(base = exp(1)) {
  trans <- function(x) -log(x, base)
  inv <- function(x) base^(-x)
  trans_new(paste0("reverselog-", format(base)), trans, inv, 
            log_breaks(base = base), 
            domain = c(1e-100, Inf))
}

plotVolcano <- function(comp, degDir, anotFn, y, p, xlab, ylab, yThres=log2(1.5), pThres=0.1, outdir=OUTDIR, pSig=1e-8){
  print(file.path(degDir, glue("DE_{comp}.tsv")))
  t <- read.table(file.path(degDir, glue("DE_{comp}.tsv")), sep='\t', header=1)
  annot <- read.table(anotFn, sep='\t', header=1)
  colnames(annot) <- c("Symbol", "Gene", "des")
  
  t <- t[complete.cases(t),]
  t$de <- 0
  t$de[(t[, y] > yThres)&(t[, p] < pThres)] <- 1
  t$de[(t[, y] < (-yThres))&(t[, p] < pThres)] <- (-1)
  t$de <- as.factor(t$de)
  t <- merge(t, annot, by="Gene")
  dat <- data.frame(DE = t$de)
  dat <- dat %>% 
    group_by(DE) %>%
    summarise(no_rows = length(DE))
  n_upreg <- dat[dat$DE==1,]$no_rows
  n_downreg <- dat[dat$DE==-1,]$no_rows
  deLevels <- levels(t$de)
  color_values <- c()
  color_labels <- c()
  if (-1 %in% deLevels){
    color_values <- c(color_values, "#0466c8")
    color_labels <- c(color_labels, "Down regulated")
  }
  color_values <- c(color_values, "#999999")
  color_labels <- c(color_labels, "Not DE genes")
  if ( 1 %in% deLevels){
    color_values <- c(color_values, "#e63946")
    color_labels <- c(color_labels, "Up regulated")
  }
  
  degname <- strsplit(degDir, "/")
  degname <- degname[[1]][length(degname[[1]])]
  grobUp <- grobTree(textGrob(glue("{n_upreg}"), x=0.9, y=0.9, gp=gpar(col="#e63946", fontsize=18)))
  grobDown <- grobTree(textGrob(glue("{n_downreg}"), x=0.1, y=0.9, gp=gpar(col="#0466c8", fontsize=18)))
  p <- ggplot(t, aes_string(x=y, y=p, color="de")) + 
    geom_point(size=1) +
    scale_color_manual(name = "DE genes", values=color_values, labels=color_labels) + 
    # geom_vline(xintercept=0, size=1.5, alpha=0.4) +
    geom_vline(xintercept=yThres, size=1, alpha=0.3) +
    geom_vline(xintercept=-yThres, size=1, alpha=0.3) +
    geom_hline(yintercept=pThres, size=1, alpha=0.3) +
    labs(x = xlab, y = ylab, title = paste(strsplit(comp, "_")[[1]], collapse = " vs ")) +
    scale_y_continuous(trans=reverselog_trans(10)) + 
    annotation_custom(grobUp) +
    annotation_custom(grobDown) + 
    geom_text_repel(
      data = t[(t[, p] < pSig)&(complete.cases(t))&(!duplicated(t)), ],
      aes(label = Symbol),
      size = 5,
      box.padding = unit(0.35, "lines"),
      point.padding = unit(0.3, "lines"),
      show.legend = F
    )
  # scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
  #               labels = trans_format("log10", math_format(10^.x))) +
  
  ggsave(file.path(OUTDIR, glue("volcano_{degname}_{comp}.jpeg")), width = 9, height = 9, units = "in", device="jpeg")
}

n <- sapply(COMPR, plotVolcano, degDir=topDir, 
            anotFn="F:/ntuh/processed_data/ensem_to_symb.tsv", 
            y=Y[1], p=P[1], 
            xlab=YLABELS[1], ylab=PLABELS[1], yThres=log2(1.5), pThres=0.1, outdir=OUTDIR)
