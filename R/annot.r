library(biomaRt)
library(glue)

hsaDBName <- "hsapiens_gene_ensembl"
OUTDIR <- "F:/ntuh"
tabFn <- "F:/ntuh/20201221_data/catboost.csv"



anoTable <- function(fn, ref.col, ref.type, mart=mart, outdir=OUTDIR){
  tab <- read.csv(fn, sep='\t', header=1)
  tab <- data.frame(tab[, c(ref.col)])
  fn_last <- unlist(strsplit(fn, "/"))[length(unlist(strsplit(fn, "/")))]
  names(tab) <- ref.type
  g = getBM(c("hgnc_symbol", "ensembl_gene_id", "ensembl_gene_id_version", 
              "description", "mim_gene_description", "mim_gene_accession", "go_id"), 
            ref.type, 
            tab[, c(ref.type)], mart)
  
  g <- merge(x = g, y = tab, by=ref.type)
  print(glue("{outdir}/Annot_{fn_last}"))
  write.table(g, glue("{outdir}/Annot_{fn_last}"), sep='\t')
}

if (interactive()){
  mart = useMart("ensembl", dataset=hsaDBName)
  anoTable(tabFn, "DGE", "ensembl_gene_id_version", mart, outdir = OUTDIR)
}


tab <- read.csv(tabFn, header = 1)
gene_list <- tab["catboost"]
if (interactive()){
  mart = useMart("ensembl", dataset=hsaDBName)
  g = getBM(c("hgnc_symbol", "ensembl_gene_id", "description"), 
            "ensembl_gene_id", 
            gene_list, mart)
  write.csv(g, "catboost_predgene_info.csv")
}
