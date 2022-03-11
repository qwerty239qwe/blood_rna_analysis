library(STRINGdb)
library(biomaRt)
library(glue)

rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

hsaDBName <- "hsapiens_gene_ensembl"
hostName <- "http://may2015.archive.ensembl.org"

string_db <- STRINGdb$new( version="11", species=9606,
                           score_threshold=200, input_directory="")

annotTable <- read.delim(file.path("../data/gene_peptide.tsv"), sep='\t')[,2:3]
table_dest <- "../results/PPI"
plot_dest <- "../results/PPI"

grp <- "4_2"
out_table <- glue("G{grp}.tsv")
out_plot <- glue("G{grp}")
table_name <- glue("DE_{grp}.tsv")
table <- read.table(file.path("../results/DEG/tables", table_name), sep='\t')
table <- table[order(table$padj, decreasing = F),]
table$gene <- row.names(table)

merged_g <- merge(x = annotTable, y = table, by="gene")
g_mapped <- string_db$map( merged_g[order(merged_g$padj, decreasing = F),], 
                          "ensembl_peptide_id", 
                          removeUnmappedRows = TRUE )

hits <- g_mapped$STRING_id[1:300]

mapped_pval05 <- string_db$add_diff_exp_color( subset(g_mapped, padj<0.05),
                                                        logFcColStr="log2FoldChange" )
enrichment <- string_db$get_enrichment( hits )

annotations <- string_db$get_annotations( hits )

write.table(mapped_pval05, file.path(table_dest, "mapped", out_table), sep = '\t')

write.table(enrichment, file.path(table_dest, "enrichment", out_table), sep = '\t')

write.table(annotations, file.path(table_dest, "annotations", out_table), sep = '\t')

payload_id <- string_db$post_payload( mapped_pval05$STRING_id,
                                      colors=mapped_pval05$color )

jpeg(file.path(plot_dest, glue("{out_plot}_top300.jpeg")),
     width = 9, height = 9, units = "in",
     res = 300, quality = 100)
string_db$plot_network( hits, payload_id=payload_id )
dev.off()

clustersList <- string_db$get_clusters(g_mapped$STRING_id[1:1000])

for (i in c(1: length(clustersList))) {
  if (length(clustersList[[i]]) > 4){
    jpeg(file.path(plot_dest, glue("{out_plot}_cluster_{i}.jpeg")),
         width = 9, height = 9, units = "in",
         res = 300, quality = 100)
    string_db$plot_network(clustersList[[i]], payload_id=payload_id)
    dev.off()
  }
}

