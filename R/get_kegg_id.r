library(downloader)

# This file is modified from pathview's source code

build_Anno <- function(path2gene, path2name) {
  if (!exists(".Anno_clusterProfiler_Env", envir = .GlobalEnv)) {
    pos <- 1
    envir <- as.environment(pos) 
    assign(".Anno_clusterProfiler_Env", new.env(), envir = envir)
  }
  Anno_clusterProfiler_Env <- get(".Anno_clusterProfiler_Env", envir= .GlobalEnv)
  
  if(class(path2gene[[2]]) == 'list') {
    ## to compatible with tibble
    path2gene <- cbind(rep(path2gene[[1]],
                           times = vapply(path2gene[[2]], length, numeric(1))),
                       unlist(path2gene[[2]]))
  }
  
  path2gene <- as.data.frame(path2gene) 
  path2gene <- path2gene[!is.na(path2gene[,1]), ]
  path2gene <- path2gene[!is.na(path2gene[,2]), ]
  path2gene <- unique(path2gene)
  
  PATHID2EXTID <- split(as.character(path2gene[,2]), as.character(path2gene[,1]))
  EXTID2PATHID <- split(as.character(path2gene[,1]), as.character(path2gene[,2]))
  
  assign("PATHID2EXTID", PATHID2EXTID, envir = Anno_clusterProfiler_Env)
  assign("EXTID2PATHID", EXTID2PATHID, envir = Anno_clusterProfiler_Env)
  
  if ( missing(path2name) || is.null(path2name) || is.na(path2name)) {
    assign("PATHID2NAME", NULL, envir = Anno_clusterProfiler_Env)
  } else {
    path2name <- as.data.frame(path2name)
    path2name <- path2name[!is.na(path2name[,1]), ]
    path2name <- path2name[!is.na(path2name[,2]), ]
    path2name <- unique(path2name)
    PATH2NAME <- as.character(path2name[,2])
    names(PATH2NAME) <- as.character(path2name[,1]) 
    assign("PATHID2NAME", PATH2NAME, envir = Anno_clusterProfiler_Env)
  }
  return(Anno_clusterProfiler_Env)
}

get_KEGG_Env <- function() {
  if (! exists(".KEGG_clusterProfiler_Env", envir = .GlobalEnv)) {
    pos <- 1
    envir <- as.environment(pos)
    assign(".KEGG_clusterProfiler_Env", new.env(), envir=envir)
  }
  get(".KEGG_clusterProfiler_Env", envir = .GlobalEnv)
}

mydownload <- function(url, method, quiet = TRUE, ...) {
  if (capabilities("libcurl")) {
    dl <- tryCatch(utils::download.file(url, quiet = quiet, method = "libcurl", ...),
                   error = function(e) NULL)
  } else {
    dl <- tryCatch(downloader::download(url, quiet = TRUE, method = method, ...),
                   error = function(e) NULL)
  }
  return(dl)
}

kegg_rest <- function(rest_url) {
  ## content <- tryCatch(suppressWarnings(readLines(rest_url)), error=function(e) NULL)
  ## if (is.null(content))
  ##     return(content)
  
  message("Reading KEGG annotation online:\n" )
  f <- tempfile()
  ## dl <- tryCatch(downloader::download(rest_url, destfile = f, quiet = TRUE),
  # if (capabilities("libcurl")) {
  # dl <- tryCatch(utils::download.file(rest_url, destfile = f, quiet = TRUE, method = "libcurl"),
  # error = function(e) NULL)
  # } else {
  # dl <- tryCatch(downloader::download(rest_url, destfile = f, quiet = TRUE),
  # error = function(e) NULL)
  # }
  
  dl <- mydownload(rest_url, destfile = f)
  
  if (is.null(dl)) {
    message("fail to download KEGG data...")
    return(NULL)
  }
  
  content <- readLines(f)
  
  content %<>% strsplit(., "\t") %>% do.call('rbind', .)
  res <- data.frame(from=content[,1],
                    to=content[,2])
  return(res)
}


kegg_link <- function(target_db, source_db) {
  url <- paste0("http://rest.kegg.jp/link/", target_db, "/", source_db, collapse="")
  kegg_rest(url)
}

kegg_list <- function(db) {
  url <- paste0("http://rest.kegg.jp/list/", db, collapse="")
  kegg_rest(url)
}

download_KEGG <- function(species, keggType="KEGG", keyType="kegg") {
  KEGG_Env <- get_KEGG_Env()
  
  use_cached <- FALSE
  
  if (exists("organism", envir = KEGG_Env, inherits = FALSE) &&
      exists("_type_", envir = KEGG_Env, inherits = FALSE) ) {
    
    org <- get("organism", envir=KEGG_Env)
    type <- get("_type_", envir=KEGG_Env)
    
    if (org == species && type == keggType &&
        exists("KEGGPATHID2NAME", envir=KEGG_Env, inherits = FALSE) &&
        exists("KEGGPATHID2EXTID", envir=KEGG_Env, inherits = FALSE)) {
      
      use_cached <- TRUE
    }
  }
  
  if (use_cached) {
    KEGGPATHID2EXTID <- get("KEGGPATHID2EXTID", envir=KEGG_Env)
    KEGGPATHID2NAME <- get("KEGGPATHID2NAME", envir=KEGG_Env)
  } else {
    if (keggType == "KEGG") {
      kres <- download.KEGG.Path(species)
    } else {
      kres <- download.KEGG.Module(species)
    }
    
    KEGGPATHID2EXTID <- kres$KEGGPATHID2EXTID
    KEGGPATHID2NAME <- kres$KEGGPATHID2NAME
    
    assign("organism", species, envir=KEGG_Env)
    assign("_type_", keggType, envir=KEGG_Env)
    assign("KEGGPATHID2NAME", KEGGPATHID2NAME, envir=KEGG_Env)
    assign("KEGGPATHID2EXTID", KEGGPATHID2EXTID, envir=KEGG_Env)
  }
  
  if (keyType != "kegg") {
    need_idconv <- FALSE
    idconv <- NULL
    if (use_cached &&
        exists("key", envir=KEGG_Env, inherits = FALSE) &&
        exists("idconv", envir=KEGG_Env, inherits = FALSE)) {
      
      key <- get("key", envir=KEGG_Env)
      if (key == keyType) {
        idconv <- get("idconv", envir=KEGG_Env)
      } else {
        need_idconv <- TRUE
      }
    } else {
      neec_idconv <- TRUE
    }
    
    if (need_idconv || is.null(idconv)) {
      idconv <- KEGG_convert("kegg", keyType, species)
      assign("key", keyType, envir=KEGG_Env)
      assign("idconv", idconv, envir=KEGG_Env)
    }
    colnames(KEGGPATHID2EXTID) <- c("from", "kegg")
    KEGGPATHID2EXTID <- merge(KEGGPATHID2EXTID, idconv, by.x='kegg', by.y='from')
    KEGGPATHID2EXTID <- unique(KEGGPATHID2EXTID[, -1])
  }
  
  return(list(KEGGPATHID2EXTID = KEGGPATHID2EXTID,
              KEGGPATHID2NAME  = KEGGPATHID2NAME))
}

prepare_KEGG <- function(species, KEGG_Type="KEGG", keyType="kegg") {
  kegg <- download_KEGG(species, KEGG_Type, keyType)
  build_Anno(kegg$KEGGPATHID2EXTID,
             kegg$KEGGPATHID2NAME)
}

download.KEGG.Path <- function(species) {
  keggpathid2extid.df <- kegg_link(species, "pathway")
  if (is.null(keggpathid2extid.df))
    stop("'species' should be one of organisms listed in 'http://www.genome.jp/kegg/catalog/org_list.html'...")
  keggpathid2extid.df[,1] %<>% gsub("[^:]+:", "", .)
  keggpathid2extid.df[,2] %<>% gsub("[^:]+:", "", .)
  
  keggpathid2name.df <- kegg_list("pathway")
  keggpathid2name.df[,1] %<>% gsub("path:map", species, .)
  
  ## if 'species="ko"', ko and map path are duplicated, only keep ko path.
  ##
  ## http://www.kegg.jp/dbget-bin/www_bget?ko+ko00010
  ## http://www.kegg.jp/dbget-bin/www_bget?ko+map0001
  ##
  keggpathid2extid.df <- keggpathid2extid.df[keggpathid2extid.df[,1] %in% keggpathid2name.df[,1],]
  
  return(list(KEGGPATHID2EXTID=keggpathid2extid.df,
              KEGGPATHID2NAME=keggpathid2name.df))
}

download.KEGG.Module <- function(species) {
  keggmodule2extid.df <- kegg_link(species, "module")
  if (is.null(keggmodule2extid.df)) {
    stop("'species' should be one of organisms listed in 'http://www.genome.jp/kegg/catalog/org_list.html'...")
  }
  
  keggmodule2extid.df[,1] %<>% gsub("[^:]+:", "", .) %>% gsub(species, "", .) %>% gsub("^_", "", .)
  keggmodule2extid.df[,2] %<>% gsub("[^:]+:", "", .)
  
  keggmodule2name.df <- kegg_list("module")
  keggmodule2name.df[,1] %<>% gsub("md:", "", .)
  return(list(KEGGPATHID2EXTID=keggmodule2extid.df,
              KEGGPATHID2NAME =keggmodule2name.df))
}

USER_DATA <- prepare_KEGG("mmu", "KEGG", "kegg")
p2e <- get("PATHID2EXTID", envir=USER_DATA)