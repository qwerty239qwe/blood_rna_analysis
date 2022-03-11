import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gmean

from pathlib import Path
import re
from functools import reduce

def get_needed(parent_dir, 
               keyword="htseq-count.txt",
               dest_dir=Path("../data/htseq_counts")):
    for d in parent_dir.iterdir():
        print(d.name)
        for fn in (d / Path("6.STATS")).iterdir():
            if keyword in fn.name:
                print(fn)
                dest = dest_dir / Path(fn.name)
                dest.write_text(fn.read_text())
                
                
def _extract_tpm_files(fn):
    if fn.is_file() and "sorted.dedup_genes.out" in fn.name:
        df = pd.read_csv(fn, sep='\t', index_col=0)[["TPM"]]
        df.columns=[fn.name[:fn.name.index("sorted.dedup_genes.out")]]
        return df


def _extract_cnt_files(fn):
    if fn.is_file():
        df = pd.read_csv(fn, sep='\t', index_col=0, header=None)
        df.columns=[fn.name[:fn.name.index("htseq-count.txt")]]
        return df
        
    
def merge_data(path_to_data = "../data/htseq_counts_2", 
               saved_path="../data/mg_counts.tsv", is_counts=True):
    dfs = []
    for fn in Path(path_to_data).iterdir():
        if is_counts:
            f = _extract_cnt_files(fn)
        else:
            f = _extract_tpm_files(fn)
        if f is not None:
            dfs.append(f)
    
    new_mg = pd.concat(dfs, axis=1)[:-9] if is_counts else pd.concat(dfs, axis=1)
    new_mg = new_mg.drop(columns="LPH_HC2JFDSX2.")
    new_mg = new_mg.fillna(0)
    pattern = re.compile("\d\d\dA")
    new_mg.columns = [("C" if "Control" in c else "R") + re.findall(pattern, c)[0] for c in new_mg.columns]
    new_mg = new_mg.sort_index(axis=1)
    new_mg = new_mg.rename(columns={"C001A": "R050A"}).sort_index(axis=1)
    new_mg.index.name = "gene"
    new_mg.to_csv(saved_path, sep='\t')
    return new_mg


def cal_rle(df):
    df = df.copy()
    ref = df.apply(gmean, axis=1)
    
    for c in df.columns:
        gene_factor = np.nanmedian(df[c] / ref)
        df[c] = df[c] / gene_factor
    return df