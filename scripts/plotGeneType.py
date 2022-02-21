import pandas as pd
from os import listdir
from os.path import isfile, isdir, join
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

sns.set_style("whitegrid")

def plot_all_bar(tabs, keys, method, type_name, fn=None):
    # plot one genes type of many samples
    fig, ax = plt.subplots(figsize=(6, 11))
    gb_result = [getattr(tab[["gene_biotype", "TPM"]].groupby("gene_biotype"), method)().loc[type_name, :].values[0] for tab in tabs ]
    df = pd.DataFrame({"Sample": keys, "TPM": gb_result})
    ax = sns.barplot(x = "TPM", y = "Sample", data=df, ax=ax)
    plt.tight_layout()
    plt.savefig(f"barplot_all_{method}_{type_name}.png" if not fn else fn, dpi=300)
    plt.show()

def plot_all_box(tabs, keys, type_name, fn=None):
    fig, ax = plt.subplots(figsize=(8, 11))
    dfs = []
    for tab, lab in zip(tabs, keys):
        gb = tab[["gene_biotype", "TPM"]].groupby("gene_biotype")
        df = gb.get_group(type_name)
        df["sample"] = [lab for _ in range(df.shape[0])]
        dfs.append(df)
    mg_df = pd.concat(dfs)
    mg_df["log2TPM"] = mg_df["TPM"].apply(lambda x: np.log2(x+1))
    g = sns.boxplot(y="sample", x="log2TPM", data=mg_df, ax=ax, width=0.2, orient='h')
    ax.set_title(type_name)
    plt.tight_layout()
    plt.savefig(f"boxplot_all_{type_name}.png" if not fn else fn, dpi=300)
    plt.show()
    
def plot_bar(tab, tab_name, method, fn=None):
    # plot all genes type of one sample
    fig, ax = plt.subplots(figsize=(9, 11))
    assert method in ["sum", "mean", "median"], "method should be one of ['sum', 'mean', 'median']"
    gb = getattr(tab[["gene_biotype", "TPM"]].groupby("gene_biotype"), method)()
    df = pd.DataFrame({"Type": gb.index.to_list(), "TPM": gb["TPM"].to_list()})
    ax = sns.barplot(x = "TPM", y = "Type", data=df, ax=ax)
    plt.tight_layout()
    plt.savefig(f"barplot_{tab_name}_{method}.png" if not fn else fn, dpi=300)
    plt.show()

def plot_box(tab, tab_name, fn=None):
    fig, ax = plt.subplots(figsize=(11, 11))
    gb = tab[["gene_biotype", "TPM"]].groupby("gene_biotype")
    labels = tab["gene_biotype"].unique()
    dfs = [gb.get_group(lab) for i, lab in enumerate(labels)]
    mg_df = pd.concat(dfs)
    mg_df["log2TPM"] = mg_df["TPM"].apply(lambda x: np.log2(x+1))
    ax = sns.boxplot(y="gene_biotype", x="log2TPM", data=mg_df, ax=ax, width=0.2, orient='h')
    plt.tight_layout()
    plt.savefig(f"boxplot_{tab_name}.png" if not fn else fn, dpi=300)
    plt.show()
    
def plot_dist(tabs, tab_name, type_name, fn=None):
    #plt.figure(figsize=(11, 11))
    dfs = []
    for lab, tab in tabs.items():
        gb = tab[["gene_biotype", "TPM"]].groupby("gene_biotype")
        labels = tab["gene_biotype"].unique()
        df = gb.get_group(type_name)
        df["sample"] = [lab for _ in range(df.shape[0])]
        df["log2TPM"] = df["TPM"].apply(np.log2)
        dfs.append(df)
    mg_df = pd.concat(dfs)
    
    g = sns.displot(data = mg_df, x="log2TPM", hue="sample", kde=True, height=11, palette="twilight_shifted_r")
    g.set_titles(type_name)
    plt.tight_layout()
    plt.savefig(f"distplot_{tab_name}_{type_name}.png" if not fn else fn, dpi=300)
    plt.show()
    
def make_gene_num_table(tabs):
    ret_df = list(tabs.values())[0].set_index(["gene_biotype", "TPM"]).count(level="gene_biotype").loc[:, "ensembl_gene_id_version"].to_frame()
    ret_df.columns = [list(tabs.keys())[0]]
    for lab, tab in tabs.items():
        if lab != list(tabs.keys())[0]:
            ret_df[lab] = tab.set_index(["gene_biotype", "TPM"]).count(level="gene_biotype")["ensembl_gene_id_version"]
    return ret_df