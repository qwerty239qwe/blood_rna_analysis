import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from tqdm import tqdm 

table_file_name = "F:\\rare_disease_genes_tables.xlsx"
data_path = "F:\\ntuh\\hg38\\"
data_dir_dic = {dir: os.listdir(data_path+"\\"+dir) for dir in os.listdir(data_path)}
data_dirs = [i for lab, dirlis in data_dir_dic.items() for i in dirlis]
out_paths = {sub_dir: f"{data_path}{dir}\\{sub_dir}\\6.STATS\\{sub_dir}.sorted_genes.out" for dir, sub_dirs in data_dir_dic.items() for sub_dir in sub_dirs}
uni_paths = {sub_dir: f"{data_path}{dir}\\{sub_dir}\\6.STATS\\{sub_dir}.sorted_genes.uni" for dir, sub_dirs in data_dir_dic.items() for sub_dir in sub_dirs}
ent_paths = {sub_dir: f"{data_path}{dir}\\{sub_dir}\\6.STATS\\{sub_dir}.sorted_genes.ent" for dir, sub_dirs in data_dir_dic.items() for sub_dir in sub_dirs}

ref_table = pd.read_excel(table_file_name, sheet_name="Table S4",header=1)
ref_data = {
            'N': set([r for r in ref_table["Neurology"].to_list() if pd.notna(r)]),
            'O': set([r for r in ref_table["Ophtalmology"].to_list() if pd.notna(r)]),
            'H': set([r for r in ref_table["Hematology"].to_list() if pd.notna(r)]),
            'M': set([r for r in ref_table["Musculoskeletal"].to_list() if pd.notna(r)]),
            'OMIM': set([r for r in ref_table["OMIM"].to_list() if pd.notna(r)]),
            }

def get_mapped_data(file_paths, ref_data=ref_data, data_dir_dic=data_dir_dic, data_dirs=data_dirs, dup_exist=False):
    dfs = {dir: pd.read_csv(file_paths[dir], sep="\t")[["Gene_Id", "TPM"]] for dirs in list(data_dir_dic.values()) for dir in dirs}
    for s, df in dfs.items():
        df["Gene_Id"] = df["Gene_Id"].apply(lambda x: x[:x.index(".")] if "." in x else x)
        if dup_exist:
            df["TPM"] = df.groupby("Gene_Id")["TPM"].transform(sum)
            df = df.drop_duplicates(subset=["Gene_Id"], ignore_index=True)
        
    gen_dict = {s: dfs[s]["Gene_Id"].to_list() for s in tqdm(data_dirs)}

    mapped_data = {s: {ref_id: {r: dfs[s][dfs[s]["Gene_Id"]==r]["TPM"].values[0] if r in gen_dict[s] else -1 
                                for r in ref_lis} 
                                for ref_id, ref_lis in ref_data.items()} 
                                for dirs in list(data_dir_dic.values()) for s in tqdm(dirs)}
    return mapped_data

def get_median_expr_dict(mapped_data, ref_data=ref_data, save_df=True, df_name="out"):
    median_expr_dict = {ref_id: {r: np.array([dat_dic[ref_id][r] for s, dat_dic in mapped_data.items() if dat_dic[ref_id][r] >= 0.0]) for r in ref_lis} for ref_id, ref_lis in ref_data.items()}
    median_expr_dict = {ref_id: {r: np.median(exp) if len(exp) > 0 else -1 for r, exp in ref_dic.items()} for ref_id, ref_dic in median_expr_dict.items()}
    if save_df:
        df_out = pd.DataFrame({"gene": [g for ref_id, dic in median_expr_dict.items() for g, e in dic.items()], 
                                "median TPM": [e for ref_id, dic in median_expr_dict.items() for g, e in dic.items()], 
                                "dataset": [ref_id for ref_id, dic in median_expr_dict.items() for _ in range(len(dic))]})
        df_out.to_csv(f"{df_name}_median.tsv", sep='\t')
    return median_expr_dict
        
def plot_fig(median_expr_dict, save_df=True, fg_name="out"):
    feature_dict = {}
    for ref_id, gene_median in median_expr_dict.items():
        val_arr = np.array(list(gene_median.values()))
        feature_dict[ref_id] = {"high": 100 * val_arr[val_arr >= 0.1].shape[0] / len(ref_data[ref_id]), 
                                "median": 100 * val_arr[(val_arr >= 0.1) & (val_arr < 10)].shape[0] / len(ref_data[ref_id]), 
                                "low": 100 * val_arr[(val_arr >= 0) & (val_arr < 0.1)].shape[0] / len(ref_data[ref_id]),
                                "miss": 100 * val_arr[val_arr == -1].shape[0] / len(ref_data[ref_id])}
    df = pd.DataFrame({'percentage': [v for lab, dic in feature_dict.items() for k, v in dic.items()], 
                        'class': [lab for lab, dic in feature_dict.items() for i in range(len(dic))], 
                        'expr': [k for lab, dic in feature_dict.items() for k, v in dic.items()]})
    if save_df:
        df.to_csv(f"{fg_name}_figure.csv")
    plt.figure(figsize=(11, 7))
    sns.set(style="whitegrid")
    ax = sns.barplot(x="expr", y="percentage", data=df, hue='class')
    plt.savefig(f"ptg_{fg_name}.png", dpi=300)
    
    
def main():
    # gene.out files
    mapped_data = get_mapped_data(out_paths, ref_data=ref_data, data_dir_dic=data_dir_dic, data_dirs=data_dirs, dup_exist=False)
    median_expr_dict = get_median_expr_dict(mapped_data, ref_data=ref_data, save_df=True, df_name="out")
    plot_fig(median_expr_dict, save_df=True, fg_name="out")
    
    # gene.ent files
    mapped_data = get_mapped_data(ent_paths, ref_data=ref_data, data_dir_dic=data_dir_dic, data_dirs=data_dirs, dup_exist=True)
    median_expr_dict = get_median_expr_dict(mapped_data, ref_data=ref_data, save_df=True, df_name="ent")
    plot_fig(median_expr_dict, save_df=True, fg_name="ent")
    
    # gene.uni files
    mapped_data = get_mapped_data(uni_paths, ref_data=ref_data, data_dir_dic=data_dir_dic, data_dirs=data_dirs, dup_exist=True)
    median_expr_dict = get_median_expr_dict(mapped_data, ref_data=ref_data, save_df=True, df_name="uni")
    plot_fig(median_expr_dict, save_df=True, fg_name="uni")
    
    
if __name__ == "__main__":
    main()