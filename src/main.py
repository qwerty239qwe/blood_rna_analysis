import pandas as pd
from pathlib import Path
import argparse

from data_processing.merge_data import *

parser = argparse.ArgumentParser(description='The main file to execute the pipeline')
parser.add_argument('--step', "-s", type=int,
                    help='The number of ')


def merge_data():
    _ = merge_data(path_to_data = "../data/htseq_counts_2/", 
           saved_path="../data/mg_counts_2203.tsv", is_counts=True)
    _ = merge_data(path_to_data = "../data/TPM_2/", 
           saved_path="../data/mg_tpm_2203.tsv", is_counts=False)
    cal_rle(counts).to_csv("../data/mg_RLE.tsv", sep='\t')
    
    
    