from scipy.stats import gmean
import pandas as pd
import numpy as np


# ref https://gitlab.com/georgy.m/conorm/-/blob/main/conorm/normalize.py
# R source: https://rdrr.io/bioc/edgeR/src/R/calcNormFactors.R
def tmm_norm_factors(data, trim_lfc=0.3, trim_mag=0.05, index_ref=None):
    """
    Compute Trimmed Means of M-values norm factors.

    Parameters
    ----------
    data : array_like
        Counts dataframe to normalize (rows are genes). Most often can be
        either pandas DataFrame or an numpy matrix.
    trim_lfc : float, optional
        Quantile cutoff for M_g (logfoldchanges). The default is 0.3.
    trim_mag : float, optional
        Quantile cutoff for A_g (log magnitude). The default is 0.05.
    index_ref : float, str, optional
        Reference index or column name to use as reference in the TMM
        algorithm. The default is None.

    Returns
    -------
    tmms : np.ndarray or pd.DataFrame
        Norm factors.

    """

    x = np.array(data, dtype=float).T
    lib_size = x.sum(axis=1)
    mask = x == 0
    if index_ref is None:
        x[:, np.all(mask, axis=0)] = np.nan
        p75 = np.nanpercentile(x, 75, axis=1)
        index_ref = np.argmin(abs(p75 - p75.mean()))
    mask[:, mask[index_ref]] = True
    x[mask] = np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        norm_x = x / lib_size[:, np.newaxis]
        logs = np.log2(norm_x)
        m_g =  logs - logs[index_ref]
        a_g = (logs + logs[index_ref]) / 2

        perc_m_g = np.nanquantile(m_g, [trim_lfc, 1 - trim_lfc], axis=1,
                                  interpolation='nearest')[..., np.newaxis]
        perc_a_g = np.nanquantile(a_g, [trim_mag, 1 - trim_mag], axis=1,
                                  interpolation='nearest')[..., np.newaxis]
        mask |= (m_g < perc_m_g[0]) | (m_g > perc_m_g[1])
        mask |= (a_g < perc_a_g[0]) | (a_g > perc_a_g[1])
        w_gk = (1 - norm_x) / x
        w_gk = 1 / (w_gk + w_gk[index_ref])
    w_gk[mask] = 0
    m_g[mask] = 0
    w_gk /= w_gk.sum(axis=1)[:, np.newaxis]
    tmms = np.sum(w_gk * m_g, axis=1)
    tmms -= tmms.mean()
    tmms = 2 ** tmms
    if type(data) is pd.DataFrame:
        tmms = pd.DataFrame(tmms, index=data.columns,
                            columns=['norm.factors'])
    return tmms


def tmm(data, trim_lfc=0.3, trim_mag=0.05, index_ref=None,
        return_norm_factors=False):
    """
    Normalize counts matrix by Trimmed Means of M-values (TMM).

    Parameters
    ----------
    data : array_like
        Counts dataframe to normalize (rows are genes). Most often can be
        either pandas DataFrame or an numpy matrix.
    trim_lfc : float, optional
        Quantile cutoff for M_g (logfoldchanges). The default is 0.3.
    trim_mag : float, optional
        Quantile cutoff for A_g (log magnitude). The default is 0.05.
    index_ref : float, str, optional
        Reference index or column name to use as reference in the TMM
        algorithm. The default is None.
    return_norm_factors : bool, optional
        If True, then norm factors are also returned. The default is False.

    Returns
    -------
    data : array_like
        Normalized data.

    """
    
    nf = tmm_norm_factors(data, trim_lfc=trim_lfc, trim_mag=trim_mag,
                          index_ref=index_ref)
    if return_norm_factors:
        return data / np.array(nf).flatten(), nf
    return data / np.array(nf).flatten()




def cal_logcpm(df):
    df = df.copy()
    

class Normalizer:
    def __init__(self, df):
        self.df = df
        
    @staticmethod
    def cal_rle(df):
        ref = df.apply(gmean, axis=1)
        for c in df.columns:
            pre = df[c] / ref
            pre[np.isinf(pre)] = np.nan
            gene_factor = np.nanmedian(pre)
            df[c] = df[c] / gene_factor
        return df
    
    
    @staticmethod
    def cal_tmm(df, **kwargs):
        return tmm(df, **kwargs)
    
    
    def get(self, unit, **kwargs):
        df = self.df.copy()
        if unit == "tmm":
            return self.cal_tmm(df=df, **kwargs)
        if unit == "rle":
            return self.cal_rle(df=df, **kwargs)



def cal_TMMwsp(df):
    df = df.copy()