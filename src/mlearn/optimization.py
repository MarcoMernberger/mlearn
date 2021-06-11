#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""optimization.py: Contains ...."""

from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple, Any, Union
from pandas import DataFrame
from numpy import ndarray
import pandas as pd
import pypipegraph as ppg
import mbf_r
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri as numpy2ri
import itertools


__author__ = "Marco Mernberger"
__copyright__ = "Copyright (c) 2020 Marco Mernberger"
__license__ = "mit"


def nbclust(
    data: DataFrame,
    kmin: int,
    kmax: int,
    method: str = "kmeans",
    index: str = "all",
    distance: str = "euclidean",
    precomputed: bool = False,
    alphabeale: float = 0.1,
) -> Tuple[DataFrame, DataFrame, DataFrame, int]:
    """
    Wrapper for the R package nbclust. Calculates a number of metrics that
    estimate the number of clusters in a data structure.

    Parameters
    ----------
    data : DataFrame
        data to be clustered. This can be a dataframe of
        n instances and m features of shape (n, m) or it can be a
        precomputed distance matrix of shape (n,n).
    kmin : int
        Minimum number of clusters.
    kmax : int
        Maximum number of clusters.
    method : str
        Clustering method to use, by default "kmeans".
    index : str
        Index to calculate, by default  "all".
    distance : str, optional
        [description], by default "euclidean".
    dissimilarity : ndarray, optional
        [description], by default None.
    alphabeale : float, optional
        Significance value for Beale's Index, by default 0.1,

    Returns
    -------
    Tuple[DataFrame, DataFrame, DataFrame, int]
        indices contains all indices and their values, best_nc contains the best
        number of clusters proposed by each index, best_partition is the partition
        corresponding to the best muber of clusters, majority is the optimal
        number of clusters proposed by majority voting. 

    Raises
    ------
    ValueError
        If invalid method is supplied.
    ValueError
        If invalid index is supplied.
    """
    ro.r("library(NbClust)")
    accepted_methods = [
        "ward.D",
        "ward.D2",
        "single",
        "complete",
        "average",
        "mcquitty",
        "median",
        "centroid",
        "kmeans",
    ]
    accepted_indices = [
        "kl",
        "ch",
        "hartigan",
        "ccc",
        "scott",
        "marriot",
        "trcovw",
        "tracew",
        "friedman",
        "rubin",
        "cindex",
        "db",
        "silhouette",
        "duda",
        "pseudot2",
        "beale",
        "ratkowsky",
        "ball",
        "ptbiserial",
        "gap",
        "frey",
        "mcclain",
        "gamma",
        "gplus",
        "tau",
        "dunn",
        "hubert",
        "sdindex",
        "dindex",
        "sdbw",
        "all",
        "alllong",
    ]
    if index not in accepted_indices:
        raise ValueError(f"'index' must be one of {accepted_indices}, was {index}.")
    if method not in accepted_methods:
        raise ValueError(f"'method' must be one of {accepted_methods}, was {method}.")
    if precomputed:
        diss = numpy2ri.py2rpy(data.values)
        distance = ro.rinterface.NULL
    else:
        diss = ro.rinterface.NULL
    rdata = mbf_r.convert_dataframe_to_r(data)
    nbclust = ro.r("NbClust")(
        rdata, diss, distance, kmin, kmax, method, index, alphabeale
    )
    indices = ro.r("function(nbclust){as.data.frame(nbclust$All.index)}")(nbclust)
    best_nc = ro.r("function(nbclust){as.data.frame(nbclust$Best.nc)}")(nbclust=nbclust)
    best_partition = ro.r("function(nbclust){as.data.frame(nbclust$Best.partition)}")(nbclust=nbclust)
    indices = mbf_r.convert_dataframe_from_r(indices)
    best_nc = mbf_r.convert_dataframe_from_r(best_nc).transpose()
    best_partition = mbf_r.convert_dataframe_from_r(best_partition).rename(columns={"nbclust$Best.partition": "Best partition"})
    majority = best_nc["Number_clusters"].mode()[0]
    return indices, best_nc, best_partition, majority

