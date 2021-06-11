#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""metrics.py: Contains ...."""

from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple, Any, Union
from sklearn.base import ClusterMixin
from pandas import DataFrame
import pandas as pd
import numpy as np
import pypipegraph as ppg

__author__ = "Marco Mernberger"
__copyright__ = "Copyright (c) 2020 Marco Mernberger"
__license__ = "mit"


def calculate_silhouette(data: DataFrame, model_class: ClusterMixin, kmin: int, kmax: int, sil_metric: str = "euclidean", parameter: Dict = {}) -> DataFrame:
    sil = []
    ks = np.arange(kmin, kmax+1)
    for k in ks:
        model = model_class(n_clusters=k, **parameter)
        model = model.fit(data)
        sil.append(silhouette_score(model, data, sil_metric))
    sil = np.insert(np.array(sil), 0, 0)
    ks = np.insert(ks, 0, kmin-1)
    return pd.DataFrame({"k": ks, "silhouette score": sil})

def silhouette_score(fitted_model, data, metric):
    labels = fitted_model.labels_
    return sklearn.metrics.silhouette_score(data, labels, metric=metric)


def silhouette_samples(fitted_model, data, metric) -> ndarray:
    """"""
    labels = fitted_model.labels_
    return sklearn.metrics.silhouette_samples(data, labels, metric=metric)


def calculate_wss(model, data):
    data = data.copy()
    cols = data.columns.values
    data["cluster"] = model.labels_
    wss = []
    for label, cluster in data.groupby("cluster"):
        centroid = cluster[cols].mean().values.reshape(1, -1)
        vectors = np.concatenate((centroid, cluster[cols].values))
        d = sklearn.metrics.pairwise_distances(vectors, metric='euclidean')
        wss.append(d[0].sum() / len(cluster))
    wss = np.array(wss).mean()
    return wss


def calculate_inertia(model, data):
    data = data.copy()
    cols = data.columns.values
    data["cluster"] = model.labels_
    n_clusters = len(data["cluster"].unique())    
    wk = 0
    for label, cluster in data.groupby("cluster"):
        centroid = cluster[cols].mean().values.reshape(1, -1)
        for _, row in cluster[cols].iterrows():
            wk += np.linalg.norm(centroid-row.values)**2
    return wk


def tibshirani(model, data):
    data = data.copy()
    cols = data.columns.values
    data["cluster"] = model.labels_
    n_clusters = len(data["cluster"].unique())    
    wk = 0
    for label, cluster in data.groupby("cluster"):
        dr = sklearn.metrics.pairwise_distances(cluster, metric="euclidean")
        dr = (dr**2).sum() / (2*len(cluster))
        wk += dr
    return wk
