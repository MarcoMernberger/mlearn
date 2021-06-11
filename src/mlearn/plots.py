#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""plots.py: Contains ...."""

from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple, Any, Union
from matplotlib.figure import Figure
from mplots import MPPlotJob
from sklearn.base import ClusterMixin
from pandas import DataFrame
from metrics import calculate_silhouette
import pandas as pd
import pypipegraph as ppg
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcls
import numpy as np
import sklearn
import mplots
from cycler import cycler


__author__ = "Marco Mernberger"
__copyright__ = "Copyright (c) 2020 Marco Mernberger"
__license__ = "mit"


def plot_matrix(df_ordered: DataFrame, clusters: List = None, title: str = "matrix", **kwargs) -> Figure:
    """
    Plot function for (clustered) matrices.

    Plots a matrix as heatmap via imshow and if given adds cluster labels on the x axis.

    Parameters
    ----------
    df_ordered : DataFrame
        DataFrame with matrix to be plotted.
    clusters : List, optional
        List of cluster indices, by default None
    title : str, optional
        title of the figure, by default "matrix"

    Returns
    -------
    Figure
        Matplotlib Figure.
    """
    dpi = kwargs.get("dpi", 100)
    fontsize = kwargs.get("fontsize", 6)
    fsize = kwargs.get("fsize", 40)
    colormap = kwargs.get("colormap", "seismic")
    show_values = kwargs.get("show_values", False)
    f = plt.figure(figsize=(fsize*1.1, fsize), dpi=dpi)
    gs = f.add_gridspec(10, 11)
    axe_main = f.add_subplot(gs[:, :10])
    axe_col = f.add_subplot(gs[:, 10:11])
    plt.sca(axe_main)
    plt.sca(axe_main)
    silscore = np.nan
    yy = df_ordered.columns.values
    xx = df_ordered.columns.values
    if clusters is not None:
        no_of_clusters = len(set(clusters))
        if 2 <= no_of_clusters < df_ordered.shape[0]:
            sil = sklearn.metrics.silhouette_samples(
                df_ordered.values,
                clusters,
                metric='euclidean'
            )
            silscore = sklearn.metrics.silhouette_score(
                df_ordered.values,
                clusters,
                metric='euclidean',
                random_state=12
                )
            title = f"{title}\nsilhouette score = {silscore}"
            yy = [f"{x} ({y:.2f})" for x, y in zip(df_ordered.columns.values, sil)]
        xx = np.sort(clusters)
    plt.xticks(
        np.arange(df_ordered.shape[1]),
        xx,
        rotation=0,
        fontsize=fontsize
        )
    plt.yticks(
        np.arange(len(df_ordered.columns)),
        yy,
        fontsize=fontsize
        )
    im = plt.imshow(df_ordered, cmap=colormap, aspect="auto")
    if show_values:
        for i in range(len(df_ordered.columns)):
            for j in range(len(df_ordered.index)):
                plt.gca().text(
                    j,
                    i,
                    f"{df_ordered.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="grey"
                    )
    plt.colorbar(im, cax=axe_col, shrink=.9, fraction=.01, pad=5)
    # df_ordered["Cluster"] = clusters
    plt.title(f"{title}")
    plt.show(f)
    return f


def generate_silhouette_plot(data: DataFrame, model_class: ClusterMixin, kmin: int, kmax: int, sil_metric: str = "euclidean", parameter: Dict = {}, **kwargs):
    df = calculate_silhouette(data, model_class, kmin, kmax, sil_metric, parameter)
    f = plot_silhouette(df, sil_metric, **kwargs)
    return f


def plot_silhouette(data: DataFrame, model_class: ClusterMixin, kmin: int, kmax: int, sil_metric: str = "euclidean", parameter: Dict = {}, **kwargs) -> Figure:
    ks = data["k"].values
    sil = data["silhouette score"].values
    title = kwargs.get("title", "Silhouette Plot")
    f = plt.figure(**kwargs)
    plt.plot(ks, sil, ls="-", color="cyan", marker="o", linewidth=.5)
    max_index = sil.argmax()+1
    max_score = sil.max()
    plt.grid(color='k', linestyle='--', linewidth=.1)
    plt.gca().axvline(max_index, ls="--", color="g", linewidth=1)
    plt.gca().axhline(max_score, ls="--", color="g", linewidth=1)
    plt.gca().text(max_index, max_score, f"{max_index}", ha="center", va="center", color="red",
                   weight='extra bold')
    plt.xlabel("No. of clusters")
    plt.ylabel("Silhouette score")
    plt.title(title+f"\nscore={max_score:.2f}")
    return f


def dr_plot(
    df,
    columns_to_use,
    title=None,
    class_label_column=None,
    custom_order=None,
    label_function=lambda x:x,
    **params):
    fontsize_title = params.get("fontsize_title", 12)
    show_names = params.get("show_names", False)
    dpi = params.get("dpi", 100)
    fig_x = params.get("fig_x", 10)
    fig_y = params.get("fig_y", 10)
    x_suffix = params.get("x_suffix", "")
    y_suffix = params.get("y_suffix", "")
    f = plt.figure(figsize=(fig_x, fig_y), dpi=dpi)
    class_labels = class_label_column in df.columns
    labels = "labels" in df.columns
    if class_labels:
        if custom_order is not None:
            df["custom_order"] = [custom_order.find(label) for label in df[class_label_column].values]
            df = df.sort_values("custom_order")
    else:
        df[class_label_column] = [""]*len(df)
    if not labels:
        df['labels'] = df.index
    dimensions = params.get("dimension", len(df.columns))
    if dimensions < 2:
        raise ValueError(f"No 2D projection possible with only {dimensions} components, set k >= 2.")
    if len(columns_to_use) > 2:
        columns_to_use = columns_to_use[:2]
    custom_cycler = (cycler(color=["b", "g", "r", "c", "k", "m", "y", "grey", "darkblue", "darkgreen", "darkred", "darkcyan", "darkviolet", "gold", "slategrey", "purple"]) +
                cycler(marker=[".", 'o', 'v', '^', '*', 's', '<', '>', '+', 'o', 'v', '^', '*', 's', '<', '>'])
                )
    plt.gca().set_prop_cycle(custom_cycler)
    for i, df_sub in df.groupby(class_label_column):
        plt.plot(df_sub[columns_to_use[0]].values, df_sub[columns_to_use[1]].values, markersize=7, alpha=0.8, label=i, linestyle="None")
    if title is not None:
        plt.title(title, fontsize=fontsize_title)
    elif class_labels:
        plt.title('Transformed samples with classes', fontsize=fontsize_title)
    else:
        plt.title('Transformed samples without classes', fontsize=fontsize_title)
    xmin = df[columns_to_use[0]].values.min()
    ymin = df[columns_to_use[1]].values.min()
    xmax = df[columns_to_use[0]].values.max()
    ymax = df[columns_to_use[1]].values.max()
    plt.gca().set_xlim([1.3*xmin, 1.3*xmax])
    plt.gca().set_ylim([1.3*ymin, 1.3*ymax])
    plt.gca().set_xlabel(f"{columns_to_use[0]}{x_suffix}")
    plt.gca().set_ylabel(f"{columns_to_use[1]}{y_suffix}")
    if class_labels:
        plt.gca().legend(loc='best')
    if show_names:
        for i, row in df.iterrows():
            plt.annotate(
                label_function(row['labels']),
                xy=(row[columns_to_use[0]], row[columns_to_use[1]]), xytext=(-1, 1),
                textcoords='offset points', ha='right', va='bottom', size=8)
    return f

def plot_scree(model, name):
    eigenvalues = model.explained_variance_ratio_
    fig = plt.figure(figsize=(10, 10))
    plt.plot(np.arange(1, 1+len(eigenvalues)), eigenvalues, marker="o", ls="-")
    plt.title(f"Scree plot {name}")
    return fig

def save(f, fname, outdir = None):
    if outdir is None:
        outpath = Path("results") / "plots"
    else:
        outpath = outdir
    outpath.mkdir(exist_ok=True, parents=True)
    f.savefig(outpath / (fname+".png"))
    f.savefig(outpath / (fname+".pdf"))
    
def cluster(data, model):
    clustering = model.fit(data)
    return clustering

def plot_dendrogram(data, model, orientation="top", no_labels=False):
    if not isinstance(model.affinity, str):
        raise ValueError("Recalculation of pairwise distances requires a distance metric as string.")
    linkage_matrix = scipy.cluster.hierarchy.linkage(
        data, 
        method=model.linkage, 
        metric=model.affinity,
        optimal_ordering=True
    )
    
    pairwise = sklearn.metrics.pairwise_distances(
                data.values, 
                metric=model.affinity
            )
    order = scipy.cluster.hierarchy.leaves_list(linkage_matrix)
    ordered_labels = data.index.values[order]
    dendrogram(linkage_matrix, orientation=orientation, no_labels=no_labels)
    return ordered_labels

        
def plot_elbow(data, model_class, kmin, kmax, title="Inertia", parameter = None, **kwargs):
    if parameter is None:
        parameter = {}
    wss = []
    ks = np.arange(kmin, kmax+1)
    for k in ks:
        model = model_class(n_clusters=k, **parameter)
        model = model.fit(data)    
        if hasattr(model, "inertia_"):
            wss.append(model.inertia_)
        elif hasattr(model, "labels_"):
            wss.append(calculate_inertia(model, data))
        else:
            raise ValueError("Don't know how to interpret cluster model.")   
    f = plt.figure(**kwargs)
    plt.plot(ks, wss, ls="-", color ="b", marker="o")
    plt.xlabel("No. of clusters")
    plt.ylabel("Within-cluster Sum of Squares")
    plt.xticks(ks)
    plt.title(title)
    plt.grid(color='k', linestyle='--', linewidth=.1)
    return f
            
def plot_scree(model, title="Scree plot"):
    eigenvalues = model.explained_variance_ratio_*100
    fig = plt.figure(figsize=(10, 10))
    x = np.arange(1, 1+len(eigenvalues))
    y = eigenvalues
    plt.bar(x, y)
    plt.plot(x, y, marker="o", ls="-", color="k")
    for i, eigen in enumerate(eigenvalues):
        plt.text(x[i], y[i], f"{eigen:.1f}%", {"fontsize":14})
    plt.title(title)
    plt.xlabel("PC")
    plt.ylabel("Percentage of explained variance")
    return fig

def plot_silhouette(data, model_class, kmin, kmax, title="Silhouette", sil_metric="euclidean", parameter=None, **kwargs):
    if parameter is None:
        parameter = {}
    sil = []
    ks = np.arange(kmin, kmax+1)
    for k in ks:
        model = model_class(n_clusters=k, **parameter)
        model = model.fit(data)    
        sil.append(silhouette_score(model, data, sil_metric))
    sil = np.insert(np.array(sil), 0, 0)
    ks = np.insert(ks, 0, 1)
    f = plt.figure(**kwargs)
    plt.plot(ks, sil, ls="-", color ="cyan", marker="o", linewidth=.5)
    max_index = sil.argmax()+1
    max_score = sil.max()
    plt.grid(color='k', linestyle='--', linewidth=.1)
    plt.gca().axvline(max_index, ls = "--", color = "g", linewidth=1)
    plt.gca().axhline(max_score, ls = "--", color = "g", linewidth=1)
    plt.gca().text(max_index, max_score, f"{max_index}", ha="center", va="center", color="red", 
                   weight='extra bold')  # backgroundcolor="lightgrey"
    plt.xlabel("No. of clusters")
    plt.ylabel("Silhouette score")
    #plt.xticks(ks)
    plt.title(title+f"\nscore={max_score:.2f}")
    return f

def plot_full_silhouette(data, model_class, kmin, kmax, embedding = None, title="Silhouette plot", 
                         sil_metric="euclidean", parameter=None, 
                         **kwargs):
    if parameter is None:
        parameter = {}
        grid_columns = 2
    if embedding is None:
        data_embedded = data.copy()
        etitle = "first features"
    else:
        etitle = embedding.__class__.__name__
        emodel = embedding.fit_transform(data)
        data_embedded = pd.DataFrame(emodel, index=data.index)
    
    ks = np.arange(kmin, kmax+1)
    
    grid_rows = kmax-kmin+1
    f = plt.figure(constrained_layout=True, figsize=(5*grid_columns, 4*grid_rows))
    gs = f.add_gridspec(grid_rows, grid_columns)
    
    for ki, k in enumerate(ks):
        model = model_class(n_clusters=k, **parameter)
        model = model.fit(data)    
        projection = data_embedded.copy()
        projection["cluster"] = model.labels_    
        sil_samples = silhouette_samples(model, data, sil_metric)
        sil_score = silhouette_score(model, data, sil_metric)
        ax = f.add_subplot(gs[ki, 0])
        ax2 = f.add_subplot(gs[ki, 1])
        plt.sca(ax)
        data_clustered = data.copy()
        data_clustered["cluster"] = model.labels_
        data_clustered["sil"] = sil_samples
        yticks = []
        labels = []
        gap = len(data)//20
        y_lower = 0
        colors = {}
        for label, cluster in data_clustered.groupby("cluster"):
            y_upper = y_lower + len(cluster)
            cluster = cluster.sort_values("sil", ascending = True)
            fill = plt.gca().fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster["sil"].values, alpha=0.7)
            mid = y_lower + (y_upper-y_lower)/2
            yticks.append(mid)
            labels.append(label)
            colors[label] = fill.get_facecolor()
            y_lower = y_upper + gap    
            plt.sca(ax2)
            pcluster = projection[projection["cluster"] == label]
            plt.scatter(pcluster[pcluster.columns[0]], pcluster[pcluster.columns[1]], color=colors[label], marker=".")
            plt.sca(ax)
        plt.yticks(yticks, labels)
        plt.ylabel("Cluster")
        plt.xlabel("Silhouette coefficient")
        plt.gca().axvline(sil_score, c="r", ls="--")
        plt.grid(color='k', linestyle='--', linewidth=.1)
        plt.title("Silhouette")
        plt.sca(ax2)
        plt.ylabel(f"Dim {data_embedded.columns[0]}")
        plt.xlabel(f"Dim {data_embedded.columns[1]}")
        plt.title(etitle)
    plt.suptitle(title)
    return f


def calculate_gap(data, model_class, kmin, kmax, random_samples, seed = None, parameter=None, use_svd=False):
    if seed is not None:
        np.random.seed(seed)
    if parameter is None:
        parameter = {}
    sks = []
    wks = []
    wkbs = []
    bs = []
    gaps = []
    try:
        np.testing.assert_almost_equal(data.mean().values, np.zeros(len(data.columns)))
    except AssertionError:
        print(data.mean().values)
        print("The data for gap statistic needs to be centered to zero.")
        raise
    def __generate_random(data):
        if use_svd:
            U, s, Vh = scipy.linalg.svd(data.values, full_matrices=True)
            Xi = np.dot(data.values, Vh.transpose())
            Zi = np.random.uniform(Xi.min(axis=0), Xi.max(axis=0), size = data.shape)
            Z = np.dot(Zi, Vh)
        else:
            Z = np.random.uniform(data.min(), data.max(), size = data.shape)
        return pd.DataFrame(
            Z,
            columns=data.columns, 
            index=data.index
        )

    for b in range(random_samples):
        rb = __generate_random(data)
        assert rb.shape == data.shape
        bs.append(rb)
    ks = np.arange(kmin, kmax+1)
    for k in ks:
        # fit the model
        model = model_class(n_clusters=k, **parameter)
        model = model.fit(data)    
        # calculate within-cluster sum of squares
        wk = np.log(tibshirani(model, data))
        wks.append(wk)
        wkbs_for_k = []
        for b in bs:
            # fit the model for all uniformely random samples
            modelb = model_class(n_clusters=k, **parameter)
            modelb = modelb.fit(b)    
            wkbs_for_k.append(np.log(tibshirani(modelb, b)))   
        wkbs_for_k = np.array(wkbs_for_k)
        bmean = wkbs_for_k.mean()
        gap = bmean - wk
        gaps.append(gap)
        wkbs.append(bmean)
        sdk = np.sqrt(((wkbs_for_k - bmean)**2).sum() / random_samples)
        sdk = np.std(wkbs_for_k)
        sk = np.sqrt(1+1/random_samples)*sdk
        sks.append(sk)
    result = pd.DataFrame({"gap": gaps, "k": ks, "Wk": wks, "Wkb": wkbs, "sk": sks})
    crit = np.zeros(len(result), dtype=bool)
    first_local = np.zeros(len(result), dtype=bool)
    first_se = np.zeros(len(result), dtype=bool)
    for i in result.index.values[:-1]:
        if (not np.any(crit)) and (result.loc[i]["gap"] >= (result.loc[i+1]["gap"]-result.loc[i+1]["sk"])):
            crit[i] = True
        if (not np.any(first_local)) and (result.loc[i]["gap"] >= (result.loc[i+1]["gap"])):
            first_local[i] = True
            fse = i
            for j in result.index.values[i::-1]:
                if result.loc[j]["gap"] >= (result.loc[i]["gap"]-result.loc[i]["sk"]):
                    fse = j
            first_se[fse] = True
    result["tibshirani"] = crit
    result["local"] = first_local
    result["local_se"] = first_se
    return result


def plot_2d(data, embedding, label_column="label"):
    labels = np.ones(len(data))
    if label_column in data.columns:
        labels = data[label_column].values
        data = data.drop(label_column, axis=1)
    data_embedded = pd.DataFrame(
        embedding.fit_transform(data), 
        index=data.index
    )
    data_embedded["label"] = labels
    f = plt.figure()
    for label, cluster in data_embedded.groupby("label"):
        plt.scatter(cluster[cluster.columns[0]], cluster[cluster.columns[1]], marker=".")
    plt.ylabel(f"Dim {data_embedded.columns[0]}")
    plt.xlabel(f"Dim {data_embedded.columns[1]}")
    plt.title(f"{embedding.__class__.__name__} projection")
    return f

        
def plot_wks(data, model_class, kmin, kmax, random_samples, title="Gap statistic", parameter = None, seed=None, **kwargs):
    data = calculate_gap(data, model_class, kmin, kmax, random_samples, seed=seed, parameter=parameter)
    f = plt.figure(**kwargs)
    plt.plot(data.k, data.Wk, ls="-", color ="b", marker="o", label="Wk")
    plt.plot(data.k, data.Wkb, ls="-", color ="r", marker="o", label="Wkb")
    plt.xlabel("No. of clusters")
    plt.ylabel("Within-cluster Sum of Squares")
    plt.legend()
    plt.xticks(data.k)
    plt.title(title)
    plt.grid(color='k', linestyle='--', linewidth=.1)    
    return f

def plot_gap(data, model_class, kmin, kmax, random_samples, title="Gap statistic", selection=None, parameter = None, seed=None, use_svd=False, **kwargs):
    data = calculate_gap(data, model_class, kmin, kmax, random_samples, seed=seed, parameter=parameter, use_svd=use_svd)
    f = plt.figure(**kwargs)
    plt.plot(data.k, data.gap, ls="-", color ="b", marker="o")
    plt.errorbar(data.k, data.gap, yerr=data.sk, color = "b", linewidth=1, capsize=4)
    plt.xlabel("No. of clusters")
    plt.ylabel("Gap statistic")
    maxi = data.gap.argmax()
    plt.gca().axvline(data.loc[maxi]["k"], ls = "--", color = "r", linewidth=1)
    plt.gca().axhline(data.loc[maxi]["gap"], ls = "--", color = "r", linewidth=1)
    plt.plot(data.loc[maxi]["k"], data.loc[maxi]["gap"], color="r", label="global")
    colors = {
        "local": "purple",
        "local_se": "cyan",
        "tibshirani": "g",
    }
    if selection is None:
        selection = colors
    for label in selection:
        c = colors[label]
        best = data[data[label]]
        if len(best) == 1:
            plt.gca().axvline(best["k"].values[0], ls = "--", color = c, linewidth=1)
            plt.gca().axhline(best["gap"].values[0], ls = "--", color = c, linewidth=1)
            plt.plot(best["k"].values[0], best["gap"].values[0], color= c, label=label)
    plt.legend()
    plt.xticks(data.k)
    plt.title(title)
    plt.grid(color='k', linestyle='--', linewidth=.1)
    return f


def plot_pearson(genes, name, threshold_factor = .6, dpi = 100, fontsize = 6, fsize=40):
    f = plt.figure(figsize=(fsize*1.2, fsize))
    #f = plt.figure(constrained_layout=True, figsize=(grid_cols*colfactor, grid_rows*rowfactor))
    gs = f.add_gridspec(10, 12)
    axe_main = f.add_subplot(gs[:, :10])
    axe_den = f.add_subplot(gs[:, 10:11])    
    axe_col = f.add_subplot(gs[:, 11:12])    
    plt.sca(axe_main)
    corr = genes.corr()
    l = len(corr)
    distance = scipy.cluster.hierarchy.distance.pdist(corr.values)
    linkage = scipy.cluster.hierarchy.linkage(distance, method='single')
    plt.sca(axe_den)
    plt.sca(axe_main)
    ind = scipy.cluster.hierarchy.fcluster(linkage, threshold_factor*distance.max(), 'distance')
    columns = [corr.columns.tolist()[i] for i in list((np.argsort(ind)))]
    #corr = corr.reindex(index=columns, columns=columns)
    no_of_clusters = len(set(ind))
    if  2 <= no_of_clusters < len(columns):
        sil = sklearn.metrics.silhouette_samples(
            corr.values, 
            ind,
            metric='euclidean'
        )
        silscore = sklearn.metrics.silhouette_score(corr.values, ind, metric='euclidean', random_state=12)
        yy = [f"{x} ({y:.2f})" for x, y in zip(genes.columns.values, sil)]
    else:
        silscore = np.nan
        yy = genes.columns.values
    plt.xticks(np.arange(len(corr.columns)), ind, rotation=0, fontsize=fontsize)
    plt.yticks(
        np.arange(len(genes.columns)), 
        yy, 
        fontsize=fontsize)
    im = plt.imshow(corr, cmap="seismic", aspect="auto") #norm=matplotlib.colors.Normalize(vmin=-1, vmax=1))
    for i in range(len(corr.columns)):
        for j in range(len(corr.index)):
            text = plt.gca().text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", color="grey")
    cb = plt.colorbar(im, cax=axe_col, shrink=.9, fraction=.01, pad = 5)
    corr["Cluster"] = ind
    plt.title(f"Pearson Correlation {name}, Silhouette score = {silscore:.2f}")
    axe_den.tick_params(labelbottom=False, labelleft=False, length=0)
    axe_den.spines["top"].set_visible(False)
    axe_den.spines["bottom"].set_visible(False)
    axe_den.spines["right"].set_visible(False)
    axe_den.spines["left"].set_visible(False)
    return f, corr