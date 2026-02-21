"""
Módulo de Avaliação para Aprendizado Não Supervisionado.

Este módulo contém funções para avaliar modelos de clustering e 
redução de dimensionalidade, incluindo métricas internas e externas.

Autor: Tech Challenge Grupo
Data: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    fowlkes_mallows_score
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# MÉTRICAS INTERNAS DE CLUSTERING
# =============================================================================

def evaluate_clustering_internal(
    X: np.ndarray,
    labels: np.ndarray,
    include_samples: bool = False
) -> Dict[str, float]:
    """
    Calcula métricas internas de avaliação de clustering.
    
    Métricas internas não requerem labels verdadeiros (ground truth).
    
    Parameters
    ----------
    X : np.ndarray
        Dados utilizados no clustering.
    labels : np.ndarray
        Labels dos clusters atribuídos.
    include_samples : bool
        Se True, inclui silhouette por amostra.
        
    Returns
    -------
    Dict[str, float]
        Dicionário com as métricas calculadas.
    """
    # Remove amostras com label -1 (ruído do DBSCAN)
    mask = labels != -1
    X_valid = X[mask]
    labels_valid = labels[mask]
    
    # Verifica se há clusters suficientes
    n_clusters = len(np.unique(labels_valid))
    
    if n_clusters < 2:
        return {
            'silhouette_score': np.nan,
            'davies_bouldin_score': np.nan,
            'calinski_harabasz_score': np.nan,
            'n_clusters': n_clusters,
            'n_noise_points': np.sum(labels == -1)
        }
    
    metrics = {
        'silhouette_score': silhouette_score(X_valid, labels_valid),
        'davies_bouldin_score': davies_bouldin_score(X_valid, labels_valid),
        'calinski_harabasz_score': calinski_harabasz_score(X_valid, labels_valid),
        'n_clusters': n_clusters,
        'n_noise_points': np.sum(labels == -1)
    }
    
    if include_samples:
        metrics['silhouette_samples'] = silhouette_samples(X_valid, labels_valid)
    
    return metrics


def evaluate_clustering_stability(
    X: np.ndarray,
    clustering_func,
    n_iterations: int = 10,
    sample_size: float = 0.8,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Avalia a estabilidade do clustering através de bootstrap.
    
    Parameters
    ----------
    X : np.ndarray
        Dados para clustering.
    clustering_func : callable
        Função que recebe X e retorna labels.
    n_iterations : int
        Número de iterações de bootstrap.
    sample_size : float
        Proporção de amostras por iteração.
    random_state : int
        Seed para reprodutibilidade.
        
    Returns
    -------
    Dict[str, float]
        Estatísticas de estabilidade.
    """
    np.random.seed(random_state)
    n_samples = len(X)
    sample_n = int(n_samples * sample_size)
    
    silhouette_scores = []
    n_clusters_list = []
    
    for i in range(n_iterations):
        # Amostra bootstrap
        indices = np.random.choice(n_samples, sample_n, replace=False)
        X_sample = X[indices]
        
        # Aplica clustering
        labels = clustering_func(X_sample)
        
        # Calcula métricas
        n_clusters = len(np.unique(labels[labels != -1]))
        n_clusters_list.append(n_clusters)
        
        if n_clusters >= 2:
            mask = labels != -1
            if np.sum(mask) > n_clusters:
                score = silhouette_score(X_sample[mask], labels[mask])
                silhouette_scores.append(score)
    
    return {
        'silhouette_mean': np.mean(silhouette_scores) if silhouette_scores else np.nan,
        'silhouette_std': np.std(silhouette_scores) if silhouette_scores else np.nan,
        'n_clusters_mean': np.mean(n_clusters_list),
        'n_clusters_std': np.std(n_clusters_list),
        'n_iterations': n_iterations
    }


# =============================================================================
# MÉTRICAS EXTERNAS DE CLUSTERING
# =============================================================================

def evaluate_clustering_external(
    labels_true: np.ndarray,
    labels_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calcula métricas externas de avaliação de clustering.
    
    Métricas externas requerem labels verdadeiros (ground truth).
    
    Parameters
    ----------
    labels_true : np.ndarray
        Labels verdadeiros (ground truth).
    labels_pred : np.ndarray
        Labels preditos pelo clustering.
        
    Returns
    -------
    Dict[str, float]
        Dicionário com as métricas calculadas.
    """
    # Converte labels não numéricos se necessário
    if labels_true.dtype == object:
        le = LabelEncoder()
        labels_true = le.fit_transform(labels_true)
    
    # Remove pontos de ruído
    mask = labels_pred != -1
    labels_true_valid = labels_true[mask]
    labels_pred_valid = labels_pred[mask]
    
    return {
        'adjusted_rand_score': adjusted_rand_score(labels_true_valid, labels_pred_valid),
        'normalized_mutual_info': normalized_mutual_info_score(labels_true_valid, labels_pred_valid),
        'homogeneity_score': homogeneity_score(labels_true_valid, labels_pred_valid),
        'completeness_score': completeness_score(labels_true_valid, labels_pred_valid),
        'v_measure_score': v_measure_score(labels_true_valid, labels_pred_valid),
        'fowlkes_mallows_score': fowlkes_mallows_score(labels_true_valid, labels_pred_valid)
    }


# =============================================================================
# ANÁLISE DE CLUSTERS
# =============================================================================

def analyze_cluster_composition(
    df: pd.DataFrame,
    cluster_column: str,
    analysis_columns: List[str],
    agg_functions: Dict[str, Union[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Analisa a composição de cada cluster.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com os dados e labels de cluster.
    cluster_column : str
        Nome da coluna com os labels de cluster.
    analysis_columns : List[str]
        Colunas para analisar.
    agg_functions : Dict[str, Union[str, List[str]]]
        Funções de agregação por coluna.
        
    Returns
    -------
    pd.DataFrame
        Estatísticas por cluster.
    """
    if agg_functions is None:
        agg_functions = {col: ['mean', 'std', 'min', 'max'] for col in analysis_columns}
    
    # Filtra apenas colunas numéricas presentes no DataFrame
    valid_columns = [col for col in analysis_columns if col in df.columns]
    valid_agg = {col: agg_functions.get(col, ['mean', 'std']) for col in valid_columns}
    
    stats = df.groupby(cluster_column)[valid_columns].agg(valid_agg)
    
    # Adiciona contagem por cluster
    counts = df.groupby(cluster_column).size()
    stats[('count', '')] = counts
    stats[('percentage', '')] = (counts / len(df) * 100).round(2)
    
    return stats


def get_cluster_centroids(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str] = None
) -> pd.DataFrame:
    """
    Calcula os centróides de cada cluster.
    
    Parameters
    ----------
    X : np.ndarray
        Dados utilizados no clustering.
    labels : np.ndarray
        Labels dos clusters.
    feature_names : List[str]
        Nomes das features.
        
    Returns
    -------
    pd.DataFrame
        DataFrame com os centróides.
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]  # Remove ruído
    
    centroids = []
    for label in unique_labels:
        mask = labels == label
        centroid = X[mask].mean(axis=0)
        centroids.append(centroid)
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    df_centroids = pd.DataFrame(
        centroids,
        index=[f'Cluster_{l}' for l in unique_labels],
        columns=feature_names
    )
    
    return df_centroids


def get_cluster_sizes(labels: np.ndarray) -> pd.DataFrame:
    """
    Retorna o tamanho de cada cluster.
    
    Parameters
    ----------
    labels : np.ndarray
        Labels dos clusters.
        
    Returns
    -------
    pd.DataFrame
        DataFrame com tamanhos dos clusters.
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    df_sizes = pd.DataFrame({
        'cluster': unique,
        'count': counts,
        'percentage': (counts / len(labels) * 100).round(2)
    })
    
    df_sizes['is_noise'] = df_sizes['cluster'] == -1
    df_sizes = df_sizes.sort_values('count', ascending=False)
    
    return df_sizes


# =============================================================================
# AVALIAÇÃO DE REDUÇÃO DE DIMENSIONALIDADE
# =============================================================================

def evaluate_dimensionality_reduction(
    X_original: np.ndarray,
    X_reduced: np.ndarray,
    method: str = 'pca'
) -> Dict[str, float]:
    """
    Avalia a qualidade da redução de dimensionalidade.
    
    Parameters
    ----------
    X_original : np.ndarray
        Dados originais.
    X_reduced : np.ndarray
        Dados reduzidos.
    method : str
        Método utilizado ('pca', 'tsne', etc.).
        
    Returns
    -------
    Dict[str, float]
        Métricas de avaliação.
    """
    from sklearn.metrics import pairwise_distances
    from scipy.stats import spearmanr
    
    # Amostra para cálculo eficiente (se necessário)
    n_samples = min(5000, len(X_original))
    if len(X_original) > n_samples:
        np.random.seed(42)
        indices = np.random.choice(len(X_original), n_samples, replace=False)
        X_orig_sample = X_original[indices]
        X_red_sample = X_reduced[indices]
    else:
        X_orig_sample = X_original
        X_red_sample = X_reduced
    
    # Calcula distâncias
    dist_original = pairwise_distances(X_orig_sample).flatten()
    dist_reduced = pairwise_distances(X_red_sample).flatten()
    
    # Correlação de Spearman (preservação de ranking de distâncias)
    correlation, p_value = spearmanr(dist_original, dist_reduced)
    
    # Trustworthiness (vizinhos preservados)
    from sklearn.manifold import trustworthiness
    trust = trustworthiness(X_orig_sample, X_red_sample, n_neighbors=10)
    
    metrics = {
        'spearman_correlation': correlation,
        'p_value': p_value,
        'trustworthiness': trust,
        'n_original_features': X_original.shape[1],
        'n_reduced_features': X_reduced.shape[1],
        'reduction_ratio': X_reduced.shape[1] / X_original.shape[1]
    }
    
    return metrics


def evaluate_pca_reconstruction(
    X_original: np.ndarray,
    pca_model,
    X_transformed: np.ndarray = None
) -> Dict[str, float]:
    """
    Avalia a qualidade da reconstrução do PCA.
    
    Parameters
    ----------
    X_original : np.ndarray
        Dados originais (normalizados).
    pca_model : sklearn.decomposition.PCA
        Modelo PCA ajustado.
    X_transformed : np.ndarray
        Dados transformados (opcional).
        
    Returns
    -------
    Dict[str, float]
        Métricas de reconstrução.
    """
    if X_transformed is None:
        X_transformed = pca_model.transform(X_original)
    
    # Reconstrói os dados
    X_reconstructed = pca_model.inverse_transform(X_transformed)
    
    # Calcula erro de reconstrução
    mse = np.mean((X_original - X_reconstructed) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(X_original - X_reconstructed))
    
    # R² da reconstrução
    ss_res = np.sum((X_original - X_reconstructed) ** 2)
    ss_tot = np.sum((X_original - np.mean(X_original)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_reconstruction': r2,
        'total_variance_explained': np.sum(pca_model.explained_variance_ratio_),
        'n_components': pca_model.n_components_
    }


# =============================================================================
# VISUALIZAÇÕES DE AVALIAÇÃO
# =============================================================================

def plot_silhouette_analysis(
    X: np.ndarray,
    labels: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    save_path: str = None
) -> plt.Figure:
    """
    Plota análise de silhouette por cluster.
    
    Parameters
    ----------
    X : np.ndarray
        Dados utilizados no clustering.
    labels : np.ndarray
        Labels dos clusters.
    figsize : Tuple[int, int]
        Tamanho da figura.
    save_path : str
        Caminho para salvar a figura.
        
    Returns
    -------
    plt.Figure
        Figura matplotlib.
    """
    # Remove ruído
    mask = labels != -1
    X_valid = X[mask]
    labels_valid = labels[mask]
    
    n_clusters = len(np.unique(labels_valid))
    
    if n_clusters < 2:
        print("Clusters insuficientes para análise de silhouette.")
        return None
    
    # Calcula silhouette
    silhouette_avg = silhouette_score(X_valid, labels_valid)
    sample_silhouette_values = silhouette_samples(X_valid, labels_valid)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_lower = 10
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    
    for i, cluster in enumerate(sorted(np.unique(labels_valid))):
        cluster_silhouette_values = sample_silhouette_values[labels_valid == cluster]
        cluster_silhouette_values.sort()
        
        size_cluster = len(cluster_silhouette_values)
        y_upper = y_lower + size_cluster
        
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouette_values,
            facecolor=colors[i],
            edgecolor=colors[i],
            alpha=0.7
        )
        
        ax.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster))
        y_lower = y_upper + 10
    
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
               label=f'Média: {silhouette_avg:.3f}')
    
    ax.set_xlabel('Coeficiente de Silhouette')
    ax.set_ylabel('Cluster')
    ax.set_title('Análise de Silhouette por Cluster')
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_cluster_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: List[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: str = None
) -> plt.Figure:
    """
    Compara métricas de diferentes configurações de clustering.
    
    Parameters
    ----------
    metrics_dict : Dict[str, Dict[str, float]]
        Dicionário com métricas por configuração.
        Ex: {'K=3': {'silhouette': 0.5, ...}, 'K=4': {...}}
    metric_names : List[str]
        Métricas a serem plotadas.
    figsize : Tuple[int, int]
        Tamanho da figura.
    save_path : str
        Caminho para salvar a figura.
        
    Returns
    -------
    plt.Figure
        Figura matplotlib.
    """
    if metric_names is None:
        # Pega todas as métricas numéricas do primeiro item
        first_metrics = list(metrics_dict.values())[0]
        metric_names = [k for k, v in first_metrics.items() 
                       if isinstance(v, (int, float)) and not np.isnan(v)]
    
    df_metrics = pd.DataFrame(metrics_dict).T
    df_metrics = df_metrics[metric_names]
    
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics_dict)))
    
    for i, metric in enumerate(metric_names):
        values = df_metrics[metric].values
        configs = df_metrics.index.tolist()
        
        bars = axes[i].bar(configs, values, color=colors)
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].set_xlabel('Configuração')
        axes[i].tick_params(axis='x', rotation=45)
        
        # Adiciona valores nas barras
        for bar, val in zip(bars, values):
            axes[i].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
    
    plt.suptitle('Comparação de Métricas de Clustering', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_evaluation_summary(
    internal_metrics: Dict[str, float],
    external_metrics: Dict[str, float] = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: str = None
) -> plt.Figure:
    """
    Plota um resumo visual das métricas de avaliação.
    
    Parameters
    ----------
    internal_metrics : Dict[str, float]
        Métricas internas de clustering.
    external_metrics : Dict[str, float]
        Métricas externas de clustering (opcional).
    figsize : Tuple[int, int]
        Tamanho da figura.
    save_path : str
        Caminho para salvar a figura.
        
    Returns
    -------
    plt.Figure
        Figura matplotlib.
    """
    n_plots = 2 if external_metrics else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    # Métricas internas
    internal_to_plot = {k: v for k, v in internal_metrics.items() 
                       if isinstance(v, float) and not np.isnan(v) 
                       and k not in ['n_clusters', 'n_noise_points']}
    
    names = list(internal_to_plot.keys())
    values = list(internal_to_plot.values())
    
    colors = ['green' if 'silhouette' in n else 
              'orange' if 'davies' in n else 
              'blue' for n in names]
    
    bars = axes[0].barh(names, values, color=colors, alpha=0.7)
    axes[0].set_title('Métricas Internas de Clustering')
    axes[0].set_xlabel('Valor')
    
    # Adiciona valores nas barras
    for bar, val in zip(bars, values):
        axes[0].text(
            val,
            bar.get_y() + bar.get_height()/2,
            f' {val:.3f}',
            va='center',
            fontsize=10
        )
    
    # Métricas externas (se disponíveis)
    if external_metrics and n_plots > 1:
        ext_names = list(external_metrics.keys())
        ext_values = list(external_metrics.values())
        
        bars = axes[1].barh(ext_names, ext_values, color='purple', alpha=0.7)
        axes[1].set_title('Métricas Externas de Clustering')
        axes[1].set_xlabel('Valor')
        axes[1].set_xlim(0, 1)
        
        for bar, val in zip(bars, ext_values):
            axes[1].text(
                val,
                bar.get_y() + bar.get_height()/2,
                f' {val:.3f}',
                va='center',
                fontsize=10
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# RELATÓRIO DE AVALIAÇÃO
# =============================================================================

def generate_clustering_report(
    X: np.ndarray,
    labels: np.ndarray,
    labels_true: np.ndarray = None,
    feature_names: List[str] = None,
    model_name: str = "Clustering"
) -> str:
    """
    Gera um relatório textual completo da avaliação de clustering.
    
    Parameters
    ----------
    X : np.ndarray
        Dados utilizados no clustering.
    labels : np.ndarray
        Labels preditos.
    labels_true : np.ndarray
        Labels verdadeiros (opcional).
    feature_names : List[str]
        Nomes das features.
    model_name : str
        Nome do modelo.
        
    Returns
    -------
    str
        Relatório em formato texto.
    """
    report = []
    report.append("=" * 60)
    report.append(f"RELATÓRIO DE AVALIAÇÃO - {model_name}")
    report.append("=" * 60)
    report.append("")
    
    # Informações básicas
    n_clusters = len(np.unique(labels[labels != -1]))
    n_noise = np.sum(labels == -1)
    
    report.append("INFORMAÇÕES GERAIS")
    report.append("-" * 30)
    report.append(f"Número de amostras: {len(labels)}")
    report.append(f"Número de features: {X.shape[1]}")
    report.append(f"Número de clusters: {n_clusters}")
    report.append(f"Pontos de ruído: {n_noise} ({n_noise/len(labels)*100:.2f}%)")
    report.append("")
    
    # Métricas internas
    internal = evaluate_clustering_internal(X, labels)
    report.append("MÉTRICAS INTERNAS")
    report.append("-" * 30)
    report.append(f"Silhouette Score: {internal['silhouette_score']:.4f}")
    report.append(f"Davies-Bouldin Index: {internal['davies_bouldin_score']:.4f}")
    report.append(f"Calinski-Harabasz Index: {internal['calinski_harabasz_score']:.2f}")
    report.append("")
    
    # Interpretação das métricas
    report.append("INTERPRETAÇÃO")
    report.append("-" * 30)
    
    sil = internal['silhouette_score']
    if not np.isnan(sil):
        if sil > 0.7:
            report.append("Silhouette: Excelente separação entre clusters")
        elif sil > 0.5:
            report.append("Silhouette: Boa separação entre clusters")
        elif sil > 0.25:
            report.append("Silhouette: Separação moderada entre clusters")
        else:
            report.append("Silhouette: Clusters sobrepostos ou mal definidos")
    
    dbi = internal['davies_bouldin_score']
    if not np.isnan(dbi):
        if dbi < 0.5:
            report.append("Davies-Bouldin: Excelente (clusters compactos e separados)")
        elif dbi < 1.0:
            report.append("Davies-Bouldin: Bom (boa estrutura de clusters)")
        else:
            report.append("Davies-Bouldin: Moderado (pode haver sobreposição)")
    
    report.append("")
    
    # Métricas externas (se disponíveis)
    if labels_true is not None:
        external = evaluate_clustering_external(labels_true, labels)
        report.append("MÉTRICAS EXTERNAS")
        report.append("-" * 30)
        report.append(f"Adjusted Rand Index: {external['adjusted_rand_score']:.4f}")
        report.append(f"Normalized Mutual Info: {external['normalized_mutual_info']:.4f}")
        report.append(f"V-Measure: {external['v_measure_score']:.4f}")
        report.append(f"Fowlkes-Mallows: {external['fowlkes_mallows_score']:.4f}")
        report.append("")
    
    # Tamanho dos clusters
    sizes = get_cluster_sizes(labels)
    report.append("DISTRIBUIÇÃO DOS CLUSTERS")
    report.append("-" * 30)
    for _, row in sizes.iterrows():
        label = "Ruído" if row['is_noise'] else f"Cluster {int(row['cluster'])}"
        report.append(f"{label}: {int(row['count'])} amostras ({row['percentage']:.2f}%)")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)


def create_evaluation_dataframe(
    experiments: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> pd.DataFrame:
    """
    Cria um DataFrame comparativo de múltiplos experimentos de clustering.
    
    Parameters
    ----------
    experiments : Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dicionário com nome do experimento -> (X, labels).
        
    Returns
    -------
    pd.DataFrame
        DataFrame com métricas de todos os experimentos.
    """
    results = []
    
    for name, (X, labels) in experiments.items():
        metrics = evaluate_clustering_internal(X, labels)
        metrics['experiment'] = name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df.set_index('experiment')
    
    # Ordena por silhouette score
    df = df.sort_values('silhouette_score', ascending=False)
    
    return df
