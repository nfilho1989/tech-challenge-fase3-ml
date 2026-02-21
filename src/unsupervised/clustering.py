"""
Módulo de Clusterização
=======================
Funções para análise de clusters usando K-Means, DBSCAN e Hierarchical Clustering.

Tech Challenge Fase 3 - Machine Learning Engineering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional, Union

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage as scipy_linkage
import joblib

# Configurações de visualização
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#2c3e50']


def scale_features(X: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, StandardScaler]:
    """
    Escala as features usando StandardScaler.
    
    Parameters
    ----------
    X : pd.DataFrame
        DataFrame com as features a serem escaladas
    scaler : StandardScaler, optional
        Scaler já ajustado. Se None, um novo será criado e ajustado.
        
    Returns
    -------
    X_scaled : np.ndarray
        Features escaladas
    scaler : StandardScaler
        Scaler utilizado (para reutilização)
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler


def find_optimal_k(X: np.ndarray, k_range: range = range(2, 11), random_state: int = 42) -> Dict:
    """
    Encontra o número ótimo de clusters usando Elbow Method e Silhouette Score.
    
    Parameters
    ----------
    X : np.ndarray
        Dados para clusterização (já escalados)
    k_range : range
        Range de valores de K a testar
    random_state : int
        Seed para reprodutibilidade
        
    Returns
    -------
    results : dict
        Dicionário com k, inertias, silhouette scores e k ótimo sugerido
        Formato compatível com pd.DataFrame()
    """
    results = {
        'k': [],
        'inertia': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': []
    }
    
    print(f'🔍 Buscando K ótimo de {min(k_range)} a {max(k_range)}...')
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        
        results['k'].append(k)
        results['inertia'].append(kmeans.inertia_)
        results['silhouette'].append(silhouette_score(X, labels))
        results['davies_bouldin'].append(davies_bouldin_score(X, labels))
        results['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
        
        print(f'   K={k}: Silhouette={results["silhouette"][-1]:.4f}, Inertia={results["inertia"][-1]:.0f}')
    
    # K ótimo sugerido pelo melhor silhouette score
    optimal_k = results['k'][np.argmax(results['silhouette'])]
    results['optimal_k'] = optimal_k
    
    print(f'\n✅ K ótimo sugerido: {optimal_k} (melhor Silhouette Score)')
    
    # Plota curvas de avaliação
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Elbow Curve (Inertia)
    axes[0].plot(results['k'], results['inertia'], 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'K ótimo = {optimal_k}')
    axes[0].set_xlabel('Número de Clusters (K)')
    axes[0].set_ylabel('Inertia (SSE)')
    axes[0].set_title('Método do Cotovelo (Elbow)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Silhouette Score
    axes[1].plot(results['k'], results['silhouette'], 'go-', linewidth=2, markersize=8)
    axes[1].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'K ótimo = {optimal_k}')
    axes[1].set_xlabel('Número de Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Davies-Bouldin (menor é melhor)
    axes[2].plot(results['k'], results['davies_bouldin'], 'ro-', linewidth=2, markersize=8)
    axes[2].axvline(x=optimal_k, color='blue', linestyle='--', alpha=0.7, label=f'K ótimo = {optimal_k}')
    axes[2].set_xlabel('Número de Clusters (K)')
    axes[2].set_ylabel('Davies-Bouldin Index')
    axes[2].set_title('Davies-Bouldin (menor = melhor)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../reports/figures/unsupervised/kmeans_optimal_k.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results


def plot_elbow_curve(results: Dict, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plota a curva Elbow com Inertia e Silhouette Score.
    
    Parameters
    ----------
    results : dict
        Resultados de find_optimal_k()
    save_path : str, optional
        Caminho para salvar a figura
        
    Returns
    -------
    fig : plt.Figure
        Figura do matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow Curve (Inertia)
    ax1 = axes[0]
    ax1.plot(results['k_range'], results['inertias'], 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Número de Clusters (K)', fontsize=12)
    ax1.set_ylabel('Inertia (SSE)', fontsize=12)
    ax1.set_title('Elbow Method - Inertia', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette Score
    ax2 = axes[1]
    ax2.plot(results['k_range'], results['silhouette_scores'], 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=results['optimal_k'], color='red', linestyle='--', 
                label=f'K ótimo = {results["optimal_k"]}')
    ax2.set_xlabel('Número de Clusters (K)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score por K', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✅ Figura salva em: {save_path}')
    
    return fig


def fit_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> Tuple[KMeans, np.ndarray]:
    """
    Ajusta modelo K-Means.
    
    Parameters
    ----------
    X : np.ndarray
        Dados para clusterização (já escalados)
    n_clusters : int
        Número de clusters
    random_state : int
        Seed para reprodutibilidade
        
    Returns
    -------
    model : KMeans
        Modelo ajustado
    labels : np.ndarray
        Labels dos clusters
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    
    return model, labels


def fit_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> Tuple[DBSCAN, np.ndarray]:
    """
    Ajusta modelo DBSCAN.
    
    Parameters
    ----------
    X : np.ndarray
        Dados para clusterização (já escalados)
    eps : float
        Raio máximo de vizinhança
    min_samples : int
        Número mínimo de amostras em uma vizinhança
        
    Returns
    -------
    model : DBSCAN
        Modelo ajustado
    labels : np.ndarray
        Labels dos clusters (-1 para outliers)
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    
    return model, labels


def fit_hierarchical(X: np.ndarray, n_clusters: int, linkage: str = 'ward') -> Tuple[AgglomerativeClustering, np.ndarray]:
    """
    Ajusta modelo de Clustering Hierárquico.
    
    Parameters
    ----------
    X : np.ndarray
        Dados para clusterização (já escalados)
    n_clusters : int
        Número de clusters
    linkage : str
        Método de ligação ('ward', 'complete', 'average', 'single')
        
    Returns
    -------
    model : AgglomerativeClustering
        Modelo ajustado
    labels : np.ndarray
        Labels dos clusters
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    
    return model, labels


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Avalia a qualidade dos clusters usando múltiplas métricas.
    
    Parameters
    ----------
    X : np.ndarray
        Dados utilizados na clusterização
    labels : np.ndarray
        Labels dos clusters
        
    Returns
    -------
    metrics : dict
        Dicionário com as métricas de avaliação
    """
    # Remover outliers (label = -1) para cálculo de métricas se existirem
    mask = labels != -1
    
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return {
            'silhouette_score': None,
            'davies_bouldin_score': None,
            'calinski_harabasz_score': None,
            'n_clusters': len(np.unique(labels[labels != -1])),
            'n_outliers': (labels == -1).sum()
        }
    
    X_clean = X[mask]
    labels_clean = labels[mask]
    
    metrics = {
        'silhouette_score': silhouette_score(X_clean, labels_clean),
        'davies_bouldin_score': davies_bouldin_score(X_clean, labels_clean),
        'calinski_harabasz_score': calinski_harabasz_score(X_clean, labels_clean),
        'n_clusters': len(np.unique(labels_clean)),
        'n_outliers': (labels == -1).sum()
    }
    
    return metrics


def plot_silhouette_analysis(X: np.ndarray, labels: np.ndarray, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plota análise de silhouette por cluster.
    
    Parameters
    ----------
    X : np.ndarray
        Dados utilizados na clusterização
    labels : np.ndarray
        Labels dos clusters
    save_path : str, optional
        Caminho para salvar a figura
        
    Returns
    -------
    fig : plt.Figure
        Figura do matplotlib
    """
    # Filtrar outliers se existirem
    mask = labels != -1
    X_clean = X[mask]
    labels_clean = labels[mask]
    
    n_clusters = len(np.unique(labels_clean))
    silhouette_avg = silhouette_score(X_clean, labels_clean)
    sample_silhouette_values = silhouette_samples(X_clean, labels_clean)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_lower = 10
    for i in range(n_clusters):
        cluster_silhouette_values = sample_silhouette_values[labels_clean == i]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
               label=f'Silhouette médio = {silhouette_avg:.3f}')
    ax.set_xlabel('Silhouette Coefficient', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    ax.set_title('Análise de Silhouette por Cluster', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✅ Figura salva em: {save_path}')
    
    return fig


def get_cluster_profiles(df: pd.DataFrame, labels: np.ndarray, features: List[str]) -> pd.DataFrame:
    """
    Gera perfil estatístico de cada cluster.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original com os dados
    labels : np.ndarray
        Labels dos clusters
    features : list
        Lista de features para análise
        
    Returns
    -------
    profiles : pd.DataFrame
        DataFrame com estatísticas por cluster
    """
    df_cluster = df.copy()
    df_cluster['Cluster'] = labels
    
    # Filtrar outliers
    df_cluster = df_cluster[df_cluster['Cluster'] != -1]
    
    # Estatísticas por cluster
    profiles = df_cluster.groupby('Cluster')[features].agg(['mean', 'median', 'std', 'count'])
    
    # Contagem de elementos por cluster
    cluster_counts = df_cluster['Cluster'].value_counts().sort_index()
    
    return profiles, cluster_counts


def plot_cluster_profiles(df: pd.DataFrame, labels: np.ndarray, features: List[str],
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plota perfil de cada cluster para as features selecionadas.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original com os dados
    labels : np.ndarray
        Labels dos clusters
    features : list
        Lista de features para visualização
    save_path : str, optional
        Caminho para salvar a figura
        
    Returns
    -------
    fig : plt.Figure
        Figura do matplotlib
    """
    df_cluster = df[features].copy()
    df_cluster['Cluster'] = labels
    df_cluster = df_cluster[df_cluster['Cluster'] != -1]
    
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = np.atleast_1d(axes).flatten()
    
    for i, feature in enumerate(features):
        ax = axes[i]
        df_cluster.boxplot(column=feature, by='Cluster', ax=ax)
        ax.set_title(feature, fontsize=12, fontweight='bold')
        ax.set_xlabel('Cluster')
        ax.set_ylabel(feature)
    
    # Remover axes vazios
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle('Perfil dos Clusters por Feature', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✅ Figura salva em: {save_path}')
    
    return fig


def plot_clusters_2d(X_2d: np.ndarray, labels: np.ndarray, 
                     title: str = 'Visualização dos Clusters',
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plota clusters em 2D.
    
    Parameters
    ----------
    X_2d : np.ndarray
        Dados em 2D (após PCA ou t-SNE)
    labels : np.ndarray
        Labels dos clusters
    title : str
        Título do gráfico
    save_path : str, optional
        Caminho para salvar a figura
        
    Returns
    -------
    fig : plt.Figure
        Figura do matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        mask = labels == label
        if label == -1:
            # Outliers em cinza
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c='gray', marker='x', 
                       s=50, alpha=0.5, label='Outliers')
        else:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       c=COLORS[label % len(COLORS)], 
                       s=50, alpha=0.6, label=f'Cluster {label}')
    
    ax.set_xlabel('Componente 1', fontsize=12)
    ax.set_ylabel('Componente 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✅ Figura salva em: {save_path}')
    
    return fig


def save_model(model, filepath: str) -> None:
    """
    Salva modelo usando joblib.
    
    Parameters
    ----------
    model : object
        Modelo a ser salvo
    filepath : str
        Caminho do arquivo
    """
    joblib.dump(model, filepath)
    print(f'✅ Modelo salvo em: {filepath}')


def load_model(filepath: str):
    """
    Carrega modelo usando joblib.
    
    Parameters
    ----------
    filepath : str
        Caminho do arquivo
        
    Returns
    -------
    model : object
        Modelo carregado
    """
    model = joblib.load(filepath)
    print(f'✅ Modelo carregado de: {filepath}')
    return model


# =============================================================================
# FUNÇÕES ADICIONAIS E ALIASES PARA COMPATIBILIDADE
# =============================================================================

def prepare_data_for_clustering(
    df: pd.DataFrame,
    features: List[str],
    method: str = 'standard'
) -> Tuple[np.ndarray, StandardScaler, List[str]]:
    """
    Prepara dados para clustering: seleciona features e normaliza.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com os dados
    features : List[str]
        Lista de features a serem utilizadas
    method : str
        Método de normalização ('standard' ou 'minmax')
        
    Returns
    -------
    X_scaled : np.ndarray
        Dados normalizados
    scaler : StandardScaler
        Scaler utilizado    feature_names : List[str]
        Nomes das features
    """
    X = df[features].values
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler, features


def apply_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42
) -> Tuple[np.ndarray, KMeans, Dict]:
    """
    Aplica K-Means e retorna labels, modelo e métricas.
    
    Parameters
    ----------
    X : np.ndarray
        Dados normalizados
    n_clusters : int
        Número de clusters
    random_state : int
        Seed para reprodutibilidade
        
    Returns
    -------
    labels : np.ndarray
        Labels dos clusters
    model : KMeans
        Modelo ajustado
    metrics : Dict
        Métricas de avaliação
    """
    model, labels = fit_kmeans(X, n_clusters, random_state)
    metrics = evaluate_clustering(X, labels)
    metrics['inertia'] = model.inertia_
    
    return labels, model, metrics


def apply_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5
) -> Tuple[np.ndarray, DBSCAN, Dict]:
    """
    Aplica DBSCAN e retorna labels, modelo e métricas.
    
    Parameters
    ----------
    X : np.ndarray
        Dados normalizados
    eps : float
        Raio máximo de vizinhança
    min_samples : int
        Número mínimo de amostras
        
    Returns
    -------
    labels : np.ndarray
        Labels dos clusters (-1 para ruído)
    model : DBSCAN
        Modelo ajustado
    metrics : Dict
        Métricas de avaliação
    """
    model, labels = fit_dbscan(X, eps, min_samples)
    metrics = evaluate_clustering(X, labels)
    
    return labels, model, metrics


def apply_hierarchical(
    X: np.ndarray,
    n_clusters: int,
    linkage: str = 'ward',
    return_linkage: bool = False
) -> Union[Tuple[np.ndarray, AgglomerativeClustering, Dict], Tuple[np.ndarray, AgglomerativeClustering, Dict, np.ndarray]]:
    """
    Aplica Clustering Hierárquico e retorna labels, modelo e métricas.
    
    Parameters
    ----------
    X : np.ndarray
        Dados normalizados
    n_clusters : int
        Número de clusters
    linkage : str
        Método de ligação
    return_linkage : bool
        Se True, retorna também a matriz de linkage
        
    Returns
    -------
    labels : np.ndarray
        Labels dos clusters
    model : AgglomerativeClustering
        Modelo ajustado
    metrics : Dict
        Métricas de avaliação    linkage_matrix : np.ndarray (opcional)
        Matriz de linkage para dendrograma
    """
    model, labels = fit_hierarchical(X, n_clusters, linkage)
    metrics = evaluate_clustering(X, labels)
    
    if return_linkage:
        linkage_matrix = scipy_linkage(X, method=linkage)
        return labels, model, metrics, linkage_matrix
    
    return labels, model, metrics


def find_dbscan_params(
    X: np.ndarray,
    k: int = 5,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Encontra parâmetros ótimos para DBSCAN usando k-distance graph.
    
    Parameters
    ----------
    X : np.ndarray
        Dados normalizados
    k : int
        Número de vizinhos para calcular distância
    save_path : str, optional
        Caminho para salvar a figura
        
    Returns
    -------    fig : plt.Figure
        Figura com o k-distance graph
    """
    # Calcula distâncias aos k vizinhos mais próximos
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)
    
    # Pega a distância ao k-ésimo vizinho
    k_distances = distances[:, k-1]
    k_distances = np.sort(k_distances)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(k_distances)), k_distances, 'b-', linewidth=1)
    ax.set_xlabel('Pontos ordenados por distância')
    ax.set_ylabel(f'Distância ao {k}º vizinho mais próximo')
    ax.set_title(f'K-Distance Graph (k={k}) - Use o "cotovelo" para estimar eps')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✅ Figura salva em: {save_path}')
    
    return fig


def plot_cluster_distribution(
    labels: np.ndarray,
    title: str = 'Distribuição dos Clusters',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plota a distribuição de amostras por cluster.
    
    Parameters
    ----------
    labels : np.ndarray
        Labels dos clusters
    title : str
        Título do gráfico
    save_path : str, optional
        Caminho para salvar a figura
        
    Returns
    -------
    fig : plt.Figure
        Figura matplotlib
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [COLORS[i % len(COLORS)] if label != -1 else '#cccccc' 
              for i, label in enumerate(unique_labels)]
    
    bars = ax.bar(unique_labels.astype(str), counts, color=colors)
    
    # Adiciona valores nas barras
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f'{count:,}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Número de Amostras')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✅ Figura salva em: {save_path}')
    
    return fig
