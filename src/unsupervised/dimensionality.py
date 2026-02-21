"""
Módulo de Redução de Dimensionalidade
======================================
Funções para PCA, t-SNE e análise de componentes principais.

Tech Challenge Fase 3 - Machine Learning Engineering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional, Union

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import joblib

# Configurações de visualização
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#2c3e50']


def fit_pca(X: np.ndarray, n_components: Optional[Union[int, float]] = None, 
            random_state: int = 42) -> Tuple[PCA, np.ndarray]:
    """
    Ajusta PCA nos dados.
    
    Parameters
    ----------
    X : np.ndarray
        Dados para redução (já escalados)
    n_components : int, float or None
        Número de componentes ou variância a manter.
        - int: número exato de componentes
        - float (0-1): variância mínima a ser explicada
        - None: mantém todos os componentes
    random_state : int
        Seed para reprodutibilidade
        
    Returns
    -------
    pca : PCA
        Modelo PCA ajustado
    X_pca : np.ndarray
        Dados transformados
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    
    return pca, X_pca


def transform_pca(pca: PCA, X: np.ndarray) -> np.ndarray:
    """
    Transforma dados usando PCA já ajustado.
    
    Parameters
    ----------
    pca : PCA
        Modelo PCA já ajustado
    X : np.ndarray
        Dados para transformar
        
    Returns
    -------
    X_pca : np.ndarray
        Dados transformados
    """
    return pca.transform(X)


def apply_pca(
    X: np.ndarray,
    n_components: Optional[Union[int, float]] = None
) -> Tuple[np.ndarray, PCA, np.ndarray]:
    """
    Aplica PCA e retorna dados transformados, modelo e variância explicada.
    
    Esta é uma função de conveniência que combina fit e transform.
    
    Parameters
    ----------
    X : np.ndarray
        Dados para transformar (já normalizados)
    n_components : int, float ou None
        Número de componentes ou variância a manter
        
    Returns
    -------
    X_pca : np.ndarray
        Dados transformados
    pca : PCA
        Modelo PCA ajustado
    explained_variance_ratio : np.ndarray
        Variância explicada por cada componente
    """
    pca, X_pca = fit_pca(X, n_components)
    return X_pca, pca, pca.explained_variance_ratio_


def find_optimal_components(X: np.ndarray, variance_threshold: float = 0.95) -> Dict:
    """
    Encontra o número ótimo de componentes para explicar a variância desejada.
    
    Parameters
    ----------
    X : np.ndarray
        Dados para análise (já escalados)
    variance_threshold : float
        Variância mínima a ser explicada (0-1)
        
    Returns
    -------
    results : dict
        Dicionário com informações sobre variância explicada
    """
    pca_full = PCA()
    pca_full.fit(X)
    
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_threshold = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    return {
        'explained_variance_ratio': pca_full.explained_variance_ratio_,
        'cumulative_variance': cumulative_variance,
        'n_components_for_threshold': n_components_threshold,
        'variance_threshold': variance_threshold,
        'total_components': len(pca_full.explained_variance_ratio_)
    }


def plot_variance_explained(results: Dict, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plota variância explicada por componente.
    
    Parameters
    ----------
    results : dict
        Resultados de find_optimal_components()
    save_path : str, optional
        Caminho para salvar a figura
        
    Returns
    -------
    fig : plt.Figure
        Figura do matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n_components = len(results['explained_variance_ratio'])
    x = range(1, n_components + 1)
    
    # Variância individual
    ax1 = axes[0]
    ax1.bar(x, results['explained_variance_ratio'], color=COLORS[0], alpha=0.7)
    ax1.set_xlabel('Componente Principal', fontsize=12)
    ax1.set_ylabel('Variância Explicada (%)', fontsize=12)
    ax1.set_title('Variância Explicada por Componente', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Variância acumulada
    ax2 = axes[1]
    ax2.plot(x, results['cumulative_variance'], 'bo-', linewidth=2, markersize=6)
    ax2.axhline(y=results['variance_threshold'], color='red', linestyle='--', 
                label=f'{results["variance_threshold"]*100:.0f}% variância')
    ax2.axvline(x=results['n_components_for_threshold'], color='green', linestyle='--',
                label=f'{results["n_components_for_threshold"]} componentes')
    ax2.fill_between(x, results['cumulative_variance'], alpha=0.3)
    ax2.set_xlabel('Número de Componentes', fontsize=12)
    ax2.set_ylabel('Variância Acumulada', fontsize=12)
    ax2.set_title('Variância Acumulada', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_xticks(x)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✅ Figura salva em: {save_path}')
    
    return fig


def get_feature_loadings(pca: PCA, feature_names: List[str], n_components: int = 5) -> pd.DataFrame:
    """
    Retorna os loadings (pesos) das features para cada componente.
    
    Parameters
    ----------
    pca : PCA
        Modelo PCA ajustado
    feature_names : list
        Nomes das features originais
    n_components : int
        Número de componentes para mostrar
        
    Returns
    -------
    loadings : pd.DataFrame
        DataFrame com os loadings
    """
    n_comp = min(n_components, pca.n_components_)
    
    loadings = pd.DataFrame(
        pca.components_[:n_comp].T,
        columns=[f'PC{i+1}' for i in range(n_comp)],
        index=feature_names
    )
    
    return loadings


def plot_feature_loadings(loadings: pd.DataFrame, n_top: int = 10, 
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plota os loadings mais importantes para cada componente.
    
    Parameters
    ----------
    loadings : pd.DataFrame
        DataFrame com os loadings
    n_top : int
        Número de features a mostrar por componente
    save_path : str, optional
        Caminho para salvar a figura
        
    Returns
    -------
    fig : plt.Figure
        Figura do matplotlib
    """
    n_components = loadings.shape[1]
    n_cols = min(3, n_components)
    n_rows = (n_components + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = np.atleast_1d(axes).flatten()
    
    for i, col in enumerate(loadings.columns):
        ax = axes[i]
        
        # Top features por valor absoluto
        top_features = loadings[col].abs().nlargest(n_top).index
        values = loadings.loc[top_features, col]
        
        colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in values]
        ax.barh(top_features, values, color=colors)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Loading')
        ax.set_title(f'{col}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    # Remover axes vazios
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle('Feature Loadings por Componente Principal', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✅ Figura salva em: {save_path}')
    
    return fig


def plot_biplot(pca: PCA, X_pca: np.ndarray, feature_names: List[str],
                labels: Optional[np.ndarray] = None, n_features: int = 10,
                save_path: Optional[str] = None) -> plt.Figure:
    """
    Plota biplot (scores + loadings) para os dois primeiros componentes.
    
    Parameters
    ----------
    pca : PCA
        Modelo PCA ajustado
    X_pca : np.ndarray
        Dados transformados
    feature_names : list
        Nomes das features
    labels : np.ndarray, optional
        Labels para colorir os pontos
    n_features : int
        Número de features a mostrar nos vetores
    save_path : str, optional
        Caminho para salvar a figura
        
    Returns
    -------
    fig : plt.Figure
        Figura do matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Scores (pontos)
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=COLORS[int(label) % len(COLORS)],
                       s=30, alpha=0.5, label=f'Cluster {label}')
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=COLORS[0], s=30, alpha=0.5)
    
    # Loadings (vetores)
    loadings = pca.components_[:2].T
    
    # Selecionar features com maiores loadings
    loading_magnitudes = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
    top_idx = np.argsort(loading_magnitudes)[-n_features:]
    
    # Escalar para visualização
    scale = np.abs(X_pca[:, :2]).max() / np.abs(loadings).max() * 0.8
    
    for idx in top_idx:
        ax.arrow(0, 0, loadings[idx, 0] * scale, loadings[idx, 1] * scale,
                 head_width=0.1, head_length=0.05, fc='red', ec='red', alpha=0.8)
        ax.text(loadings[idx, 0] * scale * 1.1, loadings[idx, 1] * scale * 1.1,
                feature_names[idx], fontsize=9, color='red')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('Biplot - PCA', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    
    if labels is not None:
        ax.legend(loc='best')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✅ Figura salva em: {save_path}')
    
    return fig


def fit_tsne(X: np.ndarray, n_components: int = 2, perplexity: float = 30.0,
             n_iter: int = 1000, random_state: int = 42) -> np.ndarray:
    """
    Aplica t-SNE para redução de dimensionalidade.
    
    Parameters
    ----------
    X : np.ndarray
        Dados para redução (já escalados)
    n_components : int
        Número de dimensões de saída (2 ou 3)
    perplexity : float
        Perplexidade (relacionada ao número de vizinhos)
    n_iter : int
        Número de iterações
    random_state : int
        Seed para reprodutibilidade
        
    Returns
    -------
    X_tsne : np.ndarray
        Dados transformados
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                n_iter=n_iter, random_state=random_state)
    X_tsne = tsne.fit_transform(X)
    
    return X_tsne


def plot_2d_projection(X_2d: np.ndarray, labels: Optional[np.ndarray] = None,
                       title: str = 'Projeção 2D', method: str = 'PCA',
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plota projeção 2D dos dados.
    
    Parameters
    ----------
    X_2d : np.ndarray
        Dados em 2D
    labels : np.ndarray, optional
        Labels para colorir os pontos
    title : str
        Título do gráfico
    method : str
        Método utilizado ('PCA', 't-SNE')
    save_path : str, optional
        Caminho para salvar a figura
        
    Returns
    -------
    fig : plt.Figure
        Figura do matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            color = 'gray' if label == -1 else COLORS[int(label) % len(COLORS)]
            label_name = 'Outliers' if label == -1 else f'Cluster {label}'
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, 
                       s=50, alpha=0.6, label=label_name)
        ax.legend(loc='best')
    else:
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=COLORS[0], s=50, alpha=0.6)
    
    ax.set_xlabel(f'{method} Componente 1', fontsize=12)
    ax.set_ylabel(f'{method} Componente 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✅ Figura salva em: {save_path}')
    
    return fig


def save_pca_model(pca: PCA, scaler: StandardScaler, filepath_pca: str, 
                   filepath_scaler: str) -> None:
    """
    Salva modelo PCA e scaler para uso futuro.
    
    Parameters
    ----------
    pca : PCA
        Modelo PCA ajustado
    scaler : StandardScaler
        Scaler utilizado
    filepath_pca : str
        Caminho para salvar o PCA
    filepath_scaler : str
        Caminho para salvar o scaler
    """
    joblib.dump(pca, filepath_pca)
    joblib.dump(scaler, filepath_scaler)
    print(f'✅ PCA salvo em: {filepath_pca}')
    print(f'✅ Scaler salvo em: {filepath_scaler}')


def load_pca_model(filepath_pca: str, filepath_scaler: str) -> Tuple[PCA, StandardScaler]:
    """
    Carrega modelo PCA e scaler.
    
    Parameters
    ----------
    filepath_pca : str
        Caminho do PCA
    filepath_scaler : str
        Caminho do scaler
        
    Returns
    -------
    pca : PCA
        Modelo PCA
    scaler : StandardScaler
        Scaler
    """
    pca = joblib.load(filepath_pca)
    scaler = joblib.load(filepath_scaler)
    print(f'✅ PCA carregado de: {filepath_pca}')
    print(f'✅ Scaler carregado de: {filepath_scaler}')
    return pca, scaler


def prepare_pca_for_supervised(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                                n_components: Union[int, float] = 0.95,
                                random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA, StandardScaler]:
    """
    Prepara dados com PCA para modelagem supervisionada (evita data leakage).
    
    O scaler e PCA são ajustados APENAS nos dados de treino e aplicados em todos.
    
    Parameters
    ----------
    X_train : np.ndarray
        Dados de treino
    X_val : np.ndarray
        Dados de validação
    X_test : np.ndarray
        Dados de teste
    n_components : int or float
        Número de componentes ou variância a manter
    random_state : int
        Seed para reprodutibilidade
        
    Returns
    -------
    X_train_pca : np.ndarray
        Treino transformado
    X_val_pca : np.ndarray
        Validação transformada
    X_test_pca : np.ndarray
        Teste transformado
    pca : PCA
        Modelo PCA ajustado
    scaler : StandardScaler
        Scaler ajustado
    """
    # 1. Ajustar scaler apenas no treino
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Ajustar PCA apenas no treino
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f'✅ PCA preparado para modelagem supervisionada')
    print(f'   Componentes: {pca.n_components_}')
    print(f'   Variância explicada: {pca.explained_variance_ratio_.sum()*100:.2f}%')
    print(f'   Shape treino: {X_train_pca.shape}')
    print(f'   Shape validação: {X_val_pca.shape}')
    print(f'   Shape teste: {X_test_pca.shape}')
    
    return X_train_pca, X_val_pca, X_test_pca, pca, scaler
