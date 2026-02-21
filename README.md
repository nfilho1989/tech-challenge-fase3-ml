# 🛫 Tech Challenge Fase 3 - Previsão de Atrasos de Voos

## Machine Learning Engineering - FIAP

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green.svg)](https://scikit-learn.org/)

---

## 📋 Objetivo do Projeto

Desenvolver um pipeline completo de ciência de dados para análise e predição de atrasos em voos domésticos nos Estados Unidos, aplicando técnicas de **Machine Learning Supervisionado** e **Não-Supervisionado**.

### Objetivos Específicos

- **Análise Exploratória**: Identificar padrões, sazonalidades e fatores críticos de atrasos
- **Modelagem Supervisionada**: Prever se um voo vai atrasar e/ou quanto tempo
- **Modelagem Não-Supervisionada**: Segmentar aeroportos/voos por perfil de atraso
- **Insights Acionáveis**: Gerar recomendações baseadas nos dados

---

## 📊 Dataset

| Informação | Valor |
|------------|-------|
| **Fonte** | [2015 Flight Delays and Cancellations - Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays) |
| **Período** | Janeiro a Dezembro de 2015 |
| **Registros** | 5.819.079 voos |
| **Companhias Aéreas** | 14 |
| **Aeroportos** | 628-629 |
| **Rotas Únicas** | 8.609 |

### Arquivos de Dados

| Arquivo | Descrição |
|---------|-----------|
| `airlines.csv` | Informações das companhias aéreas (código IATA, nome) |
| `airports.csv` | Informações dos aeroportos (código, nome, cidade, estado, coordenadas) |
| `flights.csv` | Dados detalhados de cada voo (horários, atrasos, cancelamentos, etc.) |

---

## 🗂️ Estrutura do Repositório

```
projeto_3/
│
├── 📁 config/                           # Arquivos de configuração
│   ├── model_config.yaml                # Hiperparâmetros modelos supervisionados
│   ├── clustering_config.yaml           # Hiperparâmetros modelos não-supervisionados
│   └── feature_config.yaml              # Configuração das features
│
├── 📁 data/                             # Dados (não versionados no Git)
│   ├── raw/                             # Dados brutos originais
│   │   ├── airlines.csv
│   │   ├── airports.csv
│   │   └── flights.csv
│   └── processed/                       # Dados processados
│       ├── flights_processed.parquet    # Dataset limpo para modelagem
│       ├── flights_complete.parquet     # Dataset completo enriquecido
│       ├── train.parquet                # Conjunto de treino
│       ├── validation.parquet           # Conjunto de validação
│       ├── test.parquet                 # Conjunto de teste
│       └── clusters.parquet             # Dados com labels de clusters
│
├── 📁 docs/                             # Documentação do projeto
│   ├── data_dictionary.md               # Dicionário de dados
│   └── model_card.md                    # Model Card (documentação do modelo)
│
├── 📁 models/                           # Modelos treinados (serializados)
│   ├── supervised/                      # Modelos supervisionados
│   │   ├── classifier.joblib            # Modelo de classificação
│   │   └── regressor.joblib             # Modelo de regressão
│   ├── unsupervised/                    # Modelos não-supervisionados
│   │   ├── kmeans.joblib                # Modelo K-Means
│   │   └── pca.joblib                   # PCA para redução dimensional
│   └── transformers/                    # Encoders e scalers
│       ├── label_encoders.joblib        # Encoders categóricos
│       └── scaler.joblib                # Scaler para features numéricas
│
├── 📁 notebooks/                        # Jupyter Notebooks
│   ├── 01_eda.ipynb                     # Análise Exploratória de Dados
│   ├── 02_feature_engineering.ipynb     # Engenharia de Features
│   ├── 03_modeling_classification.ipynb # Classificação (atraso Sim/Não)
│   ├── 04_modeling_regression.ipynb     # Regressão (minutos de atraso)
│   ├── 05_clustering_analysis.ipynb     # Clusterização (K-Means, DBSCAN)
│   ├── 06_dimensionality_reduction.ipynb# Redução Dimensional (PCA, t-SNE)
│   └── 07_model_evaluation.ipynb        # Comparação e conclusões finais
│
├── 📁 reports/                          # Relatórios e visualizações
│   ├── figures/                         # Gráficos gerados
│   │   ├── eda/                         # Figuras da análise exploratória
│   │   ├── supervised/                  # Figuras dos modelos supervisionados
│   │   └── unsupervised/                # Figuras dos modelos não-supervisionados
│   ├── 01_relatorio_eda.md              # Relatório da EDA
│   ├── 02_relatorio_supervisionado.md   # Relatório modelos supervisionados
│   └── 03_relatorio_nao_supervisionado.md # Relatório modelos não-supervisionados
│
├── 📁 src/                              # Código fonte reutilizável
│   ├── __init__.py
│   ├── data_loader.py                   # Carregamento de dados
│   ├── preprocessing.py                 # Pré-processamento
│   ├── feature_engineering.py           # Criação de features
│   ├── supervised/                      # Módulos supervisionados
│   │   ├── __init__.py
│   │   ├── train.py                     # Funções de treinamento
│   │   └── evaluate.py                  # Métricas supervisionadas
│   ├── unsupervised/                    # Módulos não-supervisionados
│   │   ├── __init__.py
│   │   ├── clustering.py                # K-Means, DBSCAN, Hierárquico
│   │   ├── dimensionality.py            # PCA, t-SNE
│   │   └── evaluate.py                  # Métricas não-supervisionadas
│   └── utils.py                         # Funções auxiliares
│
├── .gitignore                           # Arquivos ignorados pelo Git
├── requirements.txt                     # Dependências do projeto
└── README.md                            # Documentação principal (este arquivo)
```

---

## 📓 Notebooks

### Sequência de Execução

| # | Notebook | Tipo | Descrição | Status |
|---|----------|------|-----------|--------|
| 01 | `01_eda.ipynb` | Exploratório | Análise exploratória completa dos dados | ✅ Concluído |
| 02 | `02_feature_engineering.ipynb` | Preparação | Criação, transformação e seleção de features | ⏳ Pendente |
| 03 | `03_modeling_classification.ipynb` | Supervisionado | Prever SE haverá atraso (>= 15 min) | ⏳ Pendente |
| 04 | `04_modeling_regression.ipynb` | Supervisionado | Prever QUANTO tempo de atraso | ⏳ Pendente |
| 05 | `05_clustering_analysis.ipynb` | Não-Supervisionado | Segmentação de voos/aeroportos | ⏳ Pendente |
| 06 | `06_dimensionality_reduction.ipynb` | Não-Supervisionado | PCA, t-SNE para visualização | ⏳ Pendente |
| 07 | `07_model_evaluation.ipynb` | Avaliação | Comparação final e conclusões | ⏳ Pendente |

### Fluxo de Trabalho

```
                              01_eda.ipynb
                                   │
                                   ▼
                         02_feature_engineering.ipynb
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                   │
                 ▼                                   ▼
    ┌────────────────────────┐          ┌────────────────────────┐
    │     SUPERVISIONADO     │          │   NÃO-SUPERVISIONADO   │
    ├────────────────────────┤          ├────────────────────────┤
    │ 03_classification.ipynb│          │ 05_clustering.ipynb    │
    │ 04_regression.ipynb    │          │ 06_dim_reduction.ipynb │
    └───────────┬────────────┘          └───────────┬────────────┘
                │                                   │
                └─────────────────┬─────────────────┘
                                  │
                                  ▼
                       07_model_evaluation.ipynb
```

---

## 🤖 Modelagem

### Aprendizado Supervisionado

#### Classificação (Prever SE vai atrasar)
- **Variável Alvo**: `IS_DELAYED` (1 se atraso >= 15 min, 0 caso contrário)
- **Algoritmos**:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost
  - LightGBM
- **Métricas**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

#### Regressão (Prever QUANTO tempo de atraso)
- **Variável Alvo**: `ARRIVAL_DELAY` (minutos de atraso)
- **Algoritmos**:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - XGBoost Regressor
  - LightGBM Regressor
- **Métricas**: MAE, RMSE, R², MAPE

### Aprendizado Não-Supervisionado

#### Clusterização
- **Objetivos**:
  - Segmentar aeroportos por perfil operacional
  - Agrupar voos por padrões de atraso
  - Identificar companhias com comportamentos similares
- **Algoritmos**:
  - K-Means
  - DBSCAN
  - Hierarchical Clustering
- **Métricas**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index

#### Redução de Dimensionalidade
- **Objetivos**:
  - Visualizar dados em 2D/3D
  - Identificar features mais importantes
  - Reduzir complexidade para clustering
- **Técnicas**:
  - PCA (Principal Component Analysis)
  - t-SNE (t-Distributed Stochastic Neighbor Embedding)

---

## 📈 Principais Resultados da EDA

### Estatísticas Gerais

| Métrica | Valor |
|---------|-------|
| Taxa de Atraso (>= 15 min) | 18,56% |
| Taxa de Cancelamento | 1,54% |
| Atraso Médio | 4,41 min |
| Atraso Mediano | -5,00 min (maioria chega adiantado) |

### Insights Principais

1. **Efeito Cascata**: Principal causa de atraso é "Aeronave Atrasada" (atrasos propagados)
2. **Padrão Temporal**: Voos matutinos (5h-9h) têm 3-4x menos atrasos que voos noturnos
3. **Sazonalidade**: Fevereiro é o pior mês (tempestades de inverno); Setembro/Outubro os melhores
4. **Aeroportos Críticos**: LaGuardia (LGA) e Chicago O'Hare (ORD) são os mais problemáticos
5. **Melhores Companhias**: Hawaiian Airlines (11% atrasos) e Alaska Airlines (13% atrasos)

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.10+
- Jupyter Notebook ou JupyterLab

### Instalação

1. **Clone o repositório**
```bash
git clone <url-do-repositorio>
cd projeto_3
```

2. **Crie e ative o ambiente virtual**
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

3. **Instale as dependências**
```powershell
pip install -r requirements.txt
```

4. **Baixe os dados**
   - Acesse: [Kaggle - 2015 Flight Delays](https://www.kaggle.com/datasets/usdot/flight-delays)
   - Extraia os arquivos para `data/raw/`

5. **Execute os notebooks**
```powershell
jupyter notebook
```

---

## 📦 Dependências Principais

| Categoria | Bibliotecas |
|-----------|-------------|
| **Manipulação de Dados** | pandas, numpy, pyarrow |
| **Machine Learning** | scikit-learn, xgboost, lightgbm |
| **Balanceamento** | imbalanced-learn (SMOTE) |
| **Visualização** | matplotlib, seaborn, plotly |
| **Interpretabilidade** | shap |
| **Persistência** | joblib |

---

## 📊 Métricas de Avaliação

### Supervisionado - Classificação

| Métrica | Descrição |
|---------|-----------|
| **Accuracy** | Proporção de predições corretas |
| **Precision** | Dos previstos como atrasados, quantos realmente atrasaram |
| **Recall** | Dos que atrasaram, quantos foram previstos corretamente |
| **F1-Score** | Média harmônica entre Precision e Recall |
| **ROC-AUC** | Área sob a curva ROC |

### Supervisionado - Regressão

| Métrica | Descrição |
|---------|-----------|
| **MAE** | Erro absoluto médio (em minutos) |
| **RMSE** | Raiz do erro quadrático médio |
| **R²** | Coeficiente de determinação |
| **MAPE** | Erro percentual absoluto médio |

### Não-Supervisionado

| Métrica | Descrição |
|---------|-----------|
| **Silhouette Score** | Qualidade geral dos clusters (-1 a 1) |
| **Davies-Bouldin** | Separação entre clusters (menor = melhor) |
| **Inertia** | Coesão interna (para Elbow Method) |
| **Variância Explicada** | Qualidade da redução dimensional |

---

## 📁 Entregáveis

- [x] **Repositório GitHub** com código completo
- [ ] **Vídeo de Apresentação** (5-10 minutos)
- [x] **Relatório EDA** (`reports/01_relatorio_eda.md`)
- [ ] **Relatório Modelagem Supervisionada** (`reports/02_relatorio_supervisionado.md`)
- [ ] **Relatório Modelagem Não-Supervisionada** (`reports/03_relatorio_nao_supervisionado.md`)

---

## 👥 Equipe

| Nome | RM | Responsabilidade |
|------|-----|------------------|
| [Nome 1] | [RM1] | [Responsabilidade] |
| [Nome 2] | [RM2] | [Responsabilidade] |
| [Nome 3] | [RM3] | [Responsabilidade] |

---

## 📚 Referências

- [Documentação scikit-learn](https://scikit-learn.org/stable/)
- [Documentação XGBoost](https://xgboost.readthedocs.io/)
- [Documentação LightGBM](https://lightgbm.readthedocs.io/)
- [SHAP - SHapley Additive exPlanations](https://shap.readthedocs.io/)
- [Kaggle - 2015 Flight Delays Dataset](https://www.kaggle.com/datasets/usdot/flight-delays)

---

## 📜 Licença

Este projeto foi desenvolvido para fins educacionais como parte do programa de pós-graduação em Machine Learning Engineering da FIAP.

---

*Tech Challenge Fase 3 - Machine Learning Engineering - FIAP - 2026*
