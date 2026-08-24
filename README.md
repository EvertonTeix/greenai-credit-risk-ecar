# Evaluating Computational and Carbon Costs of Machine Learning: The Impact of Feature Dimensionality During Model Development

Repository for the experiments of the Green AI in credit risk paper. This project analyzes how feature dimensionality affects predictive performance, computational consumption, and carbon emissions throughout the experimental lifecycle.

> Para a versão em Português, role a página para baixo / For the Portuguese version, scroll down.

---

# 🇺🇸 English Version

## Abstract

> The expansion of AI increases energy demand, making Green AI crucial. This study evaluates computational and carbon trade-offs in financial credit risk across six architectures under feature variations (±10% to ±50%). Using CodeCarbon, we propose ECAR to quantify emissions per unit of predictive effectiveness. The results show that across several model–dataset configurations, reducing feature dimensionality by 20% to 50% lowered emissions, generally with limited changes in F1-score. In contrast, the evaluated feature augmentation increased computational costs without proportional predictive gains. These findings highlight feature dimensionality and model architecture as key factors in the environmental efficiency of financial ML systems.

---

## Project Overview

This repository organizes notebooks, reusable functions, emission logs, and output files to reproduce the paper's experiments.

The focus is to compare machine learning models across feature reduction and augmentation scenarios, evaluating the impact on performance, energy, and carbon footprint.

### Core Concepts

- Compare six ML architectures on credit risk datasets.
- Vary feature dimensionality from -10% to +50%.
- Track emissions and energy using CodeCarbon.
- Record the relationship between emissions and predictive effectiveness using the ECAR metric.
- Measure the effect of feature engineering on the F1-score and environmental costs.

---

## Key Findings

- Reducing feature dimensionality by up to 40% generates statistically significant carbon savings without harming the F1-score.
- Increasing the number of features raises computational costs without providing consistent predictive gains.
- Model selection and data quality are decisive for more sustainable AI in financial systems.

---

## Repository Structure

> **Important:** The definitive results organized and used in the paper are specifically located inside:

```text
reports/resultados_artigo/{Dataset}/
```

```text
.
├── data/                                 # Datasets used in the experiments
├── notebooks/                            # Main Jupyter notebook executions
│   ├── bondora.ipynb
│   ├── german_credit_risk.ipynb
│   └── organizando_resultados.ipynb
├── reports/
│   ├── codecarbon/                       # Detailed physical emission logs
│   ├── graficos/                         # Generated plots and visual outputs
│   ├── resultados/                       # Intermediate results
│   └── resultados_artigo/                # Final organized result CSVs used in the paper
├── src/                                  # Source code
│   ├── features.py                       # Training, feature extraction functions
│   └── plots.py                          # Plotting functions
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Datasets

### German Credit Dataset

- 1,000 instances
- Binary classification (good vs. bad credit risk)

🔗 [Download Dataset](https://www.kaggle.com/datasets/uciml/german-credit)

### Bondora Peer-to-Peer Lending

- 179,235 instances (original) / 121,461 instances × 104 columns (preprocessed)
- Financial and demographic features

🔗 [Download Original Dataset (Kaggle)](https://www.kaggle.com/datasets/sid321axn/bondora-peer-to-peer-lending-loan-data)  
*This is the normal, raw dataset.*

🔗 [Download Preprocessed Dataset (Google Drive)](https://drive.google.com/file/d/1gR5RGl1F02l5bpeelWzMj-l-HNJVN_Dv/view?usp=sharing)  
*This is the preprocessed dataset used in the experiments. It is hosted on Drive because the preprocessing code is not included in this repository and the file is too large for GitHub.*

The repository notebooks use these datasets via relative paths.

---

## Evaluated Models & Hyperparameters

- **KNN**: `n_neighbors`: [3, 5, 7, 15, 30, 50], `metric`: ['euclidean', 'manhattan', 'minkowski']
- **Logistic Regression**: `C`: [0.01, 0.1, 1, 10, 100], `penalty`: ['l2'], `solver`: ['lbfgs', 'liblinear']
- **Random Forest**: `n_estimators`: [100, 200, 300], `max_depth`: [None, 5, 10], `criterion`: ['gini']
- **LightGBM**: `n_estimators`: [100, 200, 500], `learning_rate`: [0.01, 0.05, 0.1], `num_leaves`: [15, 31, 63]
- **XGBoost**: `n_estimators`: [100, 200, 500], `learning_rate`: [0.01, 0.05, 0.1], `max_depth`: [3, 6, 9], `subsample`: [0.8, 1.0]
- **MLP**: `hidden_layer_sizes`: [(64,), (128,)], `alpha`: [0.0001, 0.001, 0.01], `max_iter`: [200, 300]

---

## Experimental Pipeline

1. Preprocessing and cleaning based on the dataset.
2. Feature reduction using Mutual Information.
3. Feature augmentation with financial ratios, logs, interactions, and polynomial terms.
4. Cross-validation using KFold.
5. Hyperparameter grid search within each fold.
6. Tracking of emissions, energy, and execution time.

---

## Metrics

- Accuracy
- Precision
- Recall
- F1-score
- CO₂ emissions (grams)
- Energy consumption (kWh)
- Execution time (seconds)

### ECAR

ECAR = (Total Cumulative Energy (kWh) * CI) / Mean F1-score

*Where **CI** stands for **Carbon Intensity**, a fixed factor (e.g., 46.1 gCO₂/kWh) used to convert hardware energy consumption into carbon emissions, isolating the metric from temporal fluctuations in the energy grid.*

---

## Reusable Code

### `src/features.py`

Centralizes training, feature selection, feature augmentation, and carbon tracking.

Main routines:

- `reducao_codecarbon()`
- `aumento_codecarbon()`
- `kfold_and_gridsearch()`
- `treinar_com_codecarbon()`

### `src/plots.py`

Provides simple visualization utilities, such as class distributions.

---

## Dependencies

The `requirements.txt` file lists the libraries used in the project:

- pandas
- numpy
- scikit-learn
- lightgbm
- xgboost
- tensorflow
- codecarbon
- matplotlib
- seaborn

> **Note:** `carbontracker==2.1.2` is also listed for compatibility with legacy research environments.

---

## Installation (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## How to Run

1. Activate the virtual environment.
2. Open a terminal in the repository root.
3. Enter the `notebooks` folder.
4. Open the desired notebook and run the cells in order.

```powershell
cd notebooks
jupyter notebook
```

### Available notebooks

- `german_credit_risk.ipynb`
- `bondora.ipynb`

---

## Generated Outputs

During execution, the project saves:

- Detailed emission CSVs in `reports/codecarbon/`
- Raw intermediate results in `reports/resultados/`
- Final article results in:

```text
reports/resultados_artigo/Bondora/
reports/resultados_artigo/German/
```

---

## Final Note

This project highlights the environmental cost of AI and demonstrates that data optimization can reduce carbon footprints without sacrificing predictive performance.

---

# 🇧🇷 Versão em Português

## Resumo

> A expansão da IA aumenta a demanda por energia, tornando a Green AI crucial. Este estudo avalia os trade-offs computacionais e de carbono no risco de crédito financeiro em seis arquiteturas sob variações de atributos (±10% a ±50%). Utilizando o CodeCarbon, propomos a métrica ECAR para quantificar emissões por unidade de eficácia preditiva. Os resultados mostram que, em diversas configurações de modelo-conjunto de dados, a redução da dimensionalidade dos atributos em 20% a 50% diminuiu as emissões, geralmente com alterações limitadas no F1-score. Em contraste, o aumento de atributos avaliado aumentou os custos computacionais sem ganhos preditivos proporcionais. Essas descobertas destacam a dimensionalidade dos atributos e a arquitetura do modelo como fatores-chave na eficiência ambiental de sistemas financeiros de ML.

---

## Visão Geral do Projeto

Este repositório organiza notebooks, funções reutilizáveis, logs de emissão e arquivos de resultado para reproduzir os experimentos do artigo.

O foco é comparar modelos de machine learning em cenários de redução e aumento de features, avaliando o impacto sobre desempenho, energia e pegada de carbono.

### Principais Ideias

- Comparar seis arquiteturas de ML em dados de risco de crédito.
- Variar a dimensionalidade das features de -10% a +50%.
- Rastrear emissões e energia com CodeCarbon.
- Registrar a relação entre emissão e efetividade preditiva com a métrica ECAR.
- Medir o efeito da engenharia de atributos sobre o F1-score e o custo ambiental.

---

## Principais Resultados

- Reduzir a dimensionalidade das features em até 40% gera economia de carbono estatisticamente relevante sem prejudicar o F1-score.
- Aumentar o número de features eleva o custo computacional sem trazer ganho preditivo consistente.
- A escolha do modelo e a qualidade dos dados são decisivas para uma IA mais sustentável em sistemas financeiros.

---

## Estrutura do Repositório

> **Atenção:** Os resultados finais e definitivos utilizados no artigo estão organizados em:

```text
reports/resultados_artigo/{Base}/
```

```text
.
├── data/                                 # Bases de dados usadas nos experimentos
├── notebooks/                            # Execuções principais em Jupyter
│   ├── bondora.ipynb
│   ├── german_credit_risk.ipynb
│   └── organizando_resultados.ipynb
├── reports/
│   ├── codecarbon/                       # Logs detalhados de emissão física
│   ├── graficos/                         # Gráficos gerados
│   ├── resultados/                       # Resultados intermediários
│   └── resultados_artigo/                # Resultados finais usados no artigo
├── src/                                  # Código-fonte principal
│   ├── features.py                       # Funções de treino e extração de features
│   └── plots.py                          # Funções de visualização
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Bases de Dados

### German Credit Dataset

- 1.000 registros
- Classificação binária (risco bom vs. ruim)

🔗 [Download Dataset](https://www.kaggle.com/datasets/uciml/german-credit)

### Bondora Peer-to-Peer Lending

- 179.235 registros (original) / 121.461 registros × 104 colunas (pré-processado)
- Features financeiras e demográficas

🔗 [Download da Base Original (Kaggle)](https://www.kaggle.com/datasets/sid321axn/bondora-peer-to-peer-lending-loan-data)  
*Esta é a base de dados normal, sem nenhum tratamento.*

🔗 [Download da Base Pré-processada (Google Drive)](https://drive.google.com/file/d/1gR5RGl1F02l5bpeelWzMj-l-HNJVN_Dv/view?usp=sharing)  
*Esta é a base já pré-processada usada nos experimentos. Ela está no Drive porque o código de pré-processamento não está neste repositório e o arquivo é muito grande para o GitHub.*

---

## Modelos Avaliados e Hiperparâmetros

- **KNN**: `n_neighbors`: [3, 5, 7, 15, 30, 50], `metric`: ['euclidean', 'manhattan', 'minkowski']
- **Logistic Regression**: `C`: [0.01, 0.1, 1, 10, 100], `penalty`: ['l2'], `solver`: ['lbfgs', 'liblinear']
- **Random Forest**: `n_estimators`: [100, 200, 300], `max_depth`: [None, 5, 10], `criterion`: ['gini']
- **LightGBM**: `n_estimators`: [100, 200, 500], `learning_rate`: [0.01, 0.05, 0.1], `num_leaves`: [15, 31, 63]
- **XGBoost**: `n_estimators`: [100, 200, 500], `learning_rate`: [0.01, 0.05, 0.1], `max_depth`: [3, 6, 9], `subsample`: [0.8, 1.0]
- **MLP**: `hidden_layer_sizes`: [(64,), (128,)], `alpha`: [0.0001, 0.001, 0.01], `max_iter`: [200, 300]

---

## Pipeline Experimental

1. Pré-processamento e limpeza.
2. Redução de features por Mutual Information.
3. Aumento de features com razões, logs, interações e termos polinomiais.
4. Validação cruzada com KFold.
5. Busca em grade de hiperparâmetros.
6. Rastreamento de emissão, energia e tempo de execução.

---

## Métricas

- Accuracy
- Precision
- Recall
- F1-score
- Emissões de CO₂ (g)
- Consumo de energia (kWh)
- Tempo de execução (s)

### ECAR

ECAR = (Energia Total Acumulada (kWh) * CI) / F1-score Médio

*Onde **CI** significa **Carbon Intensity** (Intensidade de Carbono), um fator fixo (ex: 46.1 gCO₂/kWh) usado para converter o consumo de energia em emissões de carbono, isolando a métrica das flutuações temporais na rede elétrica.*

---

## Código Reutilizável

### `src/features.py`

Concentra:

- Treino
- Seleção de features
- Aumento de features
- Rastreamento de carbono

Funções principais:

- `reducao_codecarbon()`
- `aumento_codecarbon()`
- `kfold_and_gridsearch()`
- `treinar_com_codecarbon()`

### `src/plots.py`

Funções simples de visualização, incluindo distribuição de classes.

---

## Dependências

Bibliotecas utilizadas:

- pandas
- numpy
- scikit-learn
- lightgbm
- xgboost
- tensorflow
- codecarbon
- matplotlib
- seaborn

> **Observação:** `carbontracker==2.1.2` permanece listado por compatibilidade com experimentos legados.

---

## Instalação (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Como Executar

```powershell
cd notebooks
jupyter notebook
```

### Notebooks disponíveis

- `german_credit_risk.ipynb`
- `bondora.ipynb`

---

## Saídas Geradas

Durante a execução:

- CSVs de emissões em `reports/codecarbon/`
- Resultados intermediários em `reports/resultados/`
- Resultados finais do artigo em:

```text
reports/resultados_artigo/Bondora/
reports/resultados_artigo/German/
```

---

## Nota Final

Este projeto destaca o custo ambiental da IA e demonstra que a otimização de dados pode reduzir emissões de carbono sem sacrificar o desempenho preditivo.
