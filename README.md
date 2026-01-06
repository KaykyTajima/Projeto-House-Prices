# 🏠 House Prices Prediction - Advanced Regression Techniques

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)


## 📄 Sobre o Projeto

Este projeto é uma solução de Data Science para a clássica competição [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) do Kaggle.

O objetivo é prever o preço final de venda de casas em Ames, Iowa, utilizando 80 variáveis explicativas (features) que descrevem quase todos os aspectos dos imóveis residenciais.

> **Resultado Alcançado:** O modelo final obteve um **RMSLE (Root Mean Squared Logarithmic Error) de 0.1282**, posicionando a solução em um nível competitivo (Avançado) no ranking.

---

## 📊 Metodologia e Workflow

O projeto segue um pipeline rigoroso de Ciência de Dados:

1.  **Análise Exploratória de Dados (EDA):**
    * Estudo da distribuição da variável alvo (`SalePrice`).
    * Análise de correlação entre variáveis numéricas e o preço.
    * Visualização de dados categóricos.

2.  **Pré-processamento e Limpeza:**
    * **Tratamento de Outliers:** Remoção de anomalias baseadas em estatística e regras de negócio (ex: casas com áreas gigantescas e preço baixo).
    * **Dados Faltantes (NA):** Imputação estratégica baseada na natureza da variável (moda ou mediana).
    * **Transformação Logarítmica:** Aplicação de `np.log1p` na variável alvo para corrigir assimetria e aproximar a distribuição da normalidade.

3.  **Engenharia de Atributos:**
    * Encoding de variáveis categóricas (One-Hot Encoding / Label Encoding).
    * Normalização de escalas numéricas.

4.  **Modelagem Preditiva:**
    * Utilização de algoritmos de Ensemble: **Random Forest** e **Gradient Boosting**.
    * Otimização de hiperparâmetros com `GridSearchCV`.
    * Avaliação cruzada para garantir a generalização do modelo.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Manipulação de Dados:** Pandas, NumPy
* **Visualização:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn
* **Ambiente:** Jupyter Notebook / Google Colab

---

## 📈 Principais Insights Técnicos

Durante o desenvolvimento, destacam-se as seguintes abordagens técnicas:

* **Normalização do Alvo:** A variável `SalePrice` possui uma cauda longa à direita. O uso de `log(1+x)` melhorou significativamente a performance dos modelos lineares e baseados em árvore.
* **Conversão Reversa:** Para a submissão final, foi utilizada a função `np.expm1` para reverter as previsões logarítmicas para a escala original de preços (Dólares).
* **Pipeline Robusta:** O tratamento de dados foi estruturado para evitar *Data Leakage* (vazamento de dados) entre treino e teste.
