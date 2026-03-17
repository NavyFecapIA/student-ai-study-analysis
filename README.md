## Student Study Performance Predictor

Este projeto é um exemplo completo, de nível iniciante-intermediário, de um fluxo de Machine Learning em Python para prever a **nota final de um estudante** com base em seus **hábitos de estudo**.

O objetivo é mostrar, de forma simples e organizada, como:

- **criar um conjunto de dados sintético**;
- **explorar os dados (EDA)** com estatísticas e gráficos;
- **treinar e avaliar modelos de regressão** (Linear Regression e Random Forest Regressor);
- **interpretar resultados** por meio de métricas e visualizações.

---

## 1. Objetivo do Projeto

Prever a variável alvo:

- `final_exam_score` – nota final no exame (0 a 100),

usando as seguintes features:

- `study_hours_per_day` – horas de estudo por dia;
- `sleep_hours` – horas de sono por noite;
- `social_media_hours` – horas em redes sociais por dia;
- `practice_exams_completed` – número de simulados concluídos;
- `class_attendance_percentage` – frequência às aulas (0 a 100%).

O dataset é **sintético**, gerado de forma controlada para simular relações realistas entre hábitos de estudo e desempenho.

---

## 2. Estrutura do Projeto

`student-ai-study-analysis/`

- `dataset_generator.py` – gera o conjunto de dados sintético e salva em CSV.
- `analysis_and_model.py` – faz EDA, treina os modelos, avalia e gera previsões/visualizações.
- `requirements.txt` – dependências de Python.
- `README.md` – este arquivo de documentação.

---

## 3. Tecnologias Utilizadas

- **Python 3.9+** (recomendado)
- **pandas** – manipulação de dados tabulares.
- **numpy** – geração de números aleatórios e operação numérica.
- **scikit-learn** – modelos de Machine Learning e métricas.
- **matplotlib** – criação de gráficos.
- **seaborn** – visualizações estatísticas mais agradáveis.

Todas as dependências principais estão listadas em `requirements.txt`.

---

## 4. Como Executar o Projeto

### 4.1. Clonar ou copiar o diretório do projeto

Coloque a pasta `student-ai-study-analysis/` em um local de sua preferência.

### 4.2. Criar um ambiente virtual (opcional, mas recomendado)

No Windows (PowerShell):

```bash
python -m venv .venv
.venv\Scripts\activate
```

No Linux/Mac:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4.3. Instalar dependências

Dentro da pasta do projeto:

```bash
pip install -r requirements.txt
```

### 4.4. Gerar o dataset sintético (opcional)

Você pode rodar o gerador explicitamente:

```bash
python dataset_generator.py
```

Isso criará o arquivo `student_study_performance.csv` com pelo menos 300 linhas.

Se você pular este passo, o script principal (`analysis_and_model.py`) irá verificar se o CSV existe; se não existir, ele será gerado automaticamente.

### 4.5. Executar a análise e o treinamento de modelos

```bash
python analysis_and_model.py
```

O script irá:

- carregar (ou gerar) o dataset;
- mostrar informações do dataset e estatísticas descritivas;
- calcular e exibir a **matriz de correlação**;
- salvar visualizações em arquivos `.png`:
  - `correlation_heatmap.png` – mapa de calor da correlação;
  - `study_hours_vs_score.png` – dispersão horas de estudo vs nota final;
  - `feature_importance_random_forest.png` – importância das features segundo Random Forest;
- dividir o conjunto de dados em **treino (80%)** e **teste (20%)**;
- treinar:
  - **Linear Regression**
  - **Random Forest Regressor**
- avaliar cada modelo usando:
  - **Mean Absolute Error (MAE)**
  - **R² score**
- imprimir previsões de nota final para alguns **estudantes de exemplo**.

---

## 5. Explicação dos Modelos de Machine Learning

O problema é de **regressão**, pois queremos prever um valor numérico contínuo (nota).

### 5.1. Linear Regression

- Modelo simples que assume uma **relação linear** entre as features e o alvo.
- Fácil de interpretar: cada coeficiente indica quanto a nota tende a mudar ao variar uma feature, mantendo as demais constantes.
- Bom como **baseline** para comparar com modelos mais complexos.

### 5.2. Random Forest Regressor

- Conjunto (ensemble) de várias **árvores de decisão** de regressão.
- Cada árvore aprende um padrão ligeiramente diferente, e o resultado final é a média das previsões.
- Captura relações **não lineares** e interações entre variáveis de forma automática.
- Fornece uma medida de **importância das features**, mostrando quais variáveis mais contribuem para a previsão da nota.

---

## 6. Saídas Importantes

Ao rodar `analysis_and_model.py`, você terá:

- **Métricas de desempenho** dos modelos no conjunto de teste:
  - `Mean Absolute Error (MAE)`: erro médio absoluto da previsão.
  - `R² score`: quanta variância da nota é explicada pelo modelo (próximo de 1 é melhor).
- **Gráficos salvos em arquivo**:
  - `correlation_heatmap.png`
  - `study_hours_vs_score.png`
  - `feature_importance_random_forest.png`
- **Previsões de exemplo** para três perfis de estudantes:
  - estudante com pouco estudo e menor frequência;
  - estudante aplicado, com boa frequência e vários simulados;
  - estudante com alto uso de redes sociais e poucos simulados.

---

## 7. Próximos Passos e Extensões

Sugestões para evoluir o projeto:

- Adicionar novas features (por exemplo, nível de motivação, qualidade do ambiente de estudo).
- Testar outros modelos de regressão (XGBoost, Gradient Boosting, SVR).
- Fazer ajuste de hiperparâmetros (GridSearchCV, RandomizedSearchCV).
- Separar o código em módulos ainda mais organizados (por exemplo, `plots.py`, `models.py`).
- Criar um pequeno **API ou interface web** para inserir hábitos de estudo e ver a nota prevista.

Este repositório foi pensado para ser um ponto de partida didático para quem está começando em Machine Learning com Python.

