# Preditor de Risco Cardíaco

Aplicação de triagem de risco cardiovascular baseada em aprendizado de máquina. A partir de um questionário de 20 perguntas sobre hábitos, condições de saúde e perfil socioeconômico, o modelo estima a probabilidade de o indivíduo apresentar doença cardíaca ou histórico de infarto.

O modelo foi treinado sobre o **BRFSS 2015 (CDC, EUA)**, com 253.680 respostas e 22 indicadores de saúde.

> **Aviso:** este é um instrumento educacional e de pesquisa. Não substitui avaliação clínica profissional.

---

## Resultados

Quatro algoritmos foram comparados sob as mesmas condições (split estratificado 80/20, `random_state=42`):

| Modelo | AUC-ROC | AUC-PR | Recall | Precision | F1 | Tempo (s) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0,8512 | 0,6338 | 0,8129 | 0,5167 | 0,6318 | 0,1 |
| Random Forest | 0,8505 | 0,6310 | 0,8159 | 0,5122 | 0,6293 | 1,0 |
| Gradient Boosting | 0,8530 | 0,6394 | 0,5045 | 0,6433 | 0,5655 | 13,5 |
| **XGBoost** (escolhido) | **0,8524** | 0,6380 | **0,8272** | 0,5041 | 0,6264 | 0,8 |

O XGBoost foi selecionado por maximizar o **recall** — em triagem de risco, um falso negativo custa mais do que um falso positivo. As probabilidades finais passam por calibração isotônica (`CalibratedClassifierCV`), e o threshold padrão de decisão é **0,35** (há também um threshold sensível de 0,25 disponível nos metadados).

---

## Estrutura

```
api/            FastAPI — endpoints de predição e utilitários
  main.py         rotas (/health, /predict, /calcular-bmi) e mount do frontend
  predictor.py    carregamento do modelo, inferência e features derivadas
  schemas.py      validação de entrada/saída (Pydantic)
frontend/
  index.html      interface web estática (servida pela própria API em /ui)
  script.js       lógica do formulário e chamadas à API
  style.css
  app.py          interface alternativa em Streamlit
model/
  heart_disease_model.pkl        XGBoost calibrado
  heart_disease_metadata.json    features, métricas, thresholds e rótulos
notebooks/
  PreProcessamentoAED.ipynb      EDA, limpeza, feature engineering e balanceamento
  Treinamento.ipynb              treino, comparação, calibração e exportação
data/
  raw/                           BRFSS 2015 original (253.680 linhas)
  processed/                     dataset completo e versão balanceada 1:3
```

---

## Instalação

Requer Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Execução

**API** (a partir da pasta `api/`, que é o diretório de trabalho esperado pelos imports):

```bash
cd api
uvicorn main:app --reload
```

- Interface web: http://127.0.0.1:8000/ui
- Documentação interativa: http://127.0.0.1:8000/docs

**Streamlit** (opcional, com a API já rodando):

```bash
streamlit run frontend/app.py
```

---

## API

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Status do modelo carregado e métricas principais |
| `POST` | `/predict` | Recebe as 23 features e retorna a probabilidade de risco |
| `POST` | `/calcular-bmi` | Calcula IMC e categoria a partir de peso e altura |

### Exemplo

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "BMI": 28.4, "BMI_cat": 2,
    "HighBP": 1, "HighChol": 0, "CholCheck": 1,
    "Smoker": 0, "Stroke": 0, "Diabetes": 0,
    "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
    "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "GenHlth": 3, "MentHlth": 2, "PhysHlth": 1,
    "DiffWalk": 0, "Sex": 1, "Age": 7,
    "Education": 5, "Income": 5, "HealthyLifestyle": 4
  }'
```

```json
{
  "probabilidade": 0.1063,
  "probabilidade_pct": "10.6%",
  "risco": { "label": "Baixo", "color": "#1D9E75" },
  "alerta": false,
  "threshold_usado": 0.35,
  "modelo": "XGBoost",
  "aviso": "Este resultado é uma estimativa de triagem e não substitui avaliação médica profissional."
}
```

Faixas de risco: **Baixo** (< 15%), **Moderado** (15–35%), **Alto** (35–60%), **Muito Alto** (≥ 60%).

---

## Modelagem

**Features derivadas.** Duas features não vêm do questionário e são calculadas antes da inferência:

- `BMI` / `BMI_cat` — IMC a partir de peso e altura, categorizado em abaixo do peso, normal, sobrepeso e obesidade.
- `HealthyLifestyle` — score de 0 a 5 somando atividade física, consumo de frutas, consumo de vegetais, ausência de consumo pesado de álcool e não tabagismo.

**Balanceamento.** O dataset original é fortemente desbalanceado (~9% de casos positivos). Foi aplicado undersampling da classe majoritária na razão 1:3, gerando `brfss_preprocessed_balanced.csv` (95.572 linhas). O XGBoost também usa `scale_pos_weight` para compensar o desbalanceamento residual.

**Codificação das variáveis.** As escalas seguem a codificação original do BRFSS: `Age` é faixa etária ordinal de 1 (18–24) a 13 (80+), `GenHlth` vai de 1 (excelente) a 5 (ruim), `Education` de 1 a 6 e `Income` de 1 a 8. A lista completa de features, seus rótulos e faixas válidas está em `model/heart_disease_metadata.json`.

---

## Dados

[Heart Disease Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset) — derivado do [BRFSS 2015](https://www.cdc.gov/brfss/annual_data/annual_2015.html) do CDC. Variável-alvo: `HeartDiseaseorAttack`.

---

## Licença

Distribuído sob a [Licença MIT](LICENSE).
