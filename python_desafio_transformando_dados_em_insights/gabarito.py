# 📦 TechGrow SaaS — Gabarito do Desafio (Módulo 05: Machine Learning na Prática)
# Autor: Rocketseat Python (estilo das aulas)
# Objetivo: Demonstrar, passo a passo, um pipeline de ML para prever churn em um SaaS:
#           exploração, pré-processamento, treino/avaliação de modelos (baseline e árvores),
#           interpretação e exportação de previsões.
#
# Observações importantes:
# - Este arquivo foi escrito em formato didático, com seções numeradas e comentários,
#   seguindo o mesmo espírito dos gabaritos anteriores (ex.: "analise_vendas_techstore.py").
# - Para executar localmente, deixe o arquivo "Caso_Pratico_base_churn_saas.csv" na mesma pasta.
# - Usamos apenas matplotlib para gráficos (um gráfico por figura e sem setar cores manualmente).

# ---
# ## 1. Importação das bibliotecas
print("== 1) Importação das bibliotecas ==")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# XGBoost é opcional: comente as 2 linhas abaixo se não tiver instalado
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# plt.switch_backend("agg")  # Descomente se precisar rodar sem interface gráfica (salvar imagens)

# ---
# ## 2. Leitura do CSV
print("\n== 2) Leitura do CSV ==")
csv_path = "Caso_Pratico_base_churn_saas.csv"
df = pd.read_csv(csv_path)
print("5 primeiras linhas:")
print(df.head(), "\n")

# ---
# ## 3. Exploração inicial (overview)
print("== 3) Overview ==")
print("Formato (linhas, colunas):", df.shape)
print("\nTipos de dados:\n", df.dtypes)
print("\nContagem de nulos:\n", df.isna().sum(), "\n")

# Inferimos o nome da coluna alvo: 'churn' (1 = cancelou, 0 = ativo).
# Ajuste aqui caso o nome seja outro.
TARGET_COL = "churn"
if TARGET_COL not in df.columns:
    raise ValueError(f"Coluna alvo '{TARGET_COL}' não encontrada no CSV.")

# Inspecionar balanceamento da classe
print("Distribuição da variável alvo (churn):")
print(df[TARGET_COL].value_counts(dropna=False), "\n")

# Gráfico 1: histograma simples do alvo (classe)
plt.figure(figsize=(6,4))
plt.hist(df[TARGET_COL].dropna())
plt.title('Distribuição da variável-alvo (churn)')
plt.xlabel('Churn (0=Não, 1=Sim)')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()

# ---
# ## 4. Seleção de features (X) e alvo (y)
print("== 4) Seleção de features (X) e alvo (y) ==")
y = df[TARGET_COL].astype(int)

# Por simplicidade didática:
# - Usaremos todas as colunas exceto o alvo como potenciais preditoras.
# - Se houver uma coluna de id (ex.: "customer_id"), podemos preservá-la para exportar previsões.
id_col = None
for cand in ["customer_id", "id_cliente", "id", "cliente_id"]:
    if cand in df.columns:
        id_col = cand
        break

X = df.drop(columns=[TARGET_COL])

# ---
# ## 5. Identificação de tipos de variáveis
print("== 5) Identificação de tipos ==")
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
if id_col and id_col in num_cols:
    # Remover id de numéricas para não entrar como feature
    num_cols = [c for c in num_cols if c != id_col]
if id_col and id_col in cat_cols:
    cat_cols = [c for c in cat_cols if c != id_col]

print("Colunas numéricas:", num_cols)
print("Colunas categóricas:", cat_cols, "\n")

# Gráfico 2: correlação numérica (quando houver)
if len(num_cols) > 1:
    corr = X[num_cols].corr(numeric_only=True)
    plt.figure(figsize=(7,6))
    im = plt.imshow(corr, aspect='auto')
    plt.title('Matriz de Correlação (features numéricas)')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

# ---
# ## 6. Divisão treino/teste (70/30) com estratificação no alvo
print("== 6) Train/Test Split ==")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print("Tamanhos:", X_train.shape, X_test.shape, "\n")

# ---
# ## 7. Pré-processamento (imputação + encoding + escala)
print("== 7) Pré-processamento ==")
# Estratégia simples e clara:
# - Numéricas: imputar mediana + padronizar
# - Categóricas: imputar constante + one-hot encoding
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# Para extrair nomes de features após o one-hot:
def get_feature_names(preprocessor, input_features):
    # pós-ajuste (fit), usamos get_feature_names_out
    return preprocessor.get_feature_names_out(input_features)

# ---
# ## 8. Modelos: baseline (Regressão Logística) + RandomForest (+ XGBoost opcional)
print("== 8) Definição de modelos ==")
models = []

log_reg = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=200, n_jobs=None))  # n_jobs não existe em Logistic; manter default
])
models.append(("LogisticRegression", log_reg))

rf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])
models.append(("RandomForest", rf))

if HAS_XGB:
    xgb = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=1.0,
            colsample_bytree=1.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        ))
    ])
    models.append(("XGBoost", xgb))

# ---
# ## 9. Treino e avaliação (métricas + matriz de confusão + ROC AUC)
print("== 9) Treino e avaliação ==")
def avaliar_modelo(name, pipeline, X_tr, y_tr, X_te, y_te):
    # Treino
    pipeline.fit(X_tr, y_tr)

    # Métricas
    y_pred = pipeline.predict(X_te)
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        y_proba = pipeline.predict_proba(X_te)[:, 1]
    else:
        # fallback para modelos sem predict_proba (não é o caso aqui)
        y_proba = None

    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    roc = roc_auc_score(y_te, y_proba) if y_proba is not None else np.nan

    print(f"\n--- {name} ---")
    print("Accuracy:", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    print("F1-score:", round(f1, 4))
    print("ROC AUC:", round(roc, 4) if not np.isnan(roc) else "N/A")
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, zero_division=0))

    # Matriz de confusão (gráfico próprio)
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(5,4))
    plt.imshow(cm)
    plt.title(f'Matriz de Confusão - {name}')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    # Anotar valores
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, int(val), ha='center', va='center')
    plt.tight_layout()
    plt.show()

    # Curva ROC (gráfico próprio)
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_te, y_proba)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr)
        plt.plot([0,1], [0,1])  # linha de referência (azar)
        plt.title(f'Curva ROC - {name}')
        plt.xlabel('Falso Positivo (FPR)')
        plt.ylabel('Verdadeiro Positivo (TPR)')
        plt.tight_layout()
        plt.show()

    return pipeline, (acc, prec, rec, f1, roc)

resultados = {}
modelos_treinados = {}
for name, pipe in models:
    modelo_treinado, mets = avaliar_modelo(name, pipe, X_train, y_train, X_test, y_test)
    resultados[name] = {
        "accuracy": mets[0],
        "precision": mets[1],
        "recall": mets[2],
        "f1": mets[3],
        "roc_auc": mets[4],
    }
    modelos_treinados[name] = modelo_treinado

print("\nResumo de resultados (dict):")
print(resultados)

# ---
# ## 10. Interpretação: importâncias / coeficientes e nomes de features
print("\n== 10) Interpretação de features ==")
# Para extrair nomes das features após o preprocessador, ajustamos com parte do pipeline
# (já foi ajustado durante o treino do primeiro modelo).
qualquer_modelo = next(iter(modelos_treinados.values()))
pre_fitted = qualquer_modelo.named_steps["preprocess"]

try:
    feature_names = get_feature_names(
        pre_fitted,
        input_features=num_cols + cat_cols
    )
    feature_names = list(feature_names)
except Exception:
    feature_names = [f"f{i}" for i in range(pre_fitted.transform(X_test).shape[1])]

def plot_importancias(name, fitted_pipeline, feature_names):
    clf = fitted_pipeline.named_steps["clf"]
    # Logistic: coef_; RF/XGB: feature_importances_
    if hasattr(clf, "coef_"):
        # Usar o módulo (valor absoluto) dos coeficientes para ranking
        coefs = np.abs(clf.coef_).ravel()
        valores = coefs
        titulo = f"Importância (|coef|) - {name}"
    elif hasattr(clf, "feature_importances_"):
        valores = clf.feature_importances_
        titulo = f"Importância (feature_importances_) - {name}"
    else:
        print(f"[Info] Modelo {name} não expõe importâncias.")
        return

    # Ordenar top 15 para visualização
    idx = np.argsort(valores)[::-1][:15]
    vals_top = valores[idx]
    names_top = [feature_names[i] for i in idx]

    plt.figure(figsize=(9,6))
    # gráfico de barras simples
    y_pos = np.arange(len(names_top))
    plt.barh(y_pos, vals_top)
    plt.yticks(y_pos, names_top)
    plt.gca().invert_yaxis()  # feature mais importante em cima
    plt.title(titulo)
    plt.xlabel('Importância (escala relativa)')
    plt.tight_layout()
    plt.show()

for name, fitted in modelos_treinados.items():
    plot_importancias(name, fitted, feature_names)

# ---
# ## 11. Exportação de previsões (para uso pelo time de produto)
print("== 11) Exportação de previsões ==")
# Escolha o melhor modelo por ROC AUC (fallback para F1 se ROC indisponível)
def escolher_melhor_modelo(res_dict):
    # Ordena por roc_auc (desc), depois f1
    def key_fn(item):
        v = item[1]
        roc = v.get("roc_auc", np.nan)
        f1 = v.get("f1", 0.0)
        if np.isnan(roc):
            roc = -1  # força para baixo se indisponível
        return (roc, f1)
    return sorted(res_dict.items(), key=key_fn, reverse=True)[0][0]

best_name = escolher_melhor_modelo(resultados)
best_model = modelos_treinados[best_name]
print(f"Melhor modelo escolhido: {best_name}")

# Probabilidade de churn para o conjunto de teste (para priorização)
if hasattr(best_model.named_steps["clf"], "predict_proba"):
    churn_prob = best_model.predict_proba(X_test)[:, 1]
else:
    churn_prob = best_model.predict(X_test).astype(float)

preds = best_model.predict(X_test)

# Montagem do dataframe de saída
saida_df = pd.DataFrame({
    "indice_original": X_test.index,
    "pred_churn": preds,
    "prob_churn": churn_prob
})

# Se houver id do cliente, adiciona
if id_col and id_col in X.columns:
    saida_df[id_col] = df.loc[X_test.index, id_col].values

saida_csv = "predicoes_churn.csv"
saida_df.to_csv(saida_csv, index=False, encoding="utf-8")
print(f"Arquivo '{saida_csv}' exportado com sucesso!\n")
print("Exemplo de linhas exportadas:")
print(saida_df.head())

# ---
# ## 12. Conclusão
# Neste gabarito, você percorreu um pipeline completo de ML:
# - overview e diagnóstico do dataset (balanceamento de classes incluso);
# - split de treino/teste estratificado (70/30);
# - pré-processamento com imputação, one-hot encoding e padronização;
# - treino de modelos (baseline: Regressão Logística; avançado: RandomForest; opcional: XGBoost);
# - avaliação com métricas (accuracy, precision, recall, f1, roc_auc), matriz de confusão e curva ROC;
# - interpretação de variáveis por coeficientes/impureza;
# - exportação de previsões para apoiar decisões do time de produto.
#
# Sugestões de estudo:
# - Experimente ajustar hiperparâmetros (GridSearchCV/RandomizedSearchCV) e comparar resultados.
# - Teste outras escalas (MinMaxScaler) e diferentes estratégias de imputação.
# - Avalie limiares de decisão (threshold) otimizados para Recall/Precision conforme objetivo do negócio.
# - Adicione validação cruzada (StratifiedKFold) para estimar variabilidade do desempenho.
print("Pipeline de Machine Learning finalizado com sucesso. 🚀")
