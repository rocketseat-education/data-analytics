# 📦 DataClean Co. — Gabarito do Desafio (Módulo 4: Tratando Dados)
# Autor: Rocketseat Python (estilo das aulas)
# Objetivo: Demonstrar, passo a passo, como tratar dados (nulos, duplicatas, outliers),
#           normalizar, codificar variáveis categóricas, trabalhar com datas e gerar visualizações.
#
# Observação importante:
# - Este arquivo foi escrito em formato didático, com seções numeradas e comentários,
#   seguindo o mesmo espírito do gabarito "analise_vendas_techstore.py".
# - Para executar localmente, deixe o arquivo "dados_clientes.csv" na mesma pasta deste script.

# ---
# ## 1. Importação das bibliotecas
# pandas para manipulação tabular, numpy para cálculos numéricos e matplotlib para gráficos.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Garantir que os gráficos abram em janelas separadas quando usado em alguns ambientes:
# (Em notebooks não é necessário.)
# plt.switch_backend("agg")  # Descomente caso precise salvar sem interface gráfica

# ---
# ## 2. Leitura do arquivo CSV
# Lemos o CSV gerado para o desafio. Ele contém intencionalmente nulos, duplicatas,
# formatações diferentes de data e alguns outliers realistas.
print("== 2) Leitura do CSV ==")
df = pd.read_csv("dados_clientes.csv")
print("5 primeiras linhas:")
print(df.head(), "\n")

# ---
# ## 3. Exploração inicial dos dados (overview)
print("== 3) Overview ==")
print("Informações gerais:")
print(df.info(), "\n")
print("Formato (linhas, colunas):", df.shape)
print("Tipos de dados:\n", df.dtypes, "\n")
print("Estatísticas descritivas (numéricas):\n", df.describe(numeric_only=True), "\n")

# ---
# ## 4. Tratamento de valores ausentes
# Estratégia didática (simples e transparente):
# - 'idade': preencher com a mediana (evita distorções por outliers).
# - 'renda_anual': preencher com a mediana.
# - 'valor_compra': preencher com a mediana por 'categoria_produto' (contextual).
# - 'data_compra': manter nulos por ora (serão tratados ao converter datas).
print("== 4) Valores ausentes ==")
print("Contagem de nulos antes:\n", df.isna().sum(), "\n")

idade_mediana = df['idade'].median(skipna=True)
renda_mediana = df['renda_anual'].median(skipna=True)

df['idade'] = df['idade'].fillna(idade_mediana)
df['renda_anual'] = df['renda_anual'].fillna(renda_mediana)

# Para valor_compra, usar mediana por categoria (quando possível)
medianas_por_cat = df.groupby('categoria_produto')['valor_compra'].median()
def preencher_valor_compra(row):
    if pd.isna(row['valor_compra']):
        cat = row['categoria_produto']
        med_cat = medianas_por_cat.get(cat, df['valor_compra'].median())
        return med_cat
    return row['valor_compra']

df['valor_compra'] = df.apply(preencher_valor_compra, axis=1)

print("Contagem de nulos depois:\n", df.isna().sum(), "\n")

# ---
# ## 5. Tratamento de duplicatas
# Duas abordagens comuns:
# - Dropar duplicatas completas (linhas idênticas).
# - Ou dropar duplicatas por um subconjunto de colunas-chave.
# Aqui, faremos primeiro o drop de duplicatas completas, depois avaliamos caso a caso.
print("== 5) Duplicatas ==")
duplicatas_antes = df.duplicated().sum()
print("Duplicatas (linhas completamente idênticas) antes:", duplicatas_antes)

df = df.drop_duplicates(keep='first')
duplicatas_depois = df.duplicated().sum()
print("Duplicatas depois:", duplicatas_depois, "\n")

# Observação: se necessário, poderíamos usar um subset, por exemplo:
# df = df.drop_duplicates(subset=['id_cliente', 'data_compra', 'categoria_produto', 'valor_compra'], keep='first')

# ---
# ## 6. Conversão e engenharia de datas
# As datas vieram em formatos mistos. Vamos padronizar para datetime,
# aceitar erros (coercion) e criar colunas derivadas (ano, mes, dia_semana).
print("== 6) Datas ==")
# Tentar dia-primeiro e também outros formatos em sequência:
def parse_data_seguro(s):
    # Tenta ISO e dia-primeiro
    dt = pd.to_datetime(s, errors='coerce', dayfirst=True)
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors='coerce')  # segunda tentativa (ISO)
    return dt

df['data_compra'] = df['data_compra'].apply(parse_data_seguro)

print("Nulos em data_compra após parse:", df['data_compra'].isna().sum())

# Criar colunas derivadas
df['ano'] = df['data_compra'].dt.year
df['mes'] = df['data_compra'].dt.month
df['dia_semana'] = df['data_compra'].dt.day_name(locale='pt_BR') if hasattr(pd.Series.dt, "day_name") else df['data_compra'].dt.dayofweek

print("Preview de colunas de data:\n", df[['data_compra', 'ano', 'mes']].head(), "\n")

# ---
# ## 7. Análise e tratamento de outliers (IQR/Boxplot rule)
# Vamos aplicar a regra do IQR nas colunas numéricas-chave: 'renda_anual' e 'valor_compra'.
print("== 7) Outliers (IQR) ==")
def winsorizar_iqr(serie):
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    # Em vez de remover, faremos "capping" (winsorização) para manter o tamanho da amostra.
    return serie.clip(lower=lim_inf, upper=lim_sup), (lim_inf, lim_sup)

df['renda_anual'], (renda_inf, renda_sup) = winsorizar_iqr(df['renda_anual'])
df['valor_compra'], (compra_inf, compra_sup) = winsorizar_iqr(df['valor_compra'])

print(f"Limites renda_anual: [{renda_inf:.2f}, {renda_sup:.2f}]")
print(f"Limites valor_compra: [{compra_inf:.2f}, {compra_sup:.2f}]\n")

# ---
# ## 8. Normalização (escala 0-1) e Encoding de categóricas
# - Normalização Min-Max em 'renda_anual' e 'valor_compra' (novas colunas *_norm).
# - One-Hot Encoding para 'genero' e 'categoria_produto' (novas colunas dummy_*).
print("== 8) Normalização e Encoding ==")
def minmax(col):
    cmin, cmax = col.min(), col.max()
    return (col - cmin) / (cmax - cmin) if cmax != cmin else col*0

df['renda_anual_norm'] = minmax(df['renda_anual'])
df['valor_compra_norm'] = minmax(df['valor_compra'])

dummies = pd.get_dummies(df[['genero', 'categoria_produto']], prefix=['genero', 'cat'], dtype=int)
df = pd.concat([df, dummies], axis=1)

print("Colunas adicionadas (normalização e dummies) criadas com sucesso.\n")

# ---
# ## 9. Visualizações (matplotlib puro)
# Observação: cada gráfico tem sua própria figura. Não definimos cores manualmente.
print("== 9) Gráficos ==")
# 9.1 Histograma da renda anual
plt.figure(figsize=(8,5))
plt.hist(df['renda_anual'].dropna(), bins=10)
plt.title('Distribuição da Renda Anual')
plt.xlabel('Renda Anual (R$)')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()

# 9.2 Boxplot do valor_compra por categoria_produto
# Preparar os dados por categoria
cats = df['categoria_produto'].dropna().unique()
data_by_cat = [df.loc[df['categoria_produto'] == c, 'valor_compra'].dropna().values for c in cats]

plt.figure(figsize=(8,5))
plt.boxplot(data_by_cat, labels=cats, showmeans=True)
plt.title('Valor de Compra por Categoria')
plt.xlabel('Categoria do Produto')
plt.ylabel('Valor da Compra (R$)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 9.3 Scatterplot idade vs valor_compra
plt.figure(figsize=(8,5))
plt.scatter(df['idade'], df['valor_compra'])
plt.title('Idade vs Valor da Compra')
plt.xlabel('Idade')
plt.ylabel('Valor da Compra (R$)')
plt.tight_layout()
plt.show()

# 9.4 "Heatmap" simples de correlação (com imshow)
corr = df.select_dtypes(include=[np.number]).corr(numeric_only=True)
plt.figure(figsize=(7,6))
im = plt.imshow(corr, aspect='auto')
plt.title('Matriz de Correlação (numéricas)')
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
plt.yticks(range(len(corr.index)), corr.index)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# ---
# ## 10. Exportação do dataset limpo
print("== 10) Exportação ==")
saida = "dados_clientes_limpos.csv"
df.to_csv(saida, index=False, encoding="utf-8")
print(f"Arquivo '{saida}' exportado com sucesso!")

# ---
# ## 11. Conclusão
# Neste gabarito, aplicamos o pipeline essencial de tratamento de dados:
# - overview e diagnóstico
# - preenchimento de valores ausentes
# - deduplicação
# - padronização e engenharia de datas
# - detecção e capping de outliers via IQR
# - normalização e encoding de categóricas
# - visualizações para explorar distribuição, relação e correlação
# - exportação do dataset limpo para uso posterior
#
# Sugestões de estudo:
# - Experimente trocar a mediana por média ou KNN-Imputer para nulos.
# - Compare remover outliers vs. winsorizar e observe impacto nas métricas.
# - Teste padronização (z-score) no lugar de min-max e compare resultados.
# - Ajuste a granularidade temporal (por mês/semana) e crie novas features.
print("Pipeline finalizado com sucesso. 🚀")
