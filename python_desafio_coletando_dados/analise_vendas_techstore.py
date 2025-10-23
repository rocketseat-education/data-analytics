# 📊 Desafio TechStore — Análise de Dados
# Autor: Luis Fellipe / Rocketseat Python
# Objetivo: Gabarito do desafio do módulo de análise de dados

# ---
# ## 1. Importação das bibliotecas
# Aqui utilizamos as bibliotecas pandas e matplotlib, como foi mostrado nas aulas 3.6 e 3.8.
# pandas será usado para manipular o CSV e matplotlib para gerar gráficos simples.

import pandas as pd
import matplotlib.pyplot as plt

# ---
# ## 2. Leitura do arquivo CSV
# Assim como fizemos nas aulas da trilha, usamos o método `read_csv()` do pandas.
# No ambiente local, o arquivo 'vendas.csv' deve estar na mesma pasta do notebook.

vendas = pd.read_csv('vendas.csv')

# Visualizando as primeiras linhas
print('Visualizando as 5 primeiras linhas:')
print(vendas.head())

# ---
# ## 3. Exploração inicial dos dados
# Vamos verificar o formato geral da base: colunas, tipos e contagem de registros.

print('\nInformações gerais:')
print(vendas.info())

print('\nNúmero total de registros:', len(vendas))

# ---
# ## 4. Cálculo da receita total
# A receita de cada venda é dada por: quantidade × preço_unitário.
# Podemos criar uma nova coluna chamada 'receita' e somar o total.

vendas['receita'] = vendas['quantidade'] * vendas['preco_unitario']

receita_total = vendas['receita'].sum()
print(f'\n💰 Receita total: R$ {receita_total:,.2f}')

# ---
# ## 5. Consultas específicas
# Agora vamos responder às perguntas de negócio do desafio.

# 5.1 Filtrar vendas da categoria "Eletrônicos"
print('\n---\nVendas da categoria Eletrônicos:')
eletronicos = vendas[vendas['categoria'] == 'Eletrônicos']
print(eletronicos.head())

# 5.2 Produto mais vendido (em quantidade)
produto_mais_vendido = vendas.groupby('produto')['quantidade'].sum().idxmax()
print(f'\n📦 Produto mais vendido: {produto_mais_vendido}')

# 5.3 Região com maior valor de compras
regiao_maior_valor = vendas.groupby('regiao')['receita'].sum().idxmax()
print(f'\n🌎 Região com maior valor de compras: {regiao_maior_valor}')

# ---
# ## 6. Visualizações (opcional)
# Como mostrado nas aulas de visualização, podemos gerar gráficos simples para interpretar melhor os dados.

# 6.1 Receita por categoria
receita_por_categoria = vendas.groupby('categoria')['receita'].sum().sort_values()
plt.figure(figsize=(8,5))
receita_por_categoria.plot(kind='bar', title='Receita por Categoria', ylabel='Receita (R$)', xlabel='Categoria')
plt.show()

# 6.2 Evolução das vendas por mês
# Convertendo a coluna 'data' para datetime e agrupando por mês
vendas['data'] = pd.to_datetime(vendas['data'])

vendas_por_mes = vendas.groupby(vendas['data'].dt.to_period('M'))['receita'].sum()
plt.figure(figsize=(8,5))
vendas_por_mes.plot(kind='line', marker='o', title='Evolução das Vendas por Mês', ylabel='Receita (R$)', xlabel='Mês')
plt.show()

# ---
# ## 7. Extra: Relatório em Excel (opcional)
# Como foi ensinado na aula sobre leitura/escrita de arquivos, podemos exportar os resultados.

relatorio = vendas.groupby(['regiao', 'categoria'])['receita'].sum().reset_index()
relatorio.to_excel('relatorio_vendas.xlsx', index=False)
print('\nArquivo relatorio_vendas.xlsx exportado com sucesso!')

# ---
# ## Conclusão
# Neste notebook, aplicamos os principais conceitos de análise de dados:
# - Leitura de arquivos CSV
# - Exploração e filtragem de dados com pandas
# - Cálculo de métricas (receita, agrupamentos)
# - Visualização com matplotlib
# - Exportação de resultados
#
# Assim como vimos nos exemplos das aulas 3.6 e 3.8, a prática é essencial.
# Teste, explore e modifique as consultas para criar novos insights 🚀
