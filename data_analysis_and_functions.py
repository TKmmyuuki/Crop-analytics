import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Leitura do dataframe e limpeza

df = pd.read_excel('Data_dirty.xlsx', sheet_name='Crops_data')

cidades = pd.read_excel('Data_dirty.xlsx', sheet_name='Districts')

df_arrumar = df[~df['Dist Name'].isin(cidades['Dist Name'])] # Criando um dataframe com os dados com as cidades de nomes incorretos 

df = df[df['Dist Name'].isin(cidades['Dist Name'])]  # Filtrando o dataframe para que contenha apenas as cidados com nomes corretos

## Correção dos nomes 

df_arrumar['Dist Name'] = df_arrumar['Dist Name'].replace('Kadap YSR', 'Kadapa YSR')
df_arrumar['Dist Name'] = df_arrumar['Dist Name'].replace('Gurdspur', 'Gurdaspur')
df_arrumar['Dist Name'] = df_arrumar['Dist Name'].replace('Ludhana', 'Ludhiana')
df_arrumar['Dist Name'] = df_arrumar['Dist Name'].replace('Madi', 'Mandi')
df_arrumar['Dist Name'] = df_arrumar['Dist Name'].replace('Korput', 'Koraput')

## Concatenando o dataframe

df = pd.concat([df, df_arrumar], axis=0)


# Definindo uma lista para as culturas, e lendo dataframe de preços

df_prices = pd.read_excel('Data_dirty.xlsx', sheet_name='Prices per kg')

df_prices.drop(['Unnamed: 0', 'Unnamed: 3'] + [i for i in range(0,10)], axis = 1, inplace = True)

df_prices.dropna(inplace = True)

df_prices['Estimated Price per kg (R$)'] = df_prices['Estimated Price per kg (R$)'].apply(lambda x: float(x.split("''")[1].split("''")[0]))

df_prices.reset_index(inplace = True)

df_prices.drop('index', axis = 1, inplace = True)

df_prices['Crop'] = df_prices['Crop'].apply(lambda x: x.upper())

df_prices.drop([6,27,7,17,14,5], inplace = True)

df_prices.reset_index(inplace = True)

df_prices.drop('index', axis = 1, inplace = True)

crops = list(df_prices['Crop'].values)

crops = [i.upper() for i in crops]


### Lendo o dataframe de distritos - problema: alguns distritos estão sem coordenadas

df_states = pd.read_csv('states_with_coordinates.csv')

### Definindo as funções de construção de gráficos

def production_yield(df, ano, crop):
    crop = str(crop).upper()
    df_filtered = df[df['Year'] == ano]

    # Filtra as colunas que contenham o nome da cultura
    column_rice = ['Year']

    for column in df_filtered.columns:
        if crop in str(column).upper():
            column_rice.append(column)

    df_filtered = df_filtered[column_rice]

    # Calcula o rendimento
    df_filtered['YIELD (1000 tons)'] = df_filtered[f'{crop} AREA (1000 ha)'] * df_filtered[f'{crop} YIELD (Kg per ha)'] / 1000

    crop_yield = df_filtered['YIELD (1000 tons)'].sum()
    crop_production = df_filtered[f'{crop} PRODUCTION (1000 tons)'].sum()

    result = crop_production / crop_yield

    return f'{round(result*100, 2)}%'


def grafico_cultura_estado(df, estado, crop): 
    crop = str(crop).upper()
    df_estado = df[df['State Name'] == estado]
    colunas_cultura = ['Year']

    # Adiciona colunas que contêm o nome da cultura (por exemplo: RICE PRODUCTION)
    for column in df_estado.columns:
        if crop in str(column).upper():
            colunas_cultura.append(column)

    if len(colunas_cultura) < 2:
        st.warning(f"Colunas para a cultura '{crop}' não encontradas no estado '{estado}'.")
        return go.Figure()  # Figura vazia como fallback

    df_estado = df_estado[colunas_cultura]

    # Identifica a coluna de produção
    col_producao = next((col for col in colunas_cultura if 'PRODUCTION' in str(col).upper()), None)

    if not col_producao:
        st.warning(f"Coluna de produção para a cultura '{crop}' não encontrada.")
        return go.Figure()

    # Agrupa a produção total por ano
    df_grouped = df_estado.groupby('Year')[col_producao].sum().reset_index()

    # Cria o gráfico
    fig = px.line(
        df_grouped,
        x='Year',
        y=col_producao,
        title=f'Produção de {crop} por ano em {estado}',
        markers=True
    )

    return fig



def lucro_anual_total(cultura, ano, df, df_prices):
    # Define o nome da coluna de produção
    nome_coluna = f"{cultura.upper()} PRODUCTION (1000 tons)"

    # Filtra a produção do ano e converte de mil toneladas para kg
    producao_ano_kg = df.loc[df['Year'] == ano, nome_coluna] * 1_000_000

    # Obtém o preço por kg da cultura
    preco = df_prices.loc[df_prices['Crop'] == cultura, 'Estimated Price per kg (R$)'].values[0]

    # Calcula o lucro total somando as produções multiplicadas pelo preço
    lucro_total = (producao_ano_kg * preco).sum()

    return f'R$ {round(lucro_total/1000000000, 2)} B'
  
def lucro_anual_state(cultura, ano, state, df, df_prices):
    # Define o nome da coluna de produção
    nome_coluna = f"{cultura.upper()} PRODUCTION (1000 tons)"

    # Filtra as linhas do ano e do estado desejado, mantendo as colunas relevantes
    producao_state = df.loc[
        (df['Year'] == ano) & (df['State Name'] == state),
        nome_coluna
    ]

    # Converte a produção de mil
    producao_kg = producao_state* 1_000_000

    # Obtém o preço por kg da cultura
    preco = df_prices.loc[
        df_prices['Crop'] == cultura,
        'Estimated Price per kg (R$)'
    ].values[0]

    # Calcula o lucro total
    lucro_total = (producao_kg * preco).sum()

    return f'R$ {round(lucro_total/1000000, 2)} M'
  
def lucro_anual_dist(cultura, ano, estado, df, df_prices):
    nome_coluna = f"{cultura.upper()} PRODUCTION (1000 tons)"

    df_filtered = df[(df['Year'] == ano) & (df['State Name'] == estado)][['Dist Name', nome_coluna]].copy()

    df_filtered[nome_coluna] = df_filtered[nome_coluna] * 1_000_000  # mil toneladas para kg

    preco = df_prices.loc[
        df_prices['Crop'].str.upper() == cultura.upper(),
        'Estimated Price per kg (R$)'
    ].values[0]

    df_filtered['Lucro (R$)'] = df_filtered[nome_coluna] * preco

    return df_filtered[['Dist Name', 'Lucro (R$)']]


import plotly.graph_objects as go

def grafico_producao_estado_ano_cultura(df, df_prices, ano, estado, cultura):
    producao_col = f"{cultura.upper()} PRODUCTION (1000 tons)"

    # Dados de produção
    df_producao = df[(df['Year'] == ano) & (df['State Name'] == estado)][['Dist Name', producao_col]].copy()

    # Dados de lucro
    df_lucro = lucro_anual_dist(cultura, ano, estado, df, df_prices)

    # Garantir merge correto por distrito
    df_merged = pd.merge(df_producao, df_lucro, on='Dist Name')

    fig = go.Figure()

    # Produção - Barras (eixo Y1)
    fig.add_trace(go.Bar(
        x=df_merged['Dist Name'],
        y=df_merged[producao_col],
        name='Produção (mil toneladas)',
        yaxis='y1'
    ))

    # Lucro - Linha (eixo Y2)
    fig.add_trace(go.Scatter(
        x=df_merged['Dist Name'],
        y=df_merged['Lucro (R$)'],
        name='Lucro (R$)',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='red')
    ))

    fig.update_layout(
        title=f'{cultura} - Produção e Lucro por Distrito em {estado} ({ano})',
        xaxis=dict(title='Distrito'),
        yaxis=dict(
            title='Produção (mil toneladas)',
            side='left'
        ),
        yaxis2=dict(
            title='Lucro (R$)',
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.01, y=0.99),
        bargap=0.3
    )

    return fig


def machine_learing(df, estado, cultura):
  df_filtered = df[df['State Name'] == estado]
  column_crop = ['Dist Name']
  cultura = cultura.upper()
  for column in list(df.columns.values):
        if cultura in column:
          column_crop.append(column)
        else:
          continue

  df_filtered = df_filtered[column_crop]

  df_filtered.drop('Dist Name', axis = 1, inplace = True)

  x_train, x_test, y_train, y_test = train_test_split(df_filtered.drop(f'{cultura} PRODUCTION (1000 tons)', axis = 1), df_filtered[f'{cultura} PRODUCTION (1000 tons)'], test_size = 11/88, random_state = 19)

  modelx = RandomForestRegressor(max_depth=20, min_samples_leaf=4,min_samples_split = 10)

  modelx.fit(x_train, y_train)

  y_pred = modelx.predict(x_test)

  mse = mean_squared_error(y_test, y_pred)

  return y_pred, mse

def previsao_classificacao(df_crops, estado, cultura):
    # 1. Filtrar estado e cultura
    df_estado = df_crops[df_crops['State Name'] == estado].copy()
    cultura = cultura.upper()

    # 2. Separar treino (2010-2017)
    df_treino = df_estado[df_estado['Year'] < 2018].copy()
    df_treino.sort_values(by=['Year'], ascending=True, inplace=True)

    # 3. Features e target
    features = [f'{cultura} AREA (1000 ha)', f'{cultura} YIELD (Kg per ha)']
    target = f'{cultura} PRODUCTION (1000 tons)'

    # 4. Remover linhas sem dados
    df_treino = df_treino.dropna(subset=features + [target])

    if df_treino.empty:
        print(f"⚠️ Não há dados suficientes para treinar {cultura} em {estado}.")
        return pd.DataFrame()

    # 5. Criar coluna de aumento ou diminuição
    df_treino['Aumento_Diminuicao'] = (df_treino[target].shift(-1) > df_treino[target]).astype(int)

    # 6. Remover a última linha (sem o próximo ano para comparação)
    df_treino = df_treino[:-1]

    # 7. Treinar modelo
    X_train = df_treino[features]
    y_train = df_treino['Aumento_Diminuicao']

    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    # 8. Criar "pseudo-dados" de 2018: média por distrito
    df_media = df_treino.groupby('Dist Name')[features].mean().reset_index()
    df_2018 = df_media.copy()

    # 9. Previsão para 2018
    X_2018 = df_2018[features]
    y_pred = model.predict(X_2018)

    # 10. Resultado final
    df_resultado = df_2018[['Dist Name']].copy()
    df_resultado['Aumento/Diminuicao (2018)'] = y_pred
    df_resultado['Ano'] = 2018

    # 11. Exibir métricas de desempenho
    y_train_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_train_pred)
    cm = confusion_matrix(y_train, y_train_pred)
    cr = classification_report(y_train, y_train_pred)

    print(f"**Desempenho do Modelo**:")
    print(f"- Acurácia: {accuracy:.2f}")
    print(f"- Matriz de Confusão:\n{cm}")
    print(f"- Relatório de Classificação:\n{cr}")

    return df_resultado

### Filtrando o ano e a cultura - gerar um dataframe 

def filter_ano_crop(df, ano, cultura, df_geo = df_states):
   df_ano = df[df['Year'] == ano]

   df_ano_crop = df_ano.groupby('State Name')[f'{cultura} PRODUCTION (1000 tons)'].sum()

   df_ano_crop = df_ano_crop.reset_index()

   df_geo.sort_values(by='State Name', inplace=True)

   df_geo.reset_index(inplace = True)

   df_geo.drop(columns=['index'], inplace=True)
   
   df_ano_crop['lat'] = df_geo['Latitude']

   df_ano_crop['lon'] = df_geo['Longitude']

   df_ano_crop.dropna(inplace = True)


   return df_ano_crop

## Função que gera o mapa

def plotly_map(df, crop):
    
    fig = px.scatter_map(
        data_frame = df,
        lat = 'lat',
        lon = 'lon',
        size = f'{crop} PRODUCTION (1000 tons)',
        hover_data = 'State Name',
        zoom = 3
    )
    
    return fig 









