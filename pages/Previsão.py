import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from pages import *

# Configuração básica da página
st.set_page_config(
    page_title="Previsão Agrícola",
    page_icon="🔮",
    layout="wide"
)

# Mapeamento de culturas com nomes compostos
MAPEAMENTO_CULTURAS = {
    'FINGER': 'FINGER MILLET',
    'KHARIF': 'KHARIF SORGHUM',
    'MINOR': 'MINOR PULSES',
    'PEARL': 'PEARL MILLET',
    'RABI': 'RABI SORGHUM',
    'RAPESEED': 'RAPESEED AND MUSTARD'
}

# Carregar dados
@st.cache_data
def load_data():
    file_id = '1O4M5uUSGTYtDnbhHSF1iXznl1OjNDC5R'
    excel_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx"
    
    df = pd.read_excel(excel_url, sheet_name='Crops_data')
    df = df.iloc[:, 1:75]
    df = df.dropna()
    
    # Correções de nomes de distritos
    df['Dist Name'] = df['Dist Name'].replace({
        'Kadap YSR': 'Kadapa YSR',
        'Gurdspur': 'Gurdaspur',
        'Ludhana': 'Ludhiana',
        'Madi': 'Mandi',
        'Korput': 'Koraput'
    })
    
    return df

df = load_data()

# Sidebar com filtros e informações
with st.sidebar:
 
    st.header("Filtro de Cultura")
    
    # Filtro de Cultura - usando nomes simplificados
    colunas_culturas = [col for col in df.columns if 'PRODUCTION' in col]
    culturas_simplificadas = sorted(list(set(col.split(' ')[0] for col in colunas_culturas)))
    
    # Filtra apenas culturas que existem no mapeamento ou têm dados completos
    culturas_disponiveis = []
    for cultura in culturas_simplificadas:
        nome_real = MAPEAMENTO_CULTURAS.get(cultura, cultura)
        if (f'{nome_real} AREA (1000 ha)' in df.columns and 
            f'{nome_real} YIELD (Kg per ha)' in df.columns and 
            f'{nome_real} PRODUCTION (1000 tons)' in df.columns):
            culturas_disponiveis.append(cultura)
    
    cultura_selecionada = st.selectbox(
        "Selecione a cultura:",
        culturas_disponiveis,
        index=culturas_disponiveis.index('BARLEY') if 'BARLEY' in culturas_disponiveis else 0
    )
    
    st.markdown("---")
    st.header("Filtro de Estado")
    
    # Filtro de Estado
    estados = sorted(df['State Name'].unique())
    estado_selecionado = st.selectbox(
        "Selecione o estado:",
        estados,
        index=estados.index('Madhya Pradesh') if 'Madhya Pradesh' in estados else 0
    )
    
    st.markdown("---")
    st.header("🔗 Links")
    st.markdown("[Instagram da Liga](https://www.instagram.com/ligadsunicamp/)")
    st.markdown("[GitHub do Projeto](https://github.com)")
    
    st.markdown("---")
    st.header("👥 Autores")
    st.markdown("- [Gabriel Honda](https://www.linkedin.com/in/gabriel-honda-192097306/)")
    st.markdown("- [Lucas Pimentel](https://www.linkedin.com/in/lucas-cl-pimentel/)")
    st.markdown("- [Maria Eduarda Gomes](https://www.linkedin.com/in/mariaeduardadesouzagomes/)")
    st.markdown("- [Tammy Kojima](https://www.linkedin.com/in/tammy-kojima-198425186/)")

# Função para obter o nome real da cultura
def obter_nome_real_cultura(cultura):
    return MAPEAMENTO_CULTURAS.get(cultura.upper(), cultura.upper())

# Função para cálculo de lucro
def lucro(y_pred, cultura, df_prices):
    try:
        nome_real = obter_nome_real_cultura(cultura)
        producao_kg = y_pred * 1_000_000
        preco = df_prices.loc[df_prices['Crop'] == nome_real, 'Estimated Price per kg (R$)'].values[0]
        return producao_kg * preco
    except:
        return 0

# Função de machine learning para previsão de produção
def machine_learing(df, estado, cultura):
    try:
        nome_real = obter_nome_real_cultura(cultura)
        required_columns = [
            f'{nome_real} AREA (1000 ha)',
            f'{nome_real} YIELD (Kg per ha)',
            f'{nome_real} PRODUCTION (1000 tons)'
        ]
        
        if not all(col in df.columns for col in required_columns):
            return []
            
        df_filtered = df[df['State Name'] == estado]
        if df_filtered.empty:
            return []
            
        column_crop = ['Dist Name'] + required_columns
        
        df_filtered = df_filtered[column_crop].dropna()
        if df_filtered.empty:
            return []
            
        df_filtered = df_filtered.drop('Dist Name', axis=1)

        x_train, x_test, y_train, y_test = train_test_split(
            df_filtered.drop(f'{nome_real} PRODUCTION (1000 tons)', axis=1),
            df_filtered[f'{nome_real} PRODUCTION (1000 tons)'],
            test_size=11/88,
            random_state=19
        )

        modelx = RandomForestRegressor(max_depth=20, min_samples_leaf=4, min_samples_split=10)
        modelx.fit(x_train, y_train)
        y_pred = modelx.predict(x_test)

        return y_pred
    except:
        return []

# Função de previsão otimizada
def previsao_classificacao(df_crops, estado, cultura):
    try:
        nome_real = obter_nome_real_cultura(cultura)
        required_columns = [
            f'{nome_real} AREA (1000 ha)',
            f'{nome_real} YIELD (Kg per ha)',
            f'{nome_real} PRODUCTION (1000 tons)'
        ]
        
        if not all(col in df_crops.columns for col in required_columns):
            return pd.DataFrame()
            
        df_estado = df_crops[df_crops['State Name'] == estado].copy()
        if df_estado.empty:
            return pd.DataFrame()
        
        # Dados de treino (2010-2017)
        df_treino = df_estado[df_estado['Year'] < 2018].copy()
        df_treino = df_treino.sort_values('Year')
        
        # Features e target
        features = [f'{nome_real} AREA (1000 ha)', f'{nome_real} YIELD (Kg per ha)']
        target = f'{nome_real} PRODUCTION (1000 tons)'
        
        # Limpeza e preparação
        df_treino = df_treino.dropna(subset=features + [target])
        if df_treino.empty:
            return pd.DataFrame()
        
        # Classificação (aumento/diminuição)
        df_treino['Classificacao'] = (df_treino[target].shift(-1) > df_treino[target])
        df_treino = df_treino[:-1]  # Remove último ano sem comparação
        
        # Modelo de previsão
        model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
        model.fit(df_treino[features], df_treino['Classificacao'])
        
        # Previsão para 2018 (média dos distritos)
        df_2018 = df_treino.groupby('Dist Name')[features].mean().reset_index()
        df_2018['Previsão'] = model.predict(df_2018[features])
        df_2018['Previsão'] = df_2018['Previsão'].map({True: 'Aumento', False: 'Diminuição'})
        
        return df_2018[['Dist Name', 'Previsão']]
    
    except Exception as e:
        return pd.DataFrame()

# Título da página
st.title("🔮 Previsão de Produção Agrícola")
st.divider()

# Executa a previsão automaticamente quando os filtros são selecionados
resultados = previsao_classificacao(df, estado_selecionado, cultura_selecionada)

# Exibe os resultados
if not resultados.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribuição das Previsões")
        fig = px.histogram(
            resultados,
            x='Previsão',
            color='Previsão',
            color_discrete_map={'Aumento': '#4CAF50', 'Diminuição': '#F44336'},
            category_orders={'Previsão': ['Aumento', 'Diminuição']}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("📈 Resumo das Previsões")
        contagem = resultados['Previsão'].value_counts()
        total = len(resultados)
        
        st.metric("Total de Distritos", total)
        if 'Aumento' in contagem:
            st.metric("Previsão de Aumento", 
                     contagem['Aumento'], 
                     f"{contagem['Aumento']/total:.1%}")
        if 'Diminuição' in contagem:
            st.metric("Previsão de Diminuição", 
                     contagem['Diminuição'], 
                     f"{contagem['Diminuição']/total:.1%}")
    
    st.subheader("📋 Lista de Distritos com Classificação")
    st.dataframe(
        resultados.sort_values('Previsão', ascending=False),
        column_config={
            "Dist Name": "Distrito",
            "Previsão": st.column_config.SelectboxColumn(
                "Classificação",
                options=["Aumento", "Diminuição"]
            )
        },
        hide_index=True,
        use_container_width=True
    )
    
else:
    st.warning("Não foi possível gerar previsões para os filtros selecionados. A cultura pode não ter dados suficientes.")
    
# Seção de Previsão de Produção e Lucro
st.markdown("---")
st.subheader("💰 Previsão de Produção e Lucro")

@st.cache_data
def load_prices():
    file_id = '1O4M5uUSGTYtDnbhHSF1iXznl1OjNDC5R'
    excel_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx"
    
    df_prices = pd.read_excel(excel_url, sheet_name='Prices per kg')
    
    cols_to_drop = ['Unnamed: 0'] + [i for i in range(0,10)]
    df_prices = df_prices.drop(columns=[col for col in cols_to_drop if col in df_prices.columns])
    df_prices = df_prices.dropna()
    
    def parse_price(price_str):
        try:
            if isinstance(price_str, str):
                if "''" in price_str:
                    parts = price_str.split("''")
                    if len(parts) > 1:
                        return float(parts[1])
                return float(price_str)
            return float(price_str)
        except:
            return None
    
    df_prices['Estimated Price per kg (R$)'] = df_prices['Estimated Price per kg (R$)'].apply(parse_price)
    df_prices = df_prices.dropna(subset=['Estimated Price per kg (R$)'])
    df_prices['Crop'] = df_prices['Crop'].str.upper()
    
    return df_prices

try:
    df_prices = load_prices()
    
    with st.spinner('Calculando previsões de produção...'):
        producoes_previstas = machine_learing(df, estado_selecionado, cultura_selecionada)
    
    if len(producoes_previstas) > 0:
        lucros = [lucro(prod, cultura_selecionada, df_prices) for prod in producoes_previstas]
        
        nome_real = obter_nome_real_cultura(cultura_selecionada)
        distritos = df[df['State Name'] == estado_selecionado]['Dist Name'].unique()
        
        df_resultados_producao = pd.DataFrame({
            'Distrito': distritos[:len(producoes_previstas)],
            'Produção Prevista (1000 tons)': producoes_previstas,
            'Lucro Estimado (R$)': lucros
        })
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Produção Média Prevista", 
                     f"{df_resultados_producao['Produção Prevista (1000 tons)'].mean():.2f} mil tons")
        
        with col2:
            st.metric("Lucro Médio Estimado", 
                     f"R$ {df_resultados_producao['Lucro Estimado (R$)'].mean():,.2f}")
        
        with col3:
            st.metric("Lucro Total Estimado", 
                     f"R$ {df_resultados_producao['Lucro Estimado (R$)'].sum():,.2f}")
        
        st.subheader(f"📈 Produção Prevista de {nome_real} por Distrito")
        fig_prod = px.bar(
            df_resultados_producao.sort_values('Produção Prevista (1000 tons)', ascending=False),
            x='Distrito',
            y='Produção Prevista (1000 tons)',
            color='Produção Prevista (1000 tons)',
            color_continuous_scale='greens'
        )
        st.plotly_chart(fig_prod, use_container_width=True)
        
        st.subheader(f"💵 Lucro Estimado por Distrito")
        fig_lucro = px.bar(
            df_resultados_producao.sort_values('Lucro Estimado (R$)', ascending=False),
            x='Distrito',
            y='Lucro Estimado (R$)',
            color='Lucro Estimado (R$)',
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig_lucro, use_container_width=True)
        
        st.subheader("📋 Detalhes por Distrito")
        st.dataframe(
            df_resultados_producao.sort_values('Lucro Estimado (R$)', ascending=False),
            column_config={
                "Produção Prevista (1000 tons)": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Lucro Estimado (R$)": st.column_config.NumberColumn(
                    format="R$ %.2f"
                )
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("Não foi possível calcular as previsões de produção. A cultura pode não ter dados suficientes.")

except Exception as e:
    st.error(f"Ocorreu um erro ao calcular as previsões de produção e lucro: {str(e)}")