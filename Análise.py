import streamlit as st
from data_analysis_and_functions import *
import plotly.express as px
from pages import *

st.set_page_config(
    page_title = 'Mapa Teste',
    page_icon = '🗺️',
    layout = 'wide'
)

# Título da página
st.title("📊 Análise Descritiva")
st.divider()

# filtro ano
lista_anos = list(df['Year'].dropna().unique())
lista_anos = [int(i) for i in lista_anos]

#filtro estado
lista_estados = list(df['State Name'].dropna().unique())

st.sidebar.header("Filtro de Cultura")
crop = st.sidebar.selectbox(
    label = 'Selecione a cultura',
    options = crops
)
st.sidebar.markdown("---")
# Filtro de ano 
st.sidebar.header("Filtro de Ano")
ano = st.sidebar.selectbox(
        label = 'Selecione o ano:',
        options = lista_anos
)
st.sidebar.markdown("---")
# Filtro de estado
st.sidebar.header("Filtro de Estado")
estado = st.sidebar.selectbox(
        label = 'Selecione o estado:',
    options = lista_estados
)
st.sidebar.markdown("---")

with st.sidebar: 
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

# página padrão 
col1, col2 = st.columns([1,1])

with col1.container(border = True):
    
    st.subheader(f'{crop} em {ano}')
    data = filter_ano_crop(df, ano, crop)

    # Plotando o mapa
    st.plotly_chart(plotly_map(data, crop), config={'displayModeBar':False})
    
    # Métricas
    st.metric(
        label = "Produção porcentagem yield", 
        value = production_yield(df,ano,crop)
    )
    
    st.metric(
        label = "Lucro anual", 
        value = lucro_anual_total(crop, ano, df, df_prices)
    )
    
    
with col2.container(border = True):
    
    
    # Plotando o mapa
    st.plotly_chart(grafico_cultura_estado(df, estado, crop))
    
    
    st.metric(
        label = "Lucro por Estado", 
        value = lucro_anual_state(crop, ano, estado, df, df_prices)
    )
    
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    

with st.container(border = True):
    # Plotando o mapa
    st.plotly_chart(grafico_producao_estado_ano_cultura(df, df_prices, ano, estado, crop), config={'displayModeBar':False})