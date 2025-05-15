
import streamlit as st
import pandas as pd
import joblib

# Carrega o modelo treinado
modelo = joblib.load("modelo_rendimento_escolar.pkl")

st.title("Previsão de Rendimento Escolar")

st.write("Preencha os dados abaixo para prever a nota média de um estudante.")

# Interface simples com base em colunas do dataset original
genero = st.selectbox("Gênero", ["female", "male"])
etnia = st.selectbox("Grupo étnico", ["group A", "group B", "group C", "group D", "group E"])
nivel_pais = st.selectbox("Nível de educação dos pais", [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
])
almoco = st.selectbox("Tipo de almoço", ["standard", "free/reduced"])
preparacao = st.selectbox("Curso de preparação para o teste", ["none", "completed"])

# Cria um dataframe com os dados de entrada
entrada = pd.DataFrame({
    "gender": [genero],
    "race/ethnicity": [etnia],
    "parental level of education": [nivel_pais],
    "lunch": [almoco],
    "test preparation course": [preparacao]
})

# Codifica as variáveis categóricas como no treino
entrada_codificada = pd.get_dummies(entrada)
colunas_modelo = modelo.feature_names_in_
for col in colunas_modelo:
    if col not in entrada_codificada.columns:
        entrada_codificada[col] = 0
entrada_codificada = entrada_codificada[colunas_modelo]

# Faz a previsão
if st.button("Prever Nota Média"):
    previsao = modelo.predict(entrada_codificada)
    st.success(f"Nota média prevista: {previsao[0]:.2f}")
