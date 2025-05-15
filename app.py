import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração da página
st.set_page_config(
    page_title="Previsão de Rendimento Escolar",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carrega o modelo treinado
try:
    modelo = joblib.load("modelo_rendimento_escolar.pkl")
except FileNotFoundError:
    st.error("Erro: Modelo não encontrado. Verifique o arquivo 'modelo_rendimento_escolar.pkl'.")
    st.stop()

# Estilização CSS personalizada
st.markdown("""
    <style>
    .main {background-color: #f9f9fb;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px;}
    .stSelectbox {background-color: #ffffff; border-radius: 5px;}
    .title {font-size: 2.5em; color: #333;}
    .sidebar .sidebar-content {background-color: #e6f3ff;}
    </style>
    """, unsafe_allow_html=True)

# Sidebar com informações e configurações
with st.sidebar:
    st.header("ℹ️ Sobre o Aplicativo")
    st.markdown("""
    Este aplicativo utiliza um modelo de **machine learning** para prever a nota média de estudantes com base em características pessoais e contextuais.
    - **Modelo**: Regressão treinada com dados educacionais.
    - **Dados**: Incluem gênero, etnia, nível educacional dos pais, tipo de almoço e curso preparatório.
    """)
    st.markdown("[Saiba mais sobre o projeto](#)", unsafe_allow_html=True)

# Título e introdução
st.markdown('<div class="title">📚 Previsão de Rendimento Escolar</div>', unsafe_allow_html=True)
st.markdown("""
Bem-vindo! Este aplicativo prevê a **nota média** de um estudante com base em informações sobre seu estilo de vida e contexto familiar.  
Preencha os campos abaixo e clique em "Prever" para ver o resultado.
""")

# Explicação do modelo (expansível)
with st.expander("🔍 Como funciona o modelo?"):
    st.markdown("""
    O modelo foi treinado com um conjunto de dados educacionais que inclui:
    - **Gênero**: Identificação de gênero do estudante.
    - **Grupo étnico**: Categoria étnica auto-declarada.
    - **Nível de educação dos pais**: Maior nível educacional alcançado pelos pais.
    - **Tipo de almoço**: Indica se o estudante recebe almoço gratuito/reduzido ou padrão.
    - **Curso de preparação**: Participação em curso preparatório para testes.
    
    O modelo utiliza essas variáveis para prever a **nota média** em uma escala de 0 a 300.
    """)

# Formulário de entrada de dados
st.subheader("📝 Insira os Dados")
with st.form(key="input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        genero = st.selectbox(
            "Gênero",
            ["female", "male"],
            help="Selecione o gênero do estudante."
        )
        etnia = st.selectbox(
            "Grupo étnico",
            ["group A", "group B", "group C", "group D", "group E"],
            help="Selecione o grupo étnico."
        )
        nivel_pais = st.selectbox(
            "Nível de educação dos pais",
            ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"],
            help="Maior nível educacional dos pais."
        )
    
    with col2:
        almoco = st.selectbox(
            "Tipo de almoço",
            ["standard", "free/reduced"],
            help="Indica se o almoço é padrão ou gratuito/reduzido."
        )
        preparacao = st.selectbox(
            "Curso de preparação para o teste",
            ["none", "completed"],
            help="Indica se o estudante fez curso preparatório."
        )
    
    submit_button = st.form_submit_button("🔮 Prever Nota Média")

# Processamento da previsão
if submit_button:
    # Monta o DataFrame de entrada
    entrada = pd.DataFrame({
        "gender": [genero],
        "race/ethnicity": [etnia],
        "parental level of education": [nivel_pais],
        "lunch": [almoco],
        "test preparation course": [preparacao]
    })

    # Codificação como no treinamento
    entrada_codificada = pd.get_dummies(entrada)
    colunas_modelo = modelo.feature_names_in_
    for col in colunas_modelo:
        if col not in entrada_codificada.columns:
            entrada_codificada[col] = 0
    entrada_codificada = entrada_codificada[colunas_modelo]

    # Previsão
    previsao = modelo.predict(entrada_codificada)[0]
    
    # Exibição do resultado
    st.success(f"🎉 **Nota média prevista**: {previsao:.2f} (em uma escala de 0 a 300)")
    
    # Dados fornecidos
    st.markdown("### 📋 Dados Fornecidos")
    st.dataframe(entrada.style.set_properties(**{'background-color': '#e6f3ff', 'border-radius': '5px'}))

    # Gráfico de comparação
    st.markdown("### 📊 Visualização do Resultado")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_style("whitegrid")
    cores = sns.color_palette("Blues")
    
    # Supondo uma nota média de referência (exemplo: 200)
    media_referencia = 200
    barras = ax.bar(["Nota Prevista", "Média de Referência"], [previsao, media_referencia], color=[cores[2], cores[0]])
    ax.set_ylim(0, 350)
    ax.set_ylabel("Pontuação", fontsize=12)
    ax.set_title("Comparação da Previsão", fontsize=14)
    
    # Adiciona valores nas barras
    for barra in barras:
        height = barra.get_height()
        ax.text(barra.get_x() + barra.get_width() / 2, height + 10, f"{height:.2f}", ha='center', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)

# Rodapé
st.markdown("---")
st.markdown("""
**Desenvolvido por Vicente**  
Projeto de aprendizado de máquina para previsão de rendimento escolar.  
📧 Contato: vicente@example.com | 🌐 [GitHub](#)  
""", unsafe_allow_html=True)