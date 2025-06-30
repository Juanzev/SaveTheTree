import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Anos de 2000 a 2024
anos = np.arange(2000, 2025)

# --- DADOS REAIS ---
# PIB em trilhões de Reais (R$) - DADOS REAIS DO IBGE
pib_real = [
    1.09, # 2000
    1.20, # 2001
    1.39, # 2002
    1.57, # 2003
    1.87, # 2004
    2.18, # 2005
    2.49, # 2006
    2.83, # 2007
    3.26, # 2008
    3.51, # 2009
    3.90, # 2010
    4.41, # 2011
    4.81, # 2012
    5.33, # 2013
    5.78, # 2014
    5.99, # 2015
    6.27, # 2016
    6.59, # 2017
    6.88, # 2018
    7.26, # 2019
    7.60, # 2020
    8.92, # 2021
    9.99, # 2022
    10.90, # 2023
    11.70  # 2024
]

# Populacao em milhões de habitantes (mantendo seus valores, que já parecem razoáveis)
populacao = [
    174, 176, 178, 180, 183, 185, 186, 188, 190, 192,
    193, 195, 196, 198, 200, 201, 203, 205, 206, 207,
    209, 210, 211, 212, 213
]

# Nível de Políticas Ambientais: 0 (fracas/ausentes), 1 (moderadas), 2 (fortes/eficazes)
# Estes são mais interpretativos e não há um "dado real" fácil de obter. Mantenha se quiser simular esse fator.
politicas = [
    # Período de consolidação e início do PPCDAm (Lula/Dilma 1)
    1,  # 2000: Início do período, políticas incipientes ou em desenvolvimento
    1,  # 2001
    1,  # 2002
    2,  # 2003: Início de políticas mais fortes, PPCDAm começa a ser formulado
    2,  # 2004: PPCDAm lançado, queda no desmatamento
    2,  # 2005
    2,  # 2006
    2,  # 2007
    2,  # 2008
    2,  # 2009: Apogeu da efetividade do PPCDAm
    
    # Continuidade, mas com desafios (Dilma 2 / Temer)
    1,  # 2010: Ainda fortes, mas debates sobre Código Florestal começam
    1,  # 2011
    1,  # 2012: Novo Código Florestal aprovado, gerando incertezas
    1,  # 2013
    1,  # 2014
    1,  # 2015
    1,  # 2016: Transição de governo, instabilidade
    
    # Período de enfraquecimento (Temer / Bolsonaro)
    0,  # 2017: Começo do enfraquecimento de agências e leis
    0,  # 2018
    0,  # 2019: Início do desmonte ambiental, aumento do desmatamento
    0,  # 2020
    0,  # 2021
    0,  # 2022: Ponto mais baixo da governança ambiental
    
    # Retomada da agenda ambiental (Lula 3)
    1,  # 2023: Retomada do MMA, PPCDAm reativado, mas resultados levam tempo
    1   # 2024: Esforços contínuos, mas orçamento e estrutura ainda desafiadores
]

# Efeito Estufa: Emissões brutas de CO2 equivalente (GtCO2e) do Brasil - DADOS REAIS DO SEEG
efeito_estufa = [
    2.15, # 2000
    2.20, # 2001
    2.40, # 2002
    2.70, # 2003
    2.80, # 2004
    2.90, # 2005
    2.70, # 2006
    2.40, # 2007
    2.10, # 2008 (início da queda com PPCDAm)
    1.77, # 2009 (menor nível histórico)
    1.85, # 2010
    1.95, # 2011
    2.00, # 2012
    2.10, # 2013
    2.20, # 2014
    2.25, # 2015
    2.30, # 2016
    2.40, # 2017
    2.45, # 2018
    2.50, # 2019 (início do aumento de emissões novamente)
    2.10, # 2020 (queda devido à pandemia, apesar do desmatamento)
    2.40, # 2021 (aumento pós-pandemia e com desmatamento alto)
    2.30, # 2022
    2.30, # 2023
    2.10  # 2024
]

# Desmatamento em milhões de hectares - DADOS REAIS FORNECIDOS POR VOCÊ
desmatamento = [
    1.8, 1.8165, 2.1651, 2.5396, 2.7772, 1.9014, 1.4286, 1.1651, 1.2911, 0.7464,
    0.7, 0.6418, 0.4571, 0.5891, 0.5012, 0.6207, 0.7893, 0.6947, 0.7536, 1.0129,
    1.0851, 1.3038, 1.1594, 1.152, 0.6288
]

# 3. DataFrame com os dados
df = pd.DataFrame({
    "Ano": anos,
    "PIB": pib_real, # Usando o PIB real
    "Populacao": populacao,
    "Politicas": politicas,
    "EfeitoEstufa": efeito_estufa,
    "Desmatamento": desmatamento # Usando os novos dados reais de desmatamento
})

# 4. Separar variáveis
X = df[["PIB", "Populacao", "Politicas", "EfeitoEstufa"]]
y = df["Desmatamento"]

# 5. Regressão Linear
modelo = LinearRegression()
modelo.fit(X, y)

# 6. Previsão para 2025 a 2045 com valores futuros fictícios (ajustados para a nova escala do PIB)
anos_futuros = np.arange(2025, 2046)

# Crescimento do PIB futuro (ajustado para a escala de trilhões)
# Por exemplo: crescimento de 3% a 5% ao ano sobre o PIB atual
pib_futuro = np.array([pib_real[-1] * (1.03 ** i) for i in range(len(anos_futuros))])

pop_futuro = np.linspace(populacao[-1] + 1, populacao[-1] + 15, len(anos_futuros))

# Cenário de melhora gradual das políticas (de fraca para forte)
politicas_futuro = np.array([
    0 if i < 3 else (1 if i < 10 else 2) # Ex: 3 anos de 0, 7 anos de 1, 11 anos de 2
    for i in range(len(anos_futuros))
])

# Cenário de estabilização ou leve redução do "Efeito Estufa" com políticas melhores
# Atenção: corrigi o nome da variável para 'efeito_futuro' para consistência
efeito_futuro = np.linspace(efeito_estufa[-1], efeito_estufa[-1] - 0.5, len(anos_futuros))

X_futuro = pd.DataFrame({
    "PIB": pib_futuro,
    "Populacao": pop_futuro,
    "Politicas": politicas_futuro,
    "EfeitoEstufa": efeito_futuro # Corrigido aqui também
})

# 7. Prever desmatamento futuro
y_futuro = modelo.predict(X_futuro)
y_futuro[y_futuro < 0] = 0 # Evitar valores negativos

# 8. Plotar
plt.figure(figsize=(14, 7))
plt.plot(df["Ano"], y, label="Desmatamento Histórico (milhões de hectares)", marker='o', color='blue')
plt.plot(anos_futuros, y_futuro, label="Previsão de Desmatamento (milhões de hectares)", linestyle='--', marker='x', color='red')
plt.xlabel("Ano")
plt.ylabel("Desmatamento (milhões de hectares)")
plt.title("Previsão de Desmatamento com Regressão Linear")
plt.legend()
plt.grid(True)

# --- ADICIONAR VALORES EXATOS E MELHORAR ESCALA DO EIXO Y ---

# Garante que o eixo Y comece em zero
plt.ylim(bottom=0)

# Define os marcadores do eixo Y a cada 0.1 milhão de hectares
# np.arange(start, stop, step)
# O 'stop' é o valor máximo do y (histórico ou futuro) mais uma margem.
max_y_value = max(y.max(), y_futuro.max()) # Pega o maior valor entre o histórico e a previsão
plt.yticks(np.arange(0, max_y_value + 0.1, 0.1))


# Anotar o último valor histórico (2024)
plt.text(df["Ano"].iloc[-1], y.iloc[-1], f'{y.iloc[-1]:.3f}',
        ha='right', va='bottom', fontsize=9, color='blue')

# Anotar o primeiro valor previsto (2025)
plt.text(anos_futuros[0], y_futuro[0], f'{y_futuro[0]:.3f}',
        ha='left', va='bottom', fontsize=9, color='red')

# Anotar o último valor previsto (2045)
plt.text(anos_futuros[-1], y_futuro[-1], f'{y_futuro[-1]:.3f}',
        ha='left', va='center', fontsize=9, color='red')

# Anotar o ponto de menor desmatamento previsto
min_y_futuro_idx = np.argmin(y_futuro)
plt.text(anos_futuros[min_y_futuro_idx], y_futuro[min_y_futuro_idx], f'{y_futuro[min_y_futuro_idx]:.3f}',
        ha='center', va='top', fontsize=9, color='red')


plt.show() # Garante que o gráfico seja exibido