import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprime avisos do TensorFlow

import random
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # Salva os gráficos em arquivo sem abrir janela
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber
from tensorflow.keras.initializers import Constant
from tensorflow.keras import backend as K
import gc

# Sementes avaliadas: o resultado de uma única semente varia muito (poucos dados),
# então o valor esperado real é a média entre várias sementes
SEMENTES = range(10)
PASTA_SAIDA = "resultados_modelo_final_BRS_Atalanta"

# Cores dos gráficos
AZUL, LARANJA = "#2a78d6", "#eb6834"
TEXTO, TEXTO_2, GRADE = "#0b0b0b", "#52514e", "#e5e4e0"

def fixar_sementes(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

tf.config.experimental.enable_op_determinism()

# Função para liberar memória
def limpar_memoria(model):
    K.clear_session()  # Libera memória da GPU/CPU associada ao modelo atual
    del model          # Remove a referência ao modelo
    gc.collect()       # Força a coleta de lixo

# Função para criar e compilar o modelo
def criar_modelo(n_entradas, combinacao, media_y):
    _, _, dropout_rate, delta, reg, alpha, lr = combinacao
    model = Sequential([
        Dense(10, kernel_regularizer=l2(reg), input_shape=(n_entradas,)),
        LeakyReLU(alpha=alpha),
        Dropout(dropout_rate),
        Dense(10, kernel_regularizer=l2(reg)),
        LeakyReLU(alpha=alpha),
        Dropout(dropout_rate),
        Dense(10, kernel_regularizer=l2(reg)),
        LeakyReLU(alpha=alpha),
        Dropout(dropout_rate),
        # Bias inicia na média do alvo (~73 dias): sem isso a rede começa prevendo ~0
        # e não converge em 300 épocas, pois o gradiente da Huber é limitado por delta
        Dense(1, activation='linear', bias_initializer=Constant(media_y))
    ])
    optimizer = Adam(learning_rate=lr)
    loss_fn = Huber(delta=delta)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mean_absolute_error'])
    return model

# Função para treinar o modelo em um único fold e devolver as previsões do conjunto de teste
def executar_fold(train_idx, test_idx, combinacao, X, y):
    batch_size, epochs = combinacao[:2]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = criar_modelo(X_train.shape[1], combinacao, np.mean(y_train))

    # O fold de teste entra como validação só para registrar as curvas; não altera o treino
    historico = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                          validation_data=(X_test, y_test), verbose=0)

    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test, verbose=0).flatten()

    # Limpar memória
    limpar_memoria(model)
    return y_pred, historico.history

# Função para executar todos os folds e devolver as previsões fora do fold de cada amostra
def executar_todos_folds(X, y, combinacao, kf):
    y_pred = np.zeros(len(y))
    historicos = []
    for train_idx, test_idx in kf.split(X):
        y_pred[test_idx], historico = executar_fold(train_idx, test_idx, combinacao, X, y)
        historicos.append(historico)
    return y_pred, historicos

# Métricas sobre as previsões fora do fold de todas as amostras.
# MAE/RMSE "média folds" seguem a métrica do grid (calculadas por fold e depois promediadas)
def calcular_metricas(y, y_pred, kf):
    erro = y - y_pred
    mse = np.mean(erro ** 2)
    folds = [test_idx for _, test_idx in kf.split(y)]
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": np.mean(np.abs(erro)),
        "R2": r2_score(y, y_pred),
        "MAE média folds": np.mean([np.mean(np.abs(erro[f])) for f in folds]),
        "RMSE média folds": np.mean([np.sqrt(np.mean(erro[f] ** 2)) for f in folds]),
    }

# Gráfico previsto x real com as previsões do ensemble (barra = ±1 dp entre sementes)
def grafico_previsto_real(y, prev_media, prev_dp, metricas, caminho):
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    lim = [min(y.min(), prev_media.min()) - 2, max(y.max(), prev_media.max()) + 2]
    ax.plot(lim, lim, color=TEXTO_2, lw=1, ls="--", zorder=1)
    ax.text(lim[0] + 2.5, lim[0] + 1, "1:1", color=TEXTO_2, ha="left", va="bottom")
    ax.errorbar(y, prev_media, yerr=prev_dp, fmt="none", ecolor=AZUL, alpha=0.45, lw=1.2, zorder=2)
    ax.scatter(y, prev_media, s=55, color=AZUL, edgecolor="white", linewidth=1.5, zorder=3)
    ax.text(0.04, 0.96,
            f"R² = {metricas['R2']:.3f}\nRMSE = {metricas['RMSE']:.2f} dias\nMAE = {metricas['MAE']:.2f} dias",
            transform=ax.transAxes, va="top", color=TEXTO,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=GRADE))
    ax.set(xlim=lim, ylim=lim, xlabel="Real (dias até a floração)", ylabel="Previsto (dias até a floração)")
    ax.set_aspect("equal")
    ax.set_title("BRS_Atalanta: previsto × real\nensemble de 10 sementes, barras = ±1 dp entre sementes",
                 loc="left", color=TEXTO)
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)

# Curvas de perda e MAE de treino e validação (média ± 1 dp dos 5 folds × 10 sementes)
def grafico_curvas(historicos, caminho):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    paineis = [("loss", "Perda (Huber + L2)"), ("mean_absolute_error", "MAE (dias)")]
    for ax, (chave, titulo) in zip(axes, paineis):
        for prefixo, cor, rotulo in [("", AZUL, "Treino"), ("val_", LARANJA, "Validação")]:
            curvas = np.array([h[prefixo + chave] for h in historicos])
            media, dp = curvas.mean(axis=0), curvas.std(axis=0)
            epocas = np.arange(1, len(media) + 1)
            ax.fill_between(epocas, np.clip(media - dp, 0, None), media + dp, color=cor, alpha=0.15, lw=0)
            ax.plot(epocas, media, color=cor, lw=2, label=rotulo)
        ax.set(xlabel="Época", title=titulo, xlim=(1, len(media)))
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
    fig.suptitle(f"Curvas de treino e validação: média ± 1 dp de {len(historicos)} treinos (5 folds × 10 sementes)",
                 x=0.01, ha="left", color=TEXTO)
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)

# Box plot das métricas entre as sementes
def grafico_boxplot(metricas_df, caminho):
    paineis = [("MAE", "MAE (dias)"), ("RMSE", "RMSE (dias)"), ("MSE", "MSE (dias²)"), ("R2", "R²")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 4))
    rng = np.random.default_rng(0)
    for ax, (chave, titulo) in zip(axes, paineis):
        valores = metricas_df[chave].values
        ax.boxplot(valores, widths=0.5, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=AZUL + "26", edgecolor=AZUL, lw=1.5),
                   medianprops=dict(color=AZUL, lw=2),
                   whiskerprops=dict(color=AZUL, lw=1.5), capprops=dict(color=AZUL, lw=1.5))
        ax.scatter(1 + rng.uniform(-0.12, 0.12, len(valores)), valores, s=30, color=AZUL,
                   edgecolor="white", linewidth=1, zorder=3)
        ax.set_title(titulo)
        ax.set_xticks([])
    fig.suptitle(f"Distribuição das métricas entre {len(metricas_df)} sementes (cada ponto = uma semente)",
                 x=0.01, ha="left", color=TEXTO)
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": TEXTO_2, "axes.labelcolor": TEXTO, "axes.titlecolor": TEXTO,
    "xtick.color": TEXTO_2, "ytick.color": TEXTO_2,
    "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "savefig.dpi": 200, "figure.facecolor": "white",
})

# Definir o diretório de trabalho
os.chdir("C:/Users/Igorb/Documents/Trabalho-Embrapa/Rede_neural_Gutenberg")
os.makedirs(PASTA_SAIDA, exist_ok=True)

# Carregar o dataset
dados = pd.read_csv("dadosFenolTratados.csv")

# Filtrar dados para o genótipo "BRS_Atalanta"
dados_atalanta = dados[dados['cultivar'] == "BRS_Atalanta"]

# Definir variáveis independentes e variável alvo
variaveis_independentes = [
    "Tmin_min_FLO", "Tmax_max_FLO", "Tmin_max_FLO", "Tmax_min_FLO",
    "Tmax_ac_FLO", "Tmin_ac_FLO", "Tmed_ac_FLO", "PP_ac_FLO",
    "Rad_ac_FLO", "DL_ac_FLO"
]
variavel_alvo = "ndias_FLO"

# Separar os dados
X = dados_atalanta[variaveis_independentes].values
y = dados_atalanta[variavel_alvo].values.astype(float)

# Normalizar as variáveis independentes
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Hiperparâmetros ajustados a partir da melhor linha do mini grid
# (resultadosMini_grid_Fenol_multigenotipos/BRS_Atalanta.csv):
# Batch Size 1 -> 4, Dropout 0.13 -> 0.0, L2 0.000464 -> 0.002154, demais mantidos.
# Com os parâmetros originais e sem o bias na média, o MAE esperado era ~3.45 ± 0.72 (12 sementes).
nomes_hiperparametros = ["Batch Size", "Epochs", "Dropout", "Huber Delta",
                         "L2 Regularizer", "Leaky ReLU Alpha", "Learning Rate"]
combinacao = (
    4,                       # Batch Size
    300,                     # Epochs
    0.0,                     # Dropout
    1.0749999999999997,      # Huber Delta
    0.002154434690031882,    # L2 Regularizer
    0.1,                     # Leaky ReLU Alpha
    0.01,                    # Learning Rate
)

# Configurar a validação cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=123)

print(f"BRS_Atalanta: {len(dados_atalanta)} registros | Hiperparâmetros {combinacao}")

resultados, previsoes, historicos = [], [], []
for seed in SEMENTES:
    fixar_sementes(seed)
    y_pred, historicos_seed = executar_todos_folds(X, y, combinacao, kf)
    metricas = calcular_metricas(y, y_pred, kf)
    print(f"Semente {seed}: " + " | ".join(f"{k} = {v:.4f}" for k, v in metricas.items()))
    resultados.append({"Semente": seed, **metricas})
    previsoes.append(y_pred)
    historicos.extend(historicos_seed)

metricas_df = pd.DataFrame(resultados)
previsoes = np.array(previsoes)

# Ensemble: média das previsões de todas as sementes (modelo final mais estável)
prev_ensemble, prev_dp = previsoes.mean(axis=0), previsoes.std(axis=0)
metricas_ensemble = calcular_metricas(y, prev_ensemble, kf)

# Arquitetura (modelo de exemplo criado só para o resumo)
linhas_resumo = []
modelo_exemplo = criar_modelo(X.shape[1], combinacao, np.mean(y))
modelo_exemplo.summary(print_fn=lambda linha, **kwargs: linhas_resumo.append(linha))
limpar_memoria(modelo_exemplo)

colunas = list(metricas_ensemble)
resumo_metricas = pd.DataFrame({
    "Média entre sementes": metricas_df[colunas].mean(),
    "Desvio padrão": metricas_df[colunas].std(),
    "Mínimo": metricas_df[colunas].min(),
    "Máximo": metricas_df[colunas].max(),
    f"Ensemble ({len(SEMENTES)} sementes)": pd.Series(metricas_ensemble),
})

relatorio = "\n".join([
    "=== Arquitetura ===",
    *linhas_resumo,
    f"Entrada: {X.shape[1]} variáveis climáticas normalizadas com MinMaxScaler ({', '.join(variaveis_independentes)})",
    f"Alvo: {variavel_alvo} (dias até a floração) | {len(y)} registros do BRS_Atalanta",
    "Camadas ocultas: 3 × Dense(10) + LeakyReLU + Dropout | Saída: Dense(1) linear, bias inicial = média do alvo no treino",
    "",
    "=== Treino ===",
    f"Otimizador: Adam (learning rate = {combinacao[6]}) | Perda: Huber (delta = {combinacao[3]})",
    f"Regularização L2 = {combinacao[4]} nos pesos das camadas Dense ocultas",
    f"{combinacao[1]} épocas, batch size {combinacao[0]}, sem early stopping",
    "Validação cruzada: KFold com 5 folds, shuffle=True, random_state=123",
    f"Sementes: {list(SEMENTES)} (pesos iniciais e ordem dos lotes)",
    "",
    "=== Hiperparâmetros ===",
    *[f"{nome}: {valor}" for nome, valor in zip(nomes_hiperparametros, combinacao)],
    "",
    "=== Métricas por semente (previsões fora do fold das 23 amostras) ===",
    metricas_df.to_string(index=False, float_format="%.4f"),
    "",
    "=== Resumo das métricas ===",
    resumo_metricas.to_string(float_format="%.4f"),
])
print("\n" + relatorio)

# Salvar relatório, tabelas e gráficos
with open(os.path.join(PASTA_SAIDA, "relatorio.txt"), "w", encoding="utf-8") as arquivo:
    arquivo.write(relatorio + "\n")
metricas_df.to_csv(os.path.join(PASTA_SAIDA, "metricas_por_semente.csv"), index=False)
resumo_metricas.to_csv(os.path.join(PASTA_SAIDA, "resumo_metricas.csv"))
pd.DataFrame({
    "ensaio": dados_atalanta["ensaio"].values,
    "real": y,
    "previsto_ensemble": prev_ensemble,
    "dp_entre_sementes": prev_dp,
    **{f"previsto_semente_{s}": p for s, p in zip(SEMENTES, previsoes)},
}).to_csv(os.path.join(PASTA_SAIDA, "previsoes.csv"), index=False)

grafico_previsto_real(y, prev_ensemble, prev_dp, metricas_ensemble,
                      os.path.join(PASTA_SAIDA, "previsto_x_real.png"))
grafico_curvas(historicos, os.path.join(PASTA_SAIDA, "curvas_perda_validacao.png"))
grafico_boxplot(metricas_df, os.path.join(PASTA_SAIDA, "boxplot_metricas.png"))
print(f"\nArquivos salvos em: {PASTA_SAIDA}/")
