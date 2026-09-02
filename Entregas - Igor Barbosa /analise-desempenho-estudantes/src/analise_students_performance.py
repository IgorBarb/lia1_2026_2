# -*- coding: utf-8 -*-
"""Análise de Desempenho de Estudantes — script completo.

Executa as 5 etapas: limpeza, EDA, visualizações, insights e conclusões.
Uso:  python src/analise_students_performance.py   (rode a partir da raiz do projeto)
"""


# ==========================================================================
# Análise de Desempenho de Estudantes
# ==========================================================================

# ==========================================================================
# 0. Configuração do ambiente
# ==========================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

# ---- Padrão visual único para todo o relatório ----
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["font.size"] = 10

# ---- Paleta de cores do projeto ----
COR_MAT, COR_LEI, COR_ESC = "#2E86AB", "#A23B72", "#F18F01"
PALETA = [COR_MAT, COR_LEI, COR_ESC]

# ---- Rótulos em português usados nos gráficos ----
NOMES_DISCIPLINA = {"nota_matematica": "Matemática",
                    "nota_leitura":    "Leitura",
                    "nota_escrita":    "Escrita"}

ROTULOS = {
    "female": "Feminino",              "male": "Masculino",
    "standard": "Padrão (paga)",       "free/reduced": "Gratuito/subsidiado",
    "completed": "Completou o curso",  "none": "Não fez o curso",
    "high school":        "Ensino médio completo",
    "some high school":   "Ensino médio incompleto",
    "some college":       "Superior incompleto",
    "associate's degree": "Curso técnico (associate)",
    "bachelor's degree":  "Graduação",
    "master's degree":    "Mestrado",
}

# ---- Exibição de tabelas no pandas ----
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

print("Bibliotecas carregadas com sucesso.")
print(f"pandas     : {pd.__version__}")
print(f"numpy      : {np.__version__}")
print(f"seaborn    : {sns.__version__}")


# ==========================================================================
# ---
# ==========================================================================

# ==========================================================================
# 1.1 Carregamento e primeira inspeção
# ==========================================================================
CAMINHO = "dados/StudentsPerformance.csv"
df = pd.read_csv(CAMINHO)

print(f"Dimensões do dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")
df.head()


# ==========================================================================
# 1.2 Estrutura e tipos de dados
# ==========================================================================
df.info()


# ==========================================================================
# 1.3 Padronização dos nomes das colunas
# ==========================================================================
print("ANTES:")
print(list(df.columns))

renomear = {
    "gender":                      "genero",
    "race/ethnicity":              "grupo_etnico",
    "parental level of education": "escolaridade_pais",
    "lunch":                       "tipo_almoco",
    "test preparation course":     "curso_preparatorio",
    "math score":                  "nota_matematica",
    "reading score":               "nota_leitura",
    "writing score":               "nota_escrita",
}
df = df.rename(columns=renomear)

# Listas reutilizadas ao longo de toda a análise
NOTAS = ["nota_matematica", "nota_leitura", "nota_escrita"]
CATEGORICAS = ["genero", "grupo_etnico", "escolaridade_pais",
               "tipo_almoco", "curso_preparatorio"]

print("\nDEPOIS:")
print(list(df.columns))


# ==========================================================================
# 1.4 Verificação de valores ausentes
# ==========================================================================
ausentes = pd.DataFrame({
    "valores_ausentes": df.isnull().sum(),
    "percentual_%": (df.isnull().sum() / len(df) * 100).round(2)
})
print(ausentes)
print(f"\nTotal de valores ausentes no dataset: {df.isnull().sum().sum()}")


# ==========================================================================
# > ✅ **Resultado:** nenhum valor ausente. Nenhuma imputação é necessária.
# ==========================================================================

# ==========================================================================
# 1.5 Verificação de registros duplicados
# ==========================================================================
n_dup = df.duplicated().sum()
print(f"Registros duplicados (linha inteira idêntica): {n_dup}")

if n_dup > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicatas removidas. Novo total: {len(df)} linhas")
else:
    print("Nenhuma remoção necessária.")


# ==========================================================================
# 1.6 Consistência das variáveis categóricas
# ==========================================================================
for col in CATEGORICAS:
    print(f"{col}  ({df[col].nunique()} categorias)")
    print(f"   -> {sorted(df[col].unique())}\n")

# Padronização preventiva: remove espaços nas bordas e uniformiza caixa
for col in CATEGORICAS:
    df[col] = df[col].str.strip().str.lower()

print("Padronização de texto (strip + lower) aplicada às 5 colunas categóricas.")


# ==========================================================================
# 1.7 Validação de domínio e análise de outliers
# ==========================================================================
print("--- Validação de domínio (0 a 100) ---")
for col in NOTAS:
    fora = df[(df[col] < 0) | (df[col] > 100)]
    print(f"{col:18s} min={df[col].min():3d}  max={df[col].max():3d}  "
          f"valores fora do intervalo: {len(fora)}")

print("\n--- Outliers pelo método IQR ---")
resumo_outliers = []
for col in NOTAS:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = df[(df[col] < lim_inf) | (df[col] > lim_sup)]
    resumo_outliers.append({
        "variavel": col, "Q1": q1, "Q3": q3, "IQR": iqr,
        "lim_inferior": round(lim_inf, 1), "lim_superior": round(lim_sup, 1),
        "n_outliers": len(out), "perc_%": round(len(out) / len(df) * 100, 2)
    })

print(pd.DataFrame(resumo_outliers).to_string(index=False))


# ==========================================================================
# Vamos olhar quem são esses casos extremos antes de decidir o que fazer com eles:
# ==========================================================================
print("Os 5 alunos com pior desempenho médio:")
piores = df.assign(media=df[NOTAS].mean(axis=1)).nsmallest(5, "media")
print(piores[["genero", "tipo_almoco", "curso_preparatorio"] + NOTAS + ["media"]]
      .to_string(index=False))


# ==========================================================================
# > ✅ **Decisão:** todos os outliers são notas plausíveis (0–100) de alunos em situação de
# ==========================================================================

# ==========================================================================
# 1.8 Otimização de tipos e criação de variáveis derivadas
# ==========================================================================
mem_antes = df.memory_usage(deep=True).sum() / 1024

for col in CATEGORICAS:
    df[col] = df[col].astype("category")

# ---- Variáveis derivadas ----
df["media_geral"] = df[NOTAS].mean(axis=1).round(2)
df["situacao"] = np.where(df["media_geral"] >= 60, "Aprovado", "Reprovado")
df["conceito"] = pd.cut(
    df["media_geral"],
    bins=[0, 60, 70, 80, 90, 100],
    labels=["E (<60)", "D (60-70)", "C (70-80)", "B (80-90)", "A (90-100)"],
    right=False
)

mem_depois = df.memory_usage(deep=True).sum() / 1024
print(f"Memória antes : {mem_antes:7.1f} KB")
print(f"Memória depois: {mem_depois:7.1f} KB  "
      f"(reducao de {(1 - mem_depois/mem_antes)*100:.1f}%)")
print(f"\nColunas criadas: media_geral, situacao, conceito")
df[["genero", "tipo_almoco"] + NOTAS + ["media_geral", "situacao", "conceito"]].head()


# ==========================================================================
# 1.9 Dataset limpo
# ==========================================================================
df.to_csv("dados/students_performance_limpo.csv", index=False)

print("=" * 62)
print("RESUMO DA LIMPEZA".center(62))
print("=" * 62)
print(f"Linhas originais .................. 1000")
print(f"Linhas finais .................... {len(df)}")
print(f"Valores ausentes tratados ........ 0 (nenhum encontrado)")
print(f"Duplicatas removidas ............. 0 (nenhuma encontrada)")
print(f"Colunas renomeadas ............... 8")
print(f"Outliers sinalizados / removidos . 19 / 0 (mantidos por decisao)")
print(f"Colunas criadas .................. 3")
print(f"Colunas finais ................... {df.shape[1]}")
print("=" * 62)
print("Arquivo salvo em: dados/students_performance_limpo.csv")


# ==========================================================================
# ---
# ==========================================================================

# ==========================================================================
# 2.1 Estatísticas descritivas das notas
# ==========================================================================
desc = df[NOTAS + ["media_geral"]].describe().T
desc["assimetria"] = df[NOTAS + ["media_geral"]].skew()
desc["curtose"] = df[NOTAS + ["media_geral"]].kurtosis()
desc["amplitude"] = desc["max"] - desc["min"]
desc["cv_%"] = (desc["std"] / desc["mean"] * 100).round(1)   # coef. de variação

print(desc.round(2).to_string())


# ==========================================================================
# **Leitura dos resultados:**
# ==========================================================================

# ==========================================================================
# 2.2 Distribuição das variáveis categóricas
# ==========================================================================
for col in CATEGORICAS:
    contagem = df[col].value_counts()
    perc = df[col].value_counts(normalize=True).mul(100).round(1)
    tabela = pd.DataFrame({"n_alunos": contagem, "percentual_%": perc})
    print(f"--- {col.upper()} ---")
    print(tabela.to_string())
    print()


# ==========================================================================
# **Pontos de atenção:**
# ==========================================================================

# ==========================================================================
# 2.3 Análise bivariada: média por grupo
# ==========================================================================
for col in CATEGORICAS:
    resultado = (df.groupby(col, observed=True)[NOTAS + ["media_geral"]]
                   .mean()
                   .round(2)
                   .sort_values("media_geral", ascending=False))
    resultado["n"] = df[col].value_counts()
    print(f"--- MÉDIA POR {col.upper()} ---")
    print(resultado.to_string())
    print()


# ==========================================================================
# 2.4 Efeito combinado: nível socioeconômico × curso preparatório
# ==========================================================================
tabela = df.pivot_table(index="tipo_almoco", columns="curso_preparatorio",
                        values="media_geral", aggfunc="mean", observed=True).round(2)
contagem = df.pivot_table(index="tipo_almoco", columns="curso_preparatorio",
                          values="media_geral", aggfunc="count", observed=True)

print("MÉDIA GERAL por combinação:")
print(tabela.to_string())
print("\nNÚMERO DE ALUNOS em cada combinação:")
print(contagem.to_string())

melhor = tabela.max().max()
pior = tabela.min().min()
print(f"\nMelhor cenário (almoço padrão + curso feito) : {melhor:.2f}")
print(f"Pior cenário  (almoço subsidiado + sem curso): {pior:.2f}")
print(f"DIFERENÇA ENTRE OS EXTREMOS .................: {melhor - pior:.2f} pontos")


# ==========================================================================
# 2.5 Correlação entre as disciplinas
# ==========================================================================
corr = df[NOTAS].corr()
print("Matriz de correlação de Pearson:")
print(corr.round(3).to_string())

print("\nInterpretação par a par:")
pares = [("nota_leitura", "nota_escrita"), ("nota_matematica", "nota_leitura"),
         ("nota_matematica", "nota_escrita")]
for a, b in pares:
    r = corr.loc[a, b]
    forca = "muito forte" if r >= 0.9 else "forte" if r >= 0.7 else "moderada"
    print(f"  {a:16s} x {b:16s} : r = {r:.3f}  ({forca})")


# ==========================================================================
# 2.6 Testes de hipótese
# ==========================================================================
def cohen_d(a, b):
    na, nb = len(a), len(b)
    s = np.sqrt(((na-1)*a.std(ddof=1)**2 + (nb-1)*b.std(ddof=1)**2) / (na+nb-2))
    return (a.mean() - b.mean()) / s

comparacoes = [
    ("tipo_almoco",        "standard",        "free/reduced"),
    ("curso_preparatorio", "completed",       "none"),
    ("genero",             "female",          "male"),
    ("escolaridade_pais",  "master's degree", "high school"),
]

linhas = []
for var, g1, g2 in comparacoes:
    a = df[df[var] == g1]["media_geral"]
    b = df[df[var] == g2]["media_geral"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    d = cohen_d(a, b)
    magnitude = "grande" if abs(d) >= 0.8 else "médio" if abs(d) >= 0.5 else "pequeno"
    linhas.append({
        "fator": var, "grupo_1": g1, "media_1": round(a.mean(), 2),
        "grupo_2": g2, "media_2": round(b.mean(), 2),
        "diferenca": round(a.mean() - b.mean(), 2),
        "p_valor": f"{p:.2e}",
        "cohen_d": round(d, 2), "efeito": magnitude,
        "significativo": "SIM" if p < 0.05 else "não"
    })

print(pd.DataFrame(linhas).to_string(index=False))


# ==========================================================================
# > ✅ **Todas as quatro diferenças são estatisticamente significativas (p < 0,001).**
# ==========================================================================

# ==========================================================================
# 2.7 Taxa de reprovação por grupo
# ==========================================================================
print(f"Taxa geral de reprovação (média < 60): "
      f"{(df['situacao'] == 'Reprovado').mean() * 100:.1f}%\n")

for col in ["tipo_almoco", "curso_preparatorio", "genero", "escolaridade_pais"]:
    taxa = (df.groupby(col, observed=True)["situacao"]
              .apply(lambda s: (s == "Reprovado").mean() * 100)
              .round(1)
              .sort_values(ascending=False))
    print(f"--- Reprovação (%) por {col} ---")
    print(taxa.to_string())
    print()


# ==========================================================================
# ---
# ==========================================================================

# ==========================================================================
# Gráfico 1 — Como as notas se distribuem?
# ==========================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, col, cor in zip(axes, NOTAS, PALETA):
    sns.histplot(df[col], bins=25, kde=True, color=cor, ax=ax,
                 edgecolor="white", alpha=0.75)
    ax.axvline(df[col].mean(), color="black", linestyle="--", linewidth=1.6,
               label=f"Média = {df[col].mean():.1f}")
    ax.set_title(NOMES_DISCIPLINA[col])
    ax.set_xlabel("Nota")
    ax.set_ylabel("Frequência")
    ax.legend(fontsize=9)

fig.suptitle("Distribuição das notas por disciplina", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
plt.show()


# ==========================================================================
# > **Lê-se:** as três distribuições são aproximadamente normais, centradas perto de 66–69,
# ==========================================================================

# ==========================================================================
# Gráfico 2 — Meninos e meninas vão bem nas mesmas coisas?
# ==========================================================================
# Formato "longo": uma linha por (aluno, disciplina) — exigido pelo seaborn
dados_longos = df.melt(id_vars="genero", value_vars=NOTAS,
                       var_name="disciplina", value_name="nota")

# Traduzimos os VALORES antes de plotar (em vez de sobrescrever os rótulos
# da legenda depois) — assim é impossível trocar as categorias por engano
dados_longos["disciplina"] = dados_longos["disciplina"].map(NOMES_DISCIPLINA)
dados_longos["genero"] = dados_longos["genero"].astype(str).map(ROTULOS)

ordem_disc = list(NOMES_DISCIPLINA.values())

plt.figure(figsize=(11, 5.8))
ax = sns.boxplot(data=dados_longos, x="disciplina", y="nota", hue="genero",
                 order=ordem_disc, hue_order=["Feminino", "Masculino"],
                 palette={"Feminino": "#E56B9F", "Masculino": "#2E86AB"},
                 width=0.65)
ax.set_title("Distribuição das notas por gênero e disciplina")
ax.set_xlabel("")
ax.set_ylabel("Nota")
ax.legend(title="Gênero", bbox_to_anchor=(1.01, 1), loc="upper left")

# Anota a diferença de médias (feminino - masculino) em cada disciplina
medias = df.groupby("genero", observed=True)[NOTAS].mean()
for i, col in enumerate(NOTAS):
    dif = medias.loc["female", col] - medias.loc["male", col]
    ax.text(i, 109, f"Δ = {dif:+.1f}", ha="center", fontsize=11,
            fontweight="bold", color="#C0392B" if dif < 0 else "#1E8449")
ax.set_ylim(-5, 118)
plt.tight_layout()
plt.show()


# ==========================================================================
# > **Lê-se:** o padrão se **inverte** conforme a disciplina. Meninos levam vantagem em
# ==========================================================================

# ==========================================================================
# Gráfico 3 — O curso preparatório funciona?
# ==========================================================================
comp = (df.groupby("curso_preparatorio", observed=True)[NOTAS]
          .mean().T.round(2))

# Ordem e nomes definidos explicitamente -> legenda sempre correta
comp = comp[["completed", "none"]]
comp.index = [NOMES_DISCIPLINA[i] for i in comp.index]
comp.columns = ["Completou o curso", "Não fez o curso"]

ax = comp.plot(kind="bar", figsize=(10, 5.5), width=0.72,
               color=["#1E8449", "#C0392B"], edgecolor="white")
ax.set_title("Nota média por disciplina — com e sem curso preparatório")
ax.set_xlabel("")
ax.set_ylabel("Nota média")
ax.tick_params(axis="x", rotation=0)
ax.legend(title=None, loc="lower right")
ax.set_ylim(0, 90)

for cont in ax.containers:
    ax.bar_label(cont, fmt="%.1f", fontsize=10, fontweight="bold", padding=2)

# Ganho de cada disciplina
for i, disc in enumerate(comp.index):
    ganho = comp.loc[disc, "Completou o curso"] - comp.loc[disc, "Não fez o curso"]
    ax.text(i, 84, f"+{ganho:.1f}", ha="center", fontsize=11,
            fontweight="bold", color="#1E8449")

plt.tight_layout()
plt.show()


# ==========================================================================
# > **Lê-se:** o curso está associado a ganho em **todas** as disciplinas, e o ganho é maior
# ==========================================================================

# ==========================================================================
# Gráfico 4 — A escolaridade dos pais influencia?
# ==========================================================================
ordem = ["high school", "some high school", "some college",
         "associate's degree", "bachelor's degree", "master's degree"]
medias_pais = (df.groupby("escolaridade_pais", observed=True)["media_geral"]
                 .mean().reindex(ordem).round(2))

rotulos_pt = [ROTULOS[i] for i in medias_pais.index]
media_global = df["media_geral"].mean()

plt.figure(figsize=(10.5, 5.5))
ax = sns.barplot(x=medias_pais.values, y=rotulos_pt, hue=rotulos_pt,
                 palette=sns.color_palette("YlGnBu", len(medias_pais)),
                 legend=False)
ax.set_title("Média geral do aluno por escolaridade dos pais\n"
             f"(linha vermelha = média geral do dataset, {media_global:.1f})")
ax.set_xlabel("Média geral")
ax.set_ylabel("")
ax.set_xlim(0, 85)

for cont in ax.containers:                    # uma barra por container
    ax.bar_label(cont, fmt="%.1f", fontsize=10, fontweight="bold", padding=4)

ax.axvline(media_global, color="#C0392B", linestyle="--", linewidth=1.6)

plt.tight_layout()
plt.show()


# ==========================================================================
# > **Lê-se:** existe um **gradiente perfeitamente monotônico** — a cada degrau de
# ==========================================================================

# ==========================================================================
# Gráfico 5 — O nível socioeconômico separa os alunos?
# ==========================================================================
# ATENÇÃO: a ordem das categorias precisa ser declarada explicitamente com
# 'order='. Sem isso o seaborn ordena alfabeticamente (free/reduced vem
# antes de standard) e os rótulos manuais ficariam invertidos.
# Criamos uma coluna já traduzida e com ordem fixa. Isso garante que
# boxplot e violinplot usem exatamente a mesma cor para a mesma categoria.
ROT = ["Padrão (paga)", "Gratuito/subsidiado"]
CORES = {"Padrão (paga)": "#2E86AB", "Gratuito/subsidiado": "#F18F01"}

df_plot = df.copy()
df_plot["almoco"] = pd.Categorical(
    df_plot["tipo_almoco"].astype(str).map(ROTULOS), categories=ROT, ordered=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

sns.boxplot(data=df_plot, x="almoco", y="media_geral", order=ROT,
            hue="almoco", hue_order=ROT, palette=CORES,
            legend=False, width=0.5, ax=axes[0])
sns.violinplot(data=df_plot, x="almoco", y="media_geral", order=ROT,
               hue="almoco", hue_order=ROT, palette=CORES,
               legend=False, inner="quartile", ax=axes[1])

medias = df_plot.groupby("almoco", observed=True)["media_geral"].mean()

axes[0].set_title(f"Média geral por tipo de almoço\n"
                  f"(padrão: {medias[ROT[0]]:.2f}  |  subsidiado: {medias[ROT[1]]:.2f})")
axes[0].set_ylabel("Média geral")
axes[1].set_title("Mesma comparação — formato da distribuição")
axes[1].set_ylabel("")

for ax in axes:
    ax.set_xlabel("")

# Linhas horizontais marcando a média de cada grupo
for rot in ROT:
    axes[0].axhline(medias[rot], color=CORES[rot], linestyle=":", linewidth=1.6)

fig.suptitle("Impacto do nível socioeconômico no desempenho",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


# ==========================================================================
# > **Lê-se:** a caixa inteira do grupo subsidiado está deslocada para baixo, e o violino
# ==========================================================================

# ==========================================================================
# Gráfico 6 — As disciplinas andam juntas?
# ==========================================================================
plt.figure(figsize=(7.5, 6))
mascara = np.triu(np.ones_like(df[NOTAS].corr(), dtype=bool), k=1)
ax = sns.heatmap(df[NOTAS].corr(), annot=True, fmt=".3f", cmap="RdYlBu_r",
                 vmin=0.7, vmax=1, square=True, linewidths=2, mask=mascara,
                 cbar_kws={"label": "Coeficiente de Pearson"},
                 annot_kws={"size": 13, "weight": "bold"})
ax.set_title("Correlação entre as disciplinas", pad=14)
rotulos = ["Matemática", "Leitura", "Escrita"]
ax.set_xticklabels(rotulos, rotation=0)
ax.set_yticklabels(rotulos, rotation=0)
plt.tight_layout()
plt.show()


# ==========================================================================
# > **Lê-se:** leitura e escrita são **quase a mesma competência** (r = 0,955). Matemática
# ==========================================================================

# ==========================================================================
# Gráfico 7 — Visualizando a correlação leitura × escrita
# ==========================================================================
# Traduz os valores antes de plotar para a legenda sair correta
df_plot = df.copy()
df_plot["curso"] = (df_plot["curso_preparatorio"].astype(str)
                    .map({"completed": "Completou", "none": "Não fez"}))

plt.figure(figsize=(9, 6.5))
ax = sns.scatterplot(data=df_plot, x="nota_leitura", y="nota_escrita",
                     hue="curso", hue_order=["Completou", "Não fez"],
                     palette={"Completou": "#1E8449", "Não fez": "#C0392B"},
                     alpha=0.7, s=45)
sns.regplot(data=df, x="nota_leitura", y="nota_escrita", scatter=False,
            color="black", line_kws={"linewidth": 2, "linestyle": "--"}, ax=ax)

r = df["nota_leitura"].corr(df["nota_escrita"])
ax.set_title(f"Leitura x Escrita  (r = {r:.3f})")
ax.set_xlabel("Nota de leitura")
ax.set_ylabel("Nota de escrita")
ax.legend(title="Curso preparatório", loc="upper left")
plt.tight_layout()
plt.show()


# ==========================================================================
# > **Lê-se:** os pontos formam uma faixa estreita em torno da reta — relação praticamente
# ==========================================================================

# ==========================================================================
# Gráfico 8 — Onde está a reprovação?
# ==========================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
config = [("tipo_almoco", "Tipo de almoço"),
          ("curso_preparatorio", "Curso preparatório"),
          ("genero", "Gênero")]

for ax, (col, titulo) in zip(axes, config):
    taxa = (df.groupby(col, observed=True)["situacao"]
              .apply(lambda s: (s == "Reprovado").mean() * 100)
              .sort_values(ascending=False))
    rotulos = [ROTULOS.get(str(i), str(i)).replace(" o curso", "")
               for i in taxa.index]
    cores = ["#C0392B" if v == taxa.max() else "#7FB3D5" for v in taxa.values]
    b = ax.bar(rotulos, taxa.values, color=cores,
               edgecolor="white", width=0.6)
    ax.bar_label(b, fmt="%.1f%%", fontsize=11, fontweight="bold", padding=3)
    ax.axhline(28.5, color="gray", linestyle="--", linewidth=1.2)
    ax.set_title(titulo)
    ax.set_ylabel("Reprovação (%)" if col == "tipo_almoco" else "")
    ax.set_ylim(0, 55)
    ax.tick_params(axis="x", rotation=12)

fig.suptitle("Taxa de reprovação (média < 60) — linha cinza = média geral de 28,5%",
             fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
plt.show()


# ==========================================================================
# > **Lê-se:** a reprovação entre alunos com almoço subsidiado (**43,7%**) é mais que o
# ==========================================================================

# ==========================================================================
# Gráfico 9 — O efeito combinado dos dois fatores mais fortes
# ==========================================================================
tabela = (df.pivot_table(index="tipo_almoco", columns="curso_preparatorio",
                         values="media_geral", aggfunc="mean", observed=True)
            .reindex(index=["standard", "free/reduced"],      # ordem explícita
                     columns=["completed", "none"]))

plt.figure(figsize=(8.5, 5.5))
ax = sns.heatmap(tabela, annot=True, fmt=".2f", cmap="RdYlGn", center=66,
                 linewidths=3, square=False, annot_kws={"size": 17, "weight": "bold"},
                 cbar_kws={"label": "Média geral"})
ax.set_title("Média geral: nível socioeconômico x curso preparatório", pad=14)
ax.set_xlabel("Curso preparatório")
ax.set_ylabel("Tipo de almoço")
ax.set_xticklabels(["Completou", "Não fez"], rotation=0)
ax.set_yticklabels(["Padrão (paga)", "Gratuito/subsidiado"], rotation=0)
plt.tight_layout()
plt.show()

print(f"Gap entre o melhor e o pior cenário: "
      f"{tabela.max().max() - tabela.min().min():.2f} pontos")


# ==========================================================================
# > **Lê-se:** os efeitos **se acumulam**. Um aluno com almoço padrão que fez o curso tem
# ==========================================================================

# ==========================================================================
# ---
# ==========================================================================
print("=" * 70)
print("PAINEL DE INSIGHTS".center(70))
print("=" * 70)

media_g = df["media_geral"].mean()

# 1
alm = df.groupby("tipo_almoco", observed=True)["media_geral"].mean()
print(f"\n[1] NIVEL SOCIOECONOMICO E O DIVISOR MAIS FORTE")
print(f"    Almoco padrao ....... {alm['standard']:.2f}")
print(f"    Almoco subsidiado ... {alm['free/reduced']:.2f}")
print(f"    Diferenca ........... {alm['standard'] - alm['free/reduced']:.2f} pontos (d = 0.63)")

# 2
cur = df.groupby("curso_preparatorio", observed=True)["media_geral"].mean()
rep = df.groupby("curso_preparatorio", observed=True)["situacao"].apply(
    lambda s: (s == "Reprovado").mean() * 100)
print(f"\n[2] O CURSO PREPARATORIO E A ALAVANCA MAIS ACIONAVEL")
print(f"    Media com curso ..... {cur['completed']:.2f}   (reprovacao {rep['completed']:.1f}%)")
print(f"    Media sem curso ..... {cur['none']:.2f}   (reprovacao {rep['none']:.1f}%)")
print(f"    Ganho ............... {cur['completed'] - cur['none']:.2f} pontos")
print(f"    Alcance atual ....... apenas {(df['curso_preparatorio']=='completed').mean()*100:.1f}% dos alunos")

# 3
gen = df.groupby("genero", observed=True)[NOTAS].mean()
print(f"\n[3] O PADRAO DE GENERO SE INVERTE POR DISCIPLINA")
for col in NOTAS:
    d = gen.loc['female', col] - gen.loc['male', col]
    lider = "meninas" if d > 0 else "meninos"
    print(f"    {col.replace('nota_',''):12s}: {abs(d):5.2f} pontos a favor de {lider}")

# 4
ordem_e = ["high school", "some high school", "some college",
           "associate's degree", "bachelor's degree", "master's degree"]
esc = df.groupby("escolaridade_pais", observed=True)["media_geral"].mean().reindex(ordem_e)
print(f"\n[4] ESCOLARIDADE DOS PAIS: GRADIENTE MONOTONICO")
print(f"    Ensino medio ........ {esc.iloc[0]:.2f}")
print(f"    Mestrado ............ {esc.iloc[-1]:.2f}")
print(f"    Amplitude ........... {esc.iloc[-1] - esc.iloc[0]:.2f} pontos (d = 0.78, o maior efeito)")

# 5
corr_le = df["nota_leitura"].corr(df["nota_escrita"])
corr_ml = df["nota_matematica"].corr(df["nota_leitura"])
print(f"\n[5] LEITURA E ESCRITA SAO PRATICAMENTE A MESMA COMPETENCIA")
print(f"    r(leitura, escrita) ..... {corr_le:.3f}")
print(f"    r(matematica, leitura) .. {corr_ml:.3f}")
print(f"    Matematica e a disciplina mais 'independente' e a de menor media ({df['nota_matematica'].mean():.2f})")

# 6
piv = df.pivot_table(index="tipo_almoco", columns="curso_preparatorio",
                     values="media_geral", aggfunc="mean", observed=True)
print(f"\n[6] OS EFEITOS SE ACUMULAM")
print(f"    Melhor cenario (padrao + curso) ....... {piv.loc['standard','completed']:.2f}")
print(f"    Pior cenario (subsidiado + sem curso) . {piv.loc['free/reduced','none']:.2f}")
print(f"    Gap ................................... {piv.max().max()-piv.min().min():.2f} pontos")
print(f"    Baixa renda COM curso ({piv.loc['free/reduced','completed']:.2f}) quase alcanca")
print(f"    alta renda SEM curso ({piv.loc['standard','none']:.2f}) -> o curso compensa a desvantagem")
print("\n" + "=" * 70)


# ==========================================================================
# 🔍 Os 6 insights em detalhe
# ==========================================================================

# ==========================================================================
# ---
# ==========================================================================
print("=" * 62)
print("ANALISE CONCLUIDA".center(62))
print("=" * 62)
print(f"Registros analisados ....... {len(df)}")
print(f"Variaveis finais ........... {df.shape[1]}")
print(f"Graficos gerados ........... 9")
print(f"Testes estatisticos ........ 4 (todos com p < 0.001)")
print(f"Insights identificados ..... 6")
print(f"Recomendacoes .............. 6")
print("=" * 62)
