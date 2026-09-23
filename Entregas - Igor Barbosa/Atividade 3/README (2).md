# ♻️ Classificador de resíduos recicláveis com exportação ONNX

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-navegador-005CED?logo=onnx&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-GPU-F9AB00?logo=googlecolab&logoColor=white)

Rede neural que identifica o **tipo de material** de um resíduo a partir de uma
foto — papelão, vidro, metal, papel, plástico — treinada com transfer learning
sobre MobileNetV2 e exportada em **ONNX** para rodar direto no navegador.

> **Disciplina:** Laboratório de Inovação

---

## O que este projeto entrega

| Arquivo | Para quê |
|---|---|
| `modelo.onnx` | O modelo (~9 MB), executável no navegador via `onnxruntime-web` |
| `metadados.json` | O contrato do modelo: nomes dos tensores, tamanho da imagem, classes |
| `modelo.keras` | O modelo em formato Keras, caso queira retomar o treino |

O `metadados.json` é o que permite ao site funcionar **sem nenhum valor escrito
à mão** no JavaScript — e é também o que torna possível aceitar o modelo ONNX de
outra pessoa.

---

## Como executar

1. Abra `classificador_residuos_onnx.ipynb` no [Google Colab](https://colab.research.google.com/drive/1p2zFnWn16GEov4vZEu2mYpE4l7zi_vq-?usp=sharing)
2. Ative a GPU: `Ambiente de execução → Alterar o tipo de ambiente de execução → GPU (T4)`
3. Na etapa 2, cole **apenas a sua chave da API do Roboflow** — o dataset já vem preenchido
4. Rode as células em ordem

**Tempo estimado:** 10 a 15 minutos na GPU gratuita.

> ⚠️ **A primeira célula vai imprimir um bloco vermelho de conflito do pip.
> Isso é esperado e não é falha.** O `onnx` exige protobuf mais novo que o de
> três pacotes pré-instalados no Colab (`google-ai-generativelanguage`, `ydf` e
> `grpcio-status`), nenhum deles usado aqui. O TensorFlow **não** aparece na
> lista de conflitos. A célula seguinte importa tudo de fato e confirma.

---

## O dataset

O notebook já vem apontando para um dataset de classificação verificado, então
não é preciso caçar nada no Roboflow. A chave da API sai de
**app.roboflow.com → Settings → API Keys**.

Três projetos de **classificação** conferidos no Universe:

| Workspace / Projeto | Imagens | Classes |
|---|---|---|
| `garbage-vhmrm` / `garbage-hforb` | ~2.500 | cardboard, glass, metal, paper, plastic, trash |
| `teamproject-3f4si` / `garbage-pqluq` | ~2.400 | cardboard, glass, metal, paper, plastic |
| `trashclassification-tayqe` / `trash-classification-vky3a` | ~1.600 | 8 classes, inclui compost e recycle |

O primeiro vem preenchido no notebook: seis classes claras e tamanho bom para
treinar em poucos minutos.

> Se quiser trocar, o filtro que mais importa na busca é o **tipo do projeto**:
> precisa ser **Classification**. Boa parte dos resultados de "waste" no
> Universe é detecção, e detecção não serve aqui. Exporte sempre como
> **Folder Structure**.

### Por que "Folder Structure"

Esse formato entrega as imagens organizadas com **o nome da pasta como rótulo**:

```
dados/
├── train/
│   ├── cardboard/   ├── glass/   ├── metal/   ├── paper/   └── plastic/
├── valid/
└── test/
```

É exatamente o que o `image_dataset_from_directory` do Keras espera — nenhum
código de leitura de rótulo é necessário.

---

## O que o notebook faz

| Etapa | Descrição |
|---|---|
| 1. Preparação | Instala dependências, **verifica os imports**, confirma a GPU |
| 2. Dataset | Baixa do Roboflow e descobre os caminhos reais |
| 3. Exploração | Classes, balanceamento e amostras visuais |
| 4. Pipelines | `tf.data` com cache, shuffle, prefetch e augmentation |
| 5. Modelo | MobileNetV2 + cabeça, com normalização embutida |
| 6. Treino | Duas fases: base congelada, depois ajuste fino |
| 7. Avaliação | Matriz de confusão, métricas por classe, erros mais confiantes |
| 8. **ONNX** | Exporta e **verifica numericamente** contra o Keras |
| 9. Artefatos | Gera `metadados.json` e baixa tudo |

---

## As três decisões que definiram o projeto

### 1. Transfer learning em duas fases

Congelar a base primeiro não é formalidade. A cabeça de classificação começa
com **pesos aleatórios**, então seus primeiros gradientes são enormes e
caóticos. Com a base destravada, esses gradientes destruiriam as features do
ImageNet logo nas primeiras iterações — exatamente o que viemos aproveitar.

Só depois que a cabeça está calibrada é que destravamos as camadas finais da
base, com taxa de aprendizado **dez vezes menor**.

> ⚠️ **Pegadinha:** as camadas `BatchNormalization` continuam congeladas no
> ajuste fino. Com lote pequeno, atualizar suas estatísticas desestabiliza o
> treino. E é **obrigatório recompilar** o modelo depois de mudar `trainable`,
> senão o Keras mantém o grafo antigo e o destravamento não tem efeito nenhum.

### 2. Normalização dentro do modelo

A camada `Rescaling(1/127.5, offset=-1)` fica **dentro** do grafo exportado.

O caminho mais comum em tutoriais é normalizar no pipeline de dados e deixar o
modelo esperando valores já em `[-1, 1]`. Mas aí o JavaScript do site teria que
reproduzir exatamente essa conta. Qualquer divergência — dividir por 255 em vez
de 127.5, esquecer o deslocamento — produz predições **silenciosamente
erradas**. Nada quebra; o modelo apenas passa a errar.

Com a conta dentro do grafo, o contrato vira: **mande pixels de 0 a 255, receba
probabilidades**. Não há como errar.

### 3. Augmentation fora do modelo

No projeto anterior (CIFAR-100) as camadas de augmentation ficavam dentro do
modelo. Aqui elas ficam no pipeline de dados, pela razão oposta: camadas
aleatórias no grafo exportado geram operações estranhas no ONNX, ou fazem a
conversão falhar. Mantendo o modelo "puro", o ONNX sai limpo.

---

## Por que classificação, e não detecção

Boa parte dos datasets de resíduo no Roboflow é de **detecção com caixas
delimitadoras**. A escolha por classificação foi deliberada, por duas razões.

**Prática:** detecção não se treina com Keras/TensorFlow de forma razoável —
exigiria outro framework, abandonando o pipeline que a disciplina pede.

**Decisiva para o site:** modelos de detecção têm formatos de saída
incompatíveis entre si.

| Modelo | Formato da saída |
|---|---|
| YOLOv5 | `[1, 25200, 85]` |
| YOLOv8 | `[1, 84, 8400]` |
| YOLO26 (end-to-end) | `[1, 300, 6]` |

Um site que aceite "qualquer ONNX" não teria como adivinhar a decodificação
correta. Em classificação a saída é sempre `[1, n_classes]` com softmax —
**universal**. O recurso "traga seu próprio modelo" só é viável assim.

---

## Exportação para ONNX

O `tf2onnx` **não aceita um modelo Keras 3 diretamente** — a API mudou e o
conversor não acompanhou. O caminho que funciona tem duas etapas:

```
Modelo Keras 3  →  SavedModel (model.export)  →  ONNX (tf2onnx --opset 13)
```

O `model.export()` é diferente do `model.save()`: grava no formato SavedModel do
TensorFlow, já em modo de inferência.

### A verificação que ninguém faz

Converter é fácil; converter **certo** é outra coisa. O notebook roda as mesmas
imagens nos dois modelos e compara as saídas número a número:

```
KERAS  x  ONNX
  Diferença máxima nas probabilidades . 0.00e+00
  Mesma classe prevista ............... 100.0%
```

Cinco linhas de código que separam "sei que está correto" de "espero que
esteja".

---

## O contrato do modelo

Este é o conteúdo que o site consome. Os nomes dos tensores foram verificados na
prática, não presumidos:

```json
{
  "classes": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
  "entrada": {
    "nome": "imagem",
    "formato": "NHWC",
    "altura": 224, "largura": 224, "canais": 3,
    "tipo": "float32",
    "faixa_pixels": "0-255"
  },
  "saida": {
    "nome": "output_0",
    "formato": "[lote, n_classes]",
    "ativacao": "softmax"
  }
}
```

> 💡 **Para o site:** leia os nomes de entrada e saída da própria sessão
> (`session.inputNames[0]`, `session.outputNames[0]`) em vez de fixá-los no
> código. É isso que permite carregar o ONNX de qualquer outra pessoa sem
> alterar o JavaScript.

---

## Resultados

> Preencha após rodar o notebook.

| Métrica | Valor |
|---|---|
| Classes | — |
| Acurácia no teste | — |
| Ganho do ajuste fino | — |
| Tamanho do ONNX | ~9 MB |

---

## Solução de problemas

**Bloco vermelho de conflito do pip na primeira célula**
Esperado. Veja a nota na seção "Como executar". A célula seguinte confirma que
tudo importa.

**`Roboflow: 401 Unauthorized`**
Chave da API incorreta. Pegue em app.roboflow.com → Settings → API Keys.

**As pastas de classe estão erradas ou só aparece uma classe**
O dataset provavelmente é de detecção, não de classificação. Confirme que
exportou como **Folder Structure**.

**`ResourceExhaustedError` na GPU**
Reduza `LOTE = 32` para `16` na etapa 4.

**A fase 2 não melhora nada**
Confirme que o `modelo.compile()` foi chamado **depois** de mudar
`base_mobilenet.trainable`. Sem recompilar, o destravamento não surte efeito.

**A conversão para ONNX falhou**
A célula imprime o erro do `tf2onnx`. A causa mais comum é ter camadas de
augmentation dentro do modelo — elas devem ficar no pipeline de dados.

**`NameError` em alguma variável**
As variáveis vivem na memória do kernel. Rode em ordem, de cima para baixo.

---

## Limitações

- **Fotos de dataset não são fotos de celular.** As imagens do Roboflow tendem a
  ter o objeto centralizado, bem iluminado e em fundo limpo. Com foto de cozinha
  mal iluminada e três objetos na cena, o desempenho cai.
- **O modelo não sabe dizer "não sei".** Uma foto de um gato recebe alguma
  classe de resíduo com alta confiança. Resolver exigiria uma classe "nenhum" no
  treino ou um limiar mínimo de confiança no site.
- **Um objeto por imagem.** É classificação, não detecção: garrafa e lata juntas
  recebem um rótulo só.
- **Vidro e plástico transparente se confundem.** Ambos são translúcidos e
  brilhantes. Papel e papelão também, pela cor. Essas confusões são esperadas e
  aparecem na matriz da etapa 7.

---

## Próximo passo

Com `modelo.onnx` e `metadados.json` em mãos, o próximo passo é o site no
Lovable: carregar o modelo no navegador com `onnxruntime-web`, receber a foto do
usuário pela câmera ou por upload, e exibir as probabilidades — tudo local, sem
servidor e sem enviar a imagem para lugar nenhum.

---

## Referências

- [MobileNetV2 — Sandler et al., 2018](https://arxiv.org/abs/1801.04381)
- [Transfer learning e fine-tuning no Keras](https://keras.io/guides/transfer_learning/)
- [tf2onnx](https://github.com/onnx/tensorflow-onnx)
- [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/)
- [Roboflow Universe](https://universe.roboflow.com)
