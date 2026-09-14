# 🦏 Detecção de animais selvagens com YOLO26

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![YOLO26](https://img.shields.io/badge/Ultralytics-YOLO26-0b23a0)
![Colab](https://img.shields.io/badge/Google%20Colab-GPU-F9AB00?logo=googlecolab&logoColor=white)

Detector de objetos treinado para identificar **quatro animais africanos** em
imagens e vídeo: **búfalo, elefante, rinoceronte e zebra**.

> **Disciplina:** Laboratório de Inovação

---

## Como executar

1. Abra `deteccao_animais_yolo26.ipynb` no [Google Colab](https://colab.research.google.com/drive/18pS5Kwhk1ibr5BFGYlqhFdszYYZpYoD_?usp=sharing)
2. Ative a GPU: `Ambiente de execução → Alterar o tipo de ambiente de execução → GPU (T4)`
3. Rode as células em ordem, de cima para baixo

Não precisa configurar nada nem baixar dataset. O African Wildlife é um dos
datasets oficiais da Ultralytics e vem automaticamente.

**Tempo estimado:** 25 a 35 minutos na GPU gratuita.

---

## O que o notebook faz

| Etapa | Descrição |
|---|---|
| 1. Instalação | Instala o `ultralytics` e confirma a GPU |
| 2. Dataset | Mostra a estrutura do African Wildlife (1.504 imagens, 4 classes) |
| 3. Treino | Fine-tuning do `yolo26s.pt` por 50 épocas |
| 4. Avaliação | Precisão, recall, mAP por classe e matriz de confusão |
| 5. Imagens | Detecção em imagens do conjunto de teste |
| 6. Vídeo | Detecção com rastreamento e contagem de animais únicos |
| 7. Download | Baixa os pesos treinados e o vídeo anotado |

---

## Sobre o dataset

O **African Wildlife** é mantido pela Ultralytics e baixa sozinho (~100 MB) na
primeira vez que o treino roda.

| | |
|---|---|
| Imagens | 1.504 |
| Classes | buffalo, elephant, rhino, zebra |
| Treino | 1.052 imagens |
| Validação | 225 imagens |
| Teste | 227 imagens |

As imagens vêm de reservas naturais da África do Sul. Publicado originalmente no
Kaggle por Bianca Ferreira e adaptado pela Ultralytics.

---

## Por que fine-tuning

Os checkpoints do YOLO26 já passaram por 150 épocas de pré-treino no Objects365
e depois foram ajustados no COCO — que contém `elephant` e `zebra` entre suas
classes. O modelo já sabe reconhecer animais grandes e delimitá-los numa imagem.

O treino aqui só especializa esse conhecimento nas quatro espécies. É por isso
que 1.052 imagens bastam, enquanto treinar do zero exigiria dezenas de milhares.

---

## Nota de implementação: caminhos descobertos, não fixados

Todos os caminhos de arquivo do notebook são **resolvidos em tempo de
execução**, nunca escritos à mão:

| Em vez de | O notebook usa |
|---|---|
| `"runs/detect/wildlife/"` | `modelo.trainer.save_dir` |
| `"runs/detect/wildlife/weights/best.pt"` | `modelo.trainer.best` |
| `"runs/detect/wildlife_val/"` | `metricas.save_dir` |
| `"runs/video/teste"` | `modelo.predictor.save_dir` |
| `"datasets/african-wildlife/test/images"` | `check_det_dataset()` |

Há dois motivos concretos:

1. **O Ultralytics renomeia pastas.** Se `runs/detect/wildlife` já existir de
   uma execução anterior, a próxima vai para `wildlife2`, depois `wildlife3`.
2. **A estrutura do dataset mudou entre versões.** Em releases antigas era
   `test/images`; na atual é `images/test`. Código com caminho fixo funciona
   hoje e quebra na próxima atualização do pacote.

O notebook também traz uma função `mostrar()` que, se um gráfico não existir,
avisa e lista o que há na pasta — em vez de interromper tudo com
`FileNotFoundError`.

---

## Resultados

> Preencha após rodar o notebook.

| Classe | Precisão | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| buffalo | — | — | — | — |
| elephant | — | — | — | — |
| rhino | — | — | — | — |
| zebra | — | — | — | — |
| **Geral** | — | — | — | — |

---

## Solução de problemas

**`FileNotFoundError` num gráfico (`PR_curve.png`, `results.png`)**
O Ultralytics salvou numa pasta com sufixo (`wildlife2`, `wildlife_val2`). As
células que imprimem `PASTA_TREINO` e `PASTA_VAL` mostram o caminho real.

**`FileNotFoundError` na pasta de imagens do dataset**
A estrutura mudou entre versões. A função `imagens_do_dataset()` resolve isso
via `check_det_dataset()` e, se falhar, varre a raiz do dataset atrás de imagens.

**`NameError: name 'PESOS' is not defined`**
Alguma célula anterior falhou ou foi pulada. As variáveis vivem na memória do
kernel, então o notebook precisa ser rodado em ordem.

**`CUDA out of memory` durante o treino**
Reduza `batch=16` para `batch=8` na célula de treino.

**O vídeo não aparece no navegador**
A célula recodifica para H.264 com o ffmpeg. Se ainda assim não tocar, o arquivo
está em `PASTA_VIDEO` e pode ser baixado pelo painel de arquivos do Colab.

---

## Limitações

- **Só quatro espécies.** Qualquer outro animal ou é classificado como uma das
  quatro, ou não é detectado. O modelo não tem como dizer "não sei".
- **Domínio restrito.** As imagens são de reservas sul-africanas, em boa parte
  com o animal bem enquadrado. Armadilha fotográfica noturna ou animais
  distantes teriam desempenho bem pior.
- **Rastreamento por posição.** O ByteTrack usa só a posição, não a aparência.
  Quando dois animais se cruzam, os IDs podem trocar.
- **Licença AGPL-3.0.** O Ultralytics é AGPL: uso comercial exige licença
  Enterprise. Para trabalho acadêmico não há problema.

---

## Referências

- Jocher, G. et al. **Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models.** arXiv:2606.03748, 2026.
- [Documentação do YOLO26](https://docs.ultralytics.com/models/yolo26)
- [Dataset African Wildlife](https://docs.ultralytics.com/datasets/detect/african-wildlife)
