# 🖼️ Classificação de imagens com CIFAR-100

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-GPU-F9AB00?logo=googlecolab&logoColor=white)

Rede neural convolucional que classifica imagens em **100 categorias** — de maçã
a zebra, de foguete a girassol — treinada do zero com Keras/TensorFlow.

> **Disciplina:** Laboratório de Inovação

---

## O diferencial deste projeto

Quase todo tutorial de CIFAR-100 termina em "acurácia de X%". Este vai além,
aproveitando uma característica pouco explorada do dataset: além das 100
classes, ele traz **20 superclasses** que agrupam as categorias por afinidade.

> `dolphin`, `whale`, `otter`, `seal` e `beaver` são todos `aquatic_mammals`.

Isso permite responder uma pergunta que a acurácia sozinha nunca responde:
**quando o modelo erra, ele erra feio ou erra perto?**

| Erro | Gravidade |
|---|---|
| `dolphin` classificado como `whale` | Pequeno — mesma superclasse |
| `dolphin` classificado como `wardrobe` | Grave — o modelo não entendeu nada |

Se os erros fossem aleatórios, só cerca de **4%** cairiam na superclasse certa
(4 classes irmãs entre 99 alternativas). Qualquer valor bem acima disso prova
que o modelo aprendeu **estrutura semântica**, e não apenas aparências — mesmo
sem a hierarquia ter entrado no treino.

---

## Como executar

1. Abra `classificacao_cifar100.ipynb` no [Google Colab](https://colab.research.google.com/drive/1xnWDZd-Ewgc02h99O4n3mY9Dzd4S5G-n?usp=sharing)
2. Ative a GPU: `Ambiente de execução → Alterar o tipo de ambiente de execução → GPU (T4)`
3. Rode as células em ordem, de cima para baixo

Não precisa instalar nem baixar nada. O TensorFlow já vem no Colab e o
CIFAR-100 está embutido no Keras.

**Tempo estimado:** 12 a 18 minutos na GPU gratuita.

---

## O que o notebook faz

| Etapa | Descrição |
|---|---|
| 1. Preparação | Confirma a GPU e fixa as sementes aleatórias |
| 2. Dataset | Carrega os dois níveis de rótulo e monta a hierarquia |
| 3. Arquitetura | CNN de 3 blocos com BatchNorm, Dropout e augmentation |
| 4. Treino | 60 épocas com early stopping e redução de learning rate |
| 5. Avaliação | Acurácia top-1 e top-5, classes fáceis e difíceis, pares confundidos |
| 6. **Análise hierárquica** | Erros "próximos" contra erros "distantes" |
| 7. Predições | Imagens individuais com nível de confiança |
| 8. Exportação | Salva o modelo em `.keras` e baixa |

---

## Sobre o dataset

| | |
|---|---|
| Imagens | 60.000 coloridas de 32×32 |
| Treino | 50.000 (500 por classe) |
| Teste | 10.000 (100 por classe) |
| Classes finas | 100 |
| Superclasses | 20 (5 classes finas em cada) |

O dataset é **perfeitamente balanceado**, o que elimina uma fonte comum de
viés. Mas 500 exemplos por categoria é pouco para distinguir 100 classes — esse
é o principal limitador da acurácia.

---

## A arquitetura

Três blocos convolucionais, cada um seguindo o mesmo padrão:

```
Conv2D → BatchNorm → ReLU     (duas vezes)
MaxPooling2D                   reduz a resolução pela metade
Dropout                        desliga neurônios ao acaso
```

| Bloco | Filtros | Resolução | Dropout |
|---|---|---|---|
| 1 | 64 | 32×32 → 16×16 | 20% |
| 2 | 128 | 16×16 → 8×8 | 30% |
| 3 | 256 | 8×8 → 4×4 | 40% |

Cerca de **1,24 milhão de parâmetros**.

### As decisões que mais importam

**`use_bias=False` nas convoluções** — o BatchNorm logo abaixo já tem seu
próprio termo de deslocamento, tornando o viés da convolução redundante.

**Dropout crescente** — aumenta de 20% para 40% ao longo da rede, porque as
camadas finais têm mais parâmetros e decoram mais.

**`GlobalAveragePooling` em vez de `Flatten`** — achatar um mapa 4×4×256 geraria
4.096 valores e uma densa com milhões de parâmetros. O pooling global entrega
256 valores, com muito menos overfitting.

**Augmentation dentro do modelo** — as camadas `RandomFlip`, `RandomTranslation`
e `RandomZoom` ficam no próprio modelo, não no pipeline de dados. Assim se
desligam sozinhas na avaliação e vão junto quando o modelo é salvo.

Não há flip **vertical**: um carro de cabeça para baixo não é imagem realista, e
treinar com isso atrapalha.

---

## Por que treinar do zero

Modelos pré-treinados no ImageNet esperam imagens de 224×224. As nossas têm
32×32 — ampliá-las gera um borrão e desperdiça a maior parte do cálculo em
pixels interpolados.

Para imagens pequenas, uma CNN do zero é o caminho padrão: rápida, leve e
adequada à resolução. Transfer learning **também funciona** no CIFAR-100 e chega
mais longe em acurácia, ao custo de bem mais tempo — está listado nas extensões.

---

## Resultados

> Preencha após rodar o notebook.

| Métrica | Valor | Acaso |
|---|---|---|
| Acurácia top-1 | — | 1,00% |
| Acurácia top-5 | — | 5,00% |
| Acurácia na superclasse | — | 5,00% |
| Erros na superclasse certa | — | ~4% |

---

## Solução de problemas

**`ResourceExhaustedError` / falta de memória na GPU**
Reduza `batch_size=128` para `64` na célula de treino.

**O treino está lentíssimo**
Provavelmente está rodando em CPU. A célula 1 avisa se não houver GPU.

**`NameError` em alguma variável**
As variáveis vivem na memória do kernel. O notebook precisa ser rodado em
ordem, de cima para baixo.

**A perda de validação está menor que a de treino**
Isso é esperado nas primeiras épocas: dropout e augmentation só atuam no treino,
tornando-o artificialmente mais difícil que a validação.

**O download do CIFAR-100 falhou**
O Keras guarda em `~/.keras/datasets`. Apague a pasta e rode de novo para forçar
um novo download.

---

## Limitações

- **32×32 é pouquíssima informação.** Muitas imagens são ambíguas até para uma
  pessoa. Parte do erro é irredutível sem imagens melhores.
- **500 exemplos por classe é pouco** para 100 categorias.
- **A superclasse não entra no treino.** É usada apenas para analisar os erros.
- **Superclasses funcionais vão pior que visuais.** Uma CNN enxerga forma, cor e
  textura — não utilidade. Famílias como `household_electrical_devices` (relógio,
  teclado, lâmpada, telefone, televisão) não têm coerência visual alguma.

---

## Extensões possíveis

- **Transfer learning:** redimensionar para 128×128 e usar EfficientNetB0
  pré-treinado no ImageNet passa de 80% de acurácia
- **Saída dupla:** treinar simultaneamente para classe fina e superclasse, com
  duas cabeças e perda combinada — a hierarquia passaria a guiar o aprendizado
  em vez de só analisá-lo
- **Conexões residuais:** um ResNet pequeno permite ir mais fundo sem perder o
  gradiente
- **Decaimento por cosseno** na taxa de aprendizado costuma render 2 a 3 pontos
  a mais que o `ReduceLROnPlateau`

---

## Referências

- Krizhevsky, A. **Learning Multiple Layers of Features from Tiny Images.** 2009.
- [CIFAR-100 — página oficial](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Documentação do Keras — cifar100](https://keras.io/api/datasets/cifar100/)
