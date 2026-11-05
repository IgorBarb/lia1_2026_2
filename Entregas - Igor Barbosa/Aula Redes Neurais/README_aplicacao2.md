# Aplicação 2 — Rede Neural Convolucional (CNN) com o CIFAR-10

Notebook da segunda demonstração da aula de Redes Neurais. Ele treina **uma MLP e uma CNN nos mesmos dados** e compara as duas, mostrando na prática por que as redes convolucionais funcionam tão bem com imagens.

- **Arquivo:** `aplicacao2_cnn_cifar10.ipynb`
- **Dataset:** CIFAR-10 — 60 mil fotos coloridas 32×32 de 10 classes (avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio, caminhão). É baixado automaticamente pelo Keras.
- **Tempo de execução:** poucos minutos com GPU.

---

## Como rodar no Google Colab

1. Acesse [colab.research.google.com](https://colab.research.google.com) e vá em **Arquivo → Fazer upload de notebook**, escolhendo `aplicacao2_cnn_cifar10.ipynb`.
2. Ative a GPU: **Ambiente de execução → Alterar o tipo de ambiente de execução → GPU (T4)**.
3. Rode tudo com **Ambiente de execução → Executar tudo** (ou célula por célula com `Shift + Enter`).

Não é preciso instalar nada: o Colab já tem TensorFlow, scikit-learn e Matplotlib.

**Sem GPU?** Na célula de configurações, mude `MODO_RAPIDO = True`. O notebook passa a usar 10 mil imagens de treino e menos épocas, e roda em CPU num tempo aceitável (com acurácia um pouco menor).

---

## O que o notebook faz, passo a passo

| Etapa | O que acontece | Slide relacionado |
|---|---|---|
| 0. Configurações | Épocas, tamanho do lote e modo rápido em um só lugar | Mini-batch, iteração e época |
| 1. Bibliotecas | Importa TensorFlow/Keras, NumPy, Matplotlib e scikit-learn | — |
| 2. Dados | Carrega o CIFAR-10 (50 mil treino, 10 mil teste) | — |
| 3. Visualização | Exemplos de cada classe e uma imagem separada nos canais R, G e B | CNN: a ideia (imagem como volume) |
| 4. Preparação | Normaliza os pixels (0–1), separa 5 mil imagens para validação e converte os rótulos para one-hot | Treino, validação e teste; entrada e saída |
| 5. MLP | Rede totalmente conectada de referência (Flatten → 256 → 128 → 10) | Por que a MLP sofre com imagens? |
| 6. CNN | 3 blocos [Conv → ReLU → MaxPooling] + Dropout + Densa + Softmax | Campos receptivos, pesos compartilhados, pooling, juntando tudo, dropout |
| 7. Treino da CNN | Treino com early stopping (paciência 3, volta aos melhores pesos) | Early stopping |
| 8. Curvas | Acurácia e perda de treino × validação das duas redes | Overfitting |
| 9. Comparação | Acurácia no teste, número de parâmetros e tempo de treino | Por que a MLP sofre com imagens? |
| 10. Erros | Matriz de confusão, exemplos de erros e saída da softmax | Softmax |
| 11. Por dentro da CNN | Mapas de características de cada camada e filtros aprendidos | Por que profundidade?; pesos compartilhados |
| 12. Imagem própria | Envio de uma foto para a CNN classificar (opcional) | — |
| 13. Conclusões | Resumo e exercícios para a turma | — |

### As duas redes

| Modelo | Arquitetura | Parâmetros |
|---|---|---|
| MLP | Flatten (3.072) → Dense 256 → Dense 128 → Dense 10 | 820.874 |
| CNN | Conv 32 → Pool → Conv 64 → Pool → Conv 128 → Pool → Flatten → Dropout 0,5 → Dense 128 → Dense 10 | 356.810 |

Ambas usam ReLU nas camadas ocultas, softmax na saída, entropia cruzada categórica como função de custo e o otimizador Adam.

---

## Resultados esperados

Com GPU e as configurações padrão, os valores típicos no CIFAR-10 ficam por volta de:

- **MLP:** 45–52% de acurácia no teste
- **CNN:** 70–78% de acurácia no teste, **com menos da metade dos parâmetros**

Os números exatos variam a cada execução. **Importante:** o código foi testado de ponta a ponta, sem erros, mas com dados sintéticos de mesmo formato, porque o ambiente onde o notebook foi preparado não tinha acesso ao download do CIFAR-10. Por isso, rode o notebook inteiro uma vez no Colab antes da aula para conferir os resultados reais e ajustar a fala, se precisar.

---

## Sugestão de roteiro para a demonstração (~12 min)

**Antes da aula:** rode o notebook inteiro uma vez. Assim os gráficos já ficam prontos caso a conexão ou a GPU falhem na hora.

**Durante a aula:**

1. **Dados (2 min):** mostre os exemplos e a imagem separada em canais R, G e B.
2. **MLP (2 min):** mostre o `summary()` e destaque os ~786 mil parâmetros só da primeira camada; dispare o treino.
3. **CNN (3 min):** explique a tabela da arquitetura e dispare o treino da CNN enquanto comenta cada camada.
4. **Resultados (2 min):** curvas de treino × validação e o gráfico de comparação MLP × CNN.
5. **Erros (1 min):** matriz de confusão (gato × cachorro, automóvel × caminhão).
6. **Por dentro da CNN (2 min):** mapas de características camada a camada.
7. **Bônus:** se sobrar tempo, peça uma foto para a turma e teste na célula 12.

---

## Exercícios propostos (no final do notebook)

1. Remover o Dropout e o early stopping para ver o overfitting acontecer.
2. Mudar o número de filtros e observar parâmetros e acurácia.
3. Retirar uma camada de pooling e ver os parâmetros da camada densa explodirem.
4. Trocar o Adam pelo SGD e comparar a velocidade de aprendizado.
5. Aumentar a MLP e verificar se ela alcança a CNN.

---

## Rodando fora do Colab

```bash
pip install tensorflow scikit-learn matplotlib pillow jupyter
jupyter notebook aplicacao2_cnn_cifar10.ipynb
```

A célula 12 (envio de imagem) usa um recurso do Colab; fora dele, ela apenas exibe um aviso.
