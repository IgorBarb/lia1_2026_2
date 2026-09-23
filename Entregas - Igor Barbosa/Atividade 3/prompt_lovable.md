# Prompt para o Lovable

> Copie tudo abaixo da linha e cole no Lovable como primeira mensagem.
> Antes disso, veja as instruções de preparação no final deste arquivo.

---

Construa uma aplicação web chamada **Materia** — um laboratório de visão
computacional que roda modelos ONNX inteiramente no navegador, sem servidor e
sem enviar nenhuma imagem para lugar nenhum.

O aplicativo vem com um modelo classificador de resíduos recicláveis já
embutido, mas seu diferencial é permitir que **qualquer pessoa carregue o
próprio modelo ONNX** e o teste imediatamente.

Toda a interface deve estar em **português do Brasil**.

## Stack

- React + TypeScript + Vite + Tailwind
- `onnxruntime-web` para inferência
- Nenhum backend, nenhuma chamada de rede para inferência
- Recharts para os gráficos

## Configuração do onnxruntime-web (crítico)

Esta parte quebra com facilidade. Siga exatamente:

1. Fixe uma versão exata de `onnxruntime-web` no `package.json` (não use `^`).
2. Aponte `ort.env.wasm.wasmPaths` para o CDN jsDelivr **na mesma versão exata**
   que está no `package.json`. Versão do JS diferente da versão do WASM é a
   causa número um de falha ao carregar.
3. Defina `ort.env.wasm.numThreads = 1`. O modo multithread exige cabeçalhos
   COOP/COEP e `SharedArrayBuffer`, que o preview não tem. Com uma thread
   funciona em qualquer lugar.
4. Deixe `ort.env.wasm.simd = true`.
5. Crie a sessão com `executionProviders: ['webgpu', 'wasm']` e capture o
   provider que efetivamente entrou em uso, para exibir na interface. Se o
   WebGPU falhar, caia para WASM sem quebrar.

## O modelo embutido

Os arquivos `modelo.onnx` e `metadados.json` estarão na pasta `public/`.
Carregue-os de `/modelo.onnx` e `/metadados.json` ao iniciar.

O contrato do modelo embutido:

```json
{
  "classes": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
  "entrada": { "nome": "imagem", "formato": "NHWC",
               "altura": 224, "largura": 224, "canais": 3,
               "tipo": "float32", "faixa_pixels": "0-255" },
  "saida":   { "nome": "output_0", "ativacao": "softmax" }
}
```

**Atenção ao pré-processamento.** A normalização está dentro do grafo do
modelo. Envie os pixels **crus, de 0 a 255**, como float32. Não divida por 255,
não subtraia média, não aplique nada. Errar isso não gera exceção — o modelo
simplesmente passa a dar respostas erradas com aparência de certeza.

Exiba os nomes das classes traduzidos na interface (papelão, vidro, metal,
papel, plástico, rejeito), mantendo o nome original visível em texto menor.

## Regra de ouro da implementação

**Nunca escreva no código o nome do tensor de entrada ou de saída.** Leia sempre
de `session.inputNames[0]` e `session.outputNames[0]`. É exatamente isso que
permite carregar o modelo de outra pessoa sem alterar uma linha.

## Telas

### 1. Classificar

A tela principal. Três formas de fornecer a imagem, em abas:

- **Arquivo** — arrastar e soltar, ou clicar para escolher
- **Câmera** — webcam ao vivo, com um botão de capturar e um modo contínuo que
  classifica a cada quadro
- **Exemplos** — quatro imagens de demonstração embutidas

O resultado aparece como:

- A classe vencedora em destaque grande, com o percentual
- Barras horizontais com todas as classes ordenadas por probabilidade, animadas
- Um selo de **latência** em milissegundos e o provider em uso (WebGPU ou WASM)

### 2. Meu modelo

Onde o usuário carrega o próprio ONNX. Deve funcionar assim:

1. Campo para arrastar um arquivo `.onnx`, ou colar uma URL
2. Ao carregar, abra a sessão e **descubra o contrato sozinho**:
   - nome e formato do tensor de entrada
   - detecte o layout: dimensões `[N, 3, H, W]` são **NCHW**;
     `[N, H, W, 3]` são **NHWC**
   - dimensões dinâmicas (string ou `-1`) assumem 224, mas deixe o usuário editar
   - número de classes pela dimensão da saída
3. Campo opcional para os rótulos: aceitar um `metadados.json` no mesmo formato
   do modelo embutido, ou uma lista colada separada por vírgula ou quebra de
   linha. Sem rótulos, use `Classe 0`, `Classe 1` e assim por diante.
4. Seletor de **pré-processamento**, porque não há como adivinhar:
   - `0 a 255` (padrão — é o do modelo embutido)
   - `0 a 1` (divide por 255)
   - `-1 a 1` (divide por 127,5 e subtrai 1)
   - `ImageNet` (divide por 255, subtrai média `[0.485, 0.456, 0.406]`,
     divide por desvio `[0.229, 0.224, 0.225]`)
5. **Detecção automática de softmax**: se a soma das saídas não for
   aproximadamente 1, aplique softmax e avise na interface que a saída era
   logits. Se já somar 1, não aplique de novo.

Mostre um **inspetor do modelo** com: tamanho do arquivo, versão do opset,
produtor, nomes e formatos de entrada e saída, layout detectado e número de
classes. É a tela que explica por que o modelo funcionou ou não.

Deixe claro que o arquivo nunca sai do navegador.

### 3. Por que essa resposta?

O diferencial do projeto. Implemente **mapa de sensibilidade por oclusão**:

1. Divida a imagem numa grade (padrão 8×8, ajustável de 4 a 16)
2. Para cada célula, gere uma cópia da imagem com aquele trecho coberto por um
   quadrado cinza
3. Rode a inferência em cada cópia e registre **quanto caiu** a probabilidade da
   classe originalmente vencedora
4. Normalize essas quedas de 0 a 1 e desenhe como um mapa de calor sobreposto à
   imagem, com controle de opacidade

A leitura é direta: as regiões que mais derrubam a confiança quando cobertas são
as que o modelo estava usando para decidir.

Rode em lotes quando o modelo aceitar batch dinâmico; caso contrário, uma
inferência por vez. Em ambos os casos exiba barra de progresso e mantenha a
interface responsiva, cedendo o controle entre os lotes.

### 4. Histórico

Registre cada classificação da sessão: miniatura, classe prevista, confiança,
latência e horário. Mostre um gráfico de barras com a distribuição das classes
já classificadas e a latência média. Botão para limpar. Guarde em memória
apenas, sem persistência.

## Detalhes que fazem diferença

**Aviso de incerteza.** Calcule a entropia da distribuição de saída. Quando a
confiança do topo ficar abaixo de um limiar ajustável (padrão 60%) ou a entropia
estiver alta, mostre um aviso claro: *"O modelo está em dúvida — pode ser algo
fora das categorias que ele conhece."* Isso é honesto e resolve o fato de um
classificador nunca conseguir dizer "não sei" por conta própria.

**Dicas de descarte.** Para cada classe do modelo embutido, exiba uma orientação
curta de descarte correto (como lavar antes, se é reciclável na coleta comum,
cuidados com vidro quebrado). Só para o modelo embutido; ao usar um modelo do
usuário, esconda essa seção.

**Comparar dois modelos.** Permita manter o modelo embutido e o modelo do
usuário carregados ao mesmo tempo e classificar a mesma imagem nos dois, lado a
lado, com as duas distribuições e as duas latências. É a forma mais direta de
mostrar o valor de carregar o próprio modelo.

**Medição de desempenho.** Um botão que roda 20 inferências seguidas e reporta
latência mínima, média e mediana, além de quadros por segundo estimados.

**Privacidade visível.** Um indicador fixo, discreto, informando que nenhuma
imagem sai do dispositivo. É verdade técnica aqui, e é um argumento real.

## Direção visual

Fuja do visual padrão de template. A referência é **instrumento de medição**,
não landing page.

- Tema escuro por padrão, com alternância para claro
- Fundo grafite bem escuro, quase preto, com cartões de fundo levemente mais
  claro e borda de 1px sutil
- Uma única cor de destaque: verde-limão elétrico, usada com parcimônia — só em
  valores ativos, barras e estados de foco
- Todos os números (percentuais, milissegundos, formatos de tensor) em fonte
  **monoespaçada**. Isso é o que dá cara de instrumento
- Texto em fonte sem serifa de boa legibilidade
- Cantos pouco arredondados, 6px no máximo
- Animações curtas e funcionais: barras que crescem, contadores que sobem.
  Nada de efeito decorativo
- Layout em grade, respirado, com hierarquia clara entre o resultado principal e
  os detalhes técnicos

## Robustez

Trate esses casos com mensagem clara em português, nunca com tela branca:

- Arquivo enviado não é um ONNX válido
- Modelo com formato de entrada que não é imagem (por exemplo, rank diferente
  de 4) — explique que o app só suporta classificadores de imagem
- Modelo de **detecção de objetos** (saída com 3 dimensões, como `[1, 84, 8400]`)
  — avise explicitamente que o app só suporta classificação, porque cada família
  de detector usa um formato de saída diferente e não há como decodificar
  genericamente
- Número de rótulos diferente do número de classes da saída
- Câmera negada ou indisponível
- WebGPU indisponível (caia para WASM em silêncio, apenas indicando no seletor)

## O que não fazer

- Não crie backend, banco de dados nem autenticação
- Não envie imagens ou modelos para qualquer serviço externo
- Não fixe no código os nomes dos tensores
- Não aplique normalização no modelo embutido
- Não aplique softmax duas vezes

---

## Preparação antes de colar o prompt

1. No projeto do Lovable, adicione os dois arquivos na pasta `public/`:
   - `modelo.onnx`
   - `metadados.json`
2. Se a interface do Lovable não aceitar o upload do `.onnx` por ser binário,
   conecte o projeto ao GitHub e faça o commit dos dois arquivos direto no
   repositório, dentro de `public/`.
3. Alternativa: suba o `modelo.onnx` num release do GitHub e, no prompt, troque
   `/modelo.onnx` pela URL pública. Nesse caso peça também que a origem envie
   CORS permitindo o domínio do Lovable.

## Ajustes depois da primeira geração

Se algo não funcionar, estes são os pedidos que resolvem a maioria dos casos:

- *"O modelo não carrega: fixe a versão de onnxruntime-web no package.json e use
  exatamente a mesma versão na URL de wasmPaths."*
- *"Defina ort.env.wasm.numThreads = 1 e remova qualquer dependência de
  SharedArrayBuffer."*
- *"As probabilidades estão erradas: confirme que a imagem é enviada como
  float32 de 0 a 255, no layout NHWC [1, 224, 224, 3], sem nenhuma
  normalização."*
- *"Leia os nomes dos tensores de session.inputNames[0] e
  session.outputNames[0], nunca escritos no código."*
