# EcoScan - Classificador de Resíduos com Visão Computacional

Sistema de classificação de lixo em tempo real utilizando câmera e aprendizado de máquina. O projeto foi desenvolvido para rodar tanto em notebooks quanto no Raspberry Pi, identificando 12 categorias de resíduos domésticos a partir de imagens capturadas pela câmera.

---

## Motivação

A separação correta do lixo ainda é um desafio no dia a dia - seja por falta de informação ou simplesmente por preguiça de verificar em qual lixeira cada material vai. A ideia do projeto foi criar um sistema simples e acessível que, ao apontar a câmera para um objeto, identifica automaticamente de que tipo de resíduo se trata e orienta o descarte correto.

A escolha do Raspberry Pi como plataforma de destino foi intencional: o objetivo é que o sistema possa ser instalado de forma permanente, por exemplo, em cima de uma lixeira, sem depender de um computador desktop.

---

## Tecnologias utilizadas

- **Python 3.11**
- **TensorFlow 2.x** - treinamento da CNN e exportação do modelo
- **MobileNetV2** - arquitetura base pré-treinada no ImageNet
- **OpenCV** - captura de vídeo e renderização da interface
- **NumPy** - processamento dos vetores de probabilidade
- **Matplotlib** - geração dos gráficos de treinamento

---

## Dataset

O projeto utiliza dois datasets combinados para aumentar a diversidade de imagens e melhorar a generalização do modelo:

**1. Garbage Classification (Kaggle - Mostafa Abla)**
15.150 imagens organizadas em 12 classes, coletadas principalmente em fundo neutro. Bom para aprender as características visuais de cada material.

**2. RealWaste / Garbage Classification v2 (Kaggle - sumn2u)**
~19.700 imagens coletadas em ambiente de aterro sanitário real. Muito mais diversidade de fundo, iluminação e condições, o que ajuda o modelo a generalizar para câmera real.

Os dois datasets foram fundidos em uma pasta `dataset_final/` com as mesmas 12 classes. O dataset B não tinha separação entre tipos de vidro, então as imagens da classe `glass` foram divididas aleatoriamente entre `brown-glass`, `green-glass` e `white-glass`.

**Classes reconhecidas:**

| Classe | Tradução | Descarte |
|--------|----------|----------|
| battery | Bateria | Descarte especial |
| biological | Organico | Lixo organico |
| brown-glass | Vidro Marrom | Coleta seletiva |
| cardboard | Papelao | Coleta seletiva |
| clothes | Roupa | Doacao ou descarte especial |
| green-glass | Vidro Verde | Coleta seletiva |
| metal | Metal | Coleta seletiva |
| paper | Papel | Coleta seletiva |
| plastic | Plastico | Coleta seletiva |
| shoes | Sapato | Doacao ou descarte especial |
| trash | Rejeito | Lixo comum |
| white-glass | Vidro Branco | Coleta seletiva |

---

## Arquitetura do modelo

A base é o **MobileNetV2** pré-treinado no ImageNet - escolhido por ser leve o suficiente para rodar no Raspberry Pi sem GPU, mas poderoso o suficiente para classificação de imagens do mundo real.

O treinamento foi dividido em duas fases:

**Fase 1 - Transfer Learning**
O MobileNetV2 é congelado. Apenas as camadas adicionadas no topo são treinadas: uma Dense(512) com regularização L2, Dropout(0.5), Dense(256) com L2, Dropout(0.4) e a camada de saída softmax com 12 classes. Learning rate de 0.001.

**Fase 2 - Fine-tuning**
As últimas 50 camadas do MobileNetV2 são descongeladas e retreinadas com learning rate muito menor (0.00005). Isso permite que o modelo ajuste as features de alto nível já aprendidas para o domínio específico de classificação de lixo.

**Augmentation aplicado no treino:**
- Rotação até 30 graus
- Zoom até 35%
- Flip horizontal
- Deslocamento horizontal e vertical até 20%
- Distorção (shear) até 20%
- Variação de brilho entre 50% e 150%
- Variação de canal de cor em até 30 unidades

O augmentation agressivo foi necessário porque os datasets originais têm imagens em condições controladas (fundo branco, boa iluminação), enquanto a câmera real captura objetos segurados na mão, em ambientes variados.

**Saída do treinamento:**
- `modelo_lixo.keras` - modelo completo para continuar treinando
- `modelo_lixo.tflite` - modelo quantizado para inferência no Raspberry Pi (~3 MB)
- `classes.txt` - lista ordenada das classes

---

## Como rodar

### Pré-requisitos

```bash
pip install tensorflow opencv-python numpy matplotlib
```

### 1. Preparar o dataset

Baixe os datasets do Kaggle e coloque na seguinte estrutura:

```
libras-projeto/
├── garbage_classification/    # Dataset A
│   ├── battery/
│   ├── biological/
│   └── ...
├── original/                  # Dataset B
│   ├── battery/
│   ├── glass/
│   └── ...
```

Rode o script de fusão:

```bash
python script0_fusao.py
```

Isso cria a pasta `dataset_final/` com todas as classes unificadas.

### 2. Treinar o modelo

```bash
python script1_treino.py
```

O treinamento roda em duas fases e pode levar entre 60 e 90 minutos dependendo do hardware. Ao final, gera `modelo_lixo.tflite` e `grafico_treino.png`.

### 3. Testar em tempo real

```bash
python script2_app.py
```

Aponte a câmera para um objeto. O sistema classifica a cada 5 frames e usa média móvel de 8 leituras para suavizar o resultado.

**Controles:**
- `Q` - sair
- `S` - salvar screenshot

---

## Interface

A interface foi construída diretamente com OpenCV, sem dependências de GUI externas, o que garante compatibilidade com o Raspberry Pi.

**Painel superior:** exibe o nome da classe identificada, a descrição de descarte e a porcentagem de confiança. A cor muda de acordo com a categoria.

**Retângulo central:** área de foco onde o objeto deve ser posicionado. Os cantos mudam de cor quando um objeto é identificado com confiança.

**Painel lateral:** mostra as 5 classes com maior probabilidade em tempo real, com barras de progresso coloridas por categoria.

**Rodapé:** histórico das últimas classificações da sessão.

**Estados da interface:**
- Câmera sem objeto ou muito escura: exibe "Nenhum objeto detectado"
- Objeto presente mas confiança baixa: exibe "Identificando..."
- Confiança acima de 55%: exibe o nome e categoria do objeto

---

## Rodando no Raspberry Pi

O modelo foi exportado em formato TFLite com quantização padrão, reduzindo o tamanho de ~10 MB para ~3 MB e acelerando a inferência em hardware limitado.

No Raspberry Pi, instale o runtime leve em vez do TensorFlow completo:

```bash
pip install tflite-runtime opencv-python numpy
```

O `script2_app.py` detecta automaticamente qual está disponível e usa o mais leve.

A câmera do Raspberry Pi é reconhecida normalmente pelo OpenCV como `VideoCapture(0)`.

---

## Estrutura do projeto

```
libras-projeto/
├── dataset_final/             # Dataset combinado (gerado pelo script0)
├── garbage_classification/    # Dataset A original
├── original/                  # Dataset B original
├── venv/                      # Ambiente virtual Python
├── script0_fusao.py           # Combina os dois datasets
├── script1_treino.py          # Treina o modelo CNN
├── script2_app.py             # App de classificacao em tempo real
├── modelo_lixo.keras          # Modelo completo (gerado pelo script1)
├── modelo_lixo.tflite         # Modelo para Raspberry Pi (gerado pelo script1)
├── classes.txt                # Lista de classes (gerado pelo script1)
└── grafico_treino.png         # Grafico de acuracia/loss (gerado pelo script1)
```

---

## Resultados

O modelo treinado com os dois datasets combinados e augmentation agressivo atingiu aproximadamente **92% de acurácia** na validação. Na prática com câmera real, o desempenho varia conforme a iluminação e o fundo da cena - objetos segurados na mão em ambiente bem iluminado são classificados corretamente na grande maioria dos casos.

As classes com melhor desempenho foram papelao, metal e plastic, que têm características visuais mais distintas. As classes de vidro (brown, green, white) tendem a se confundir entre si, o que é esperado dado que a diferença entre elas é predominantemente a cor.

---

## Possíveis melhorias

- Coletar imagens próprias dos objetos segurados na mão para retreinar com contexto real
- Adicionar uma classe "desconhecido" para quando o objeto não pertence a nenhuma categoria treinada
- Implementar detecção de objeto (bounding box) antes da classificação, para ignorar o fundo da cena
- Testar com EfficientNetB0, que tende a ter melhor acurácia com tamanho de modelo semelhante

---

## Referências

- Howard, A. et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv, 2017.
- Garbage Classification Dataset - Mostafa Abla (Kaggle) - https://www.kaggle.com/datasets/mostafaabla/garbage-classification
- Garbage Classification v2 - sumn2u (Kaggle) - https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2
- TensorFlow Lite - https://www.tensorflow.org/lite
