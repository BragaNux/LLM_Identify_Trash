# EcoScan - Rodando no Notebook

Este guia explica como transferir e rodar o EcoScan no notebook após o modelo ter sido treinado no computador principal.

---

## O que você precisa copiar

Copie exatamente esses 3 arquivos do computador onde o treino rodou para uma pasta no notebook:

```
modelo_lixo.tflite
classes.txt
script2_app.py
```

Pode jogar num pendrive, mandar pelo Google Drive, ou copiar direto pela rede - tanto faz. Só não esquece os 3, porque sem qualquer um deles o app não abre.

---

## Configurando o ambiente

Abre o terminal na pasta onde você colocou os arquivos e cria uma venv:

```bash
python -m venv venv
```

Ativa:

**Windows:**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

Instala as dependências:

```bash
pip install tensorflow opencv-python numpy
```

---

## Rodando

```bash
python script2_app.py
```

A janela da câmera vai abrir automaticamente. Aponte para um objeto e o sistema classifica em tempo real.

**Controles:**
- `Q` - fechar o app
- `S` - salvar screenshot da tela atual

---

## Se der erro de câmera

O notebook pode ter mais de uma câmera (webcam integrada + externa). Se der erro ou abrir a câmera errada, edita a linha 235 do `script2_app.py`:

```python
# Troca o 0 pelo índice da câmera que quiser usar
cap = cv2.VideoCapture(0)   # webcam integrada
cap = cv2.VideoCapture(1)   # câmera externa
```

---

## Observações

- Não precisa levar o dataset, os scripts de treino, nem o modelo `.keras` - só o `.tflite` mesmo
- O notebook vai usar a CPU para inferência, o que é suficiente para rodar em tempo real
- Quanto melhor a iluminação do ambiente, melhor o resultado da classificação
