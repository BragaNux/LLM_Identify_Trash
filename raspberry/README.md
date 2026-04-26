# EcoScan - Rodando no Raspberry Pi

Este guia explica como transferir e rodar o EcoScan no Raspberry Pi após o modelo ter sido treinado no computador principal.

---

## O que você precisa copiar

Copie exatamente esses 3 arquivos para o Raspberry Pi:

```
modelo_lixo.tflite
classes.txt
script2_app.py
```

A forma mais fácil de transferir é via pendrive ou pela rede com SCP:

```bash
scp modelo_lixo.tflite classes.txt script2_app.py pi@<IP_DO_PI>:/home/pi/ecoscan/
```

Substitui `<IP_DO_PI>` pelo IP do seu Raspberry Pi na rede local.

---

## Configurando o ambiente

Conecta no Raspberry Pi (via SSH ou com monitor/teclado) e navega até a pasta dos arquivos:

```bash
cd /home/pi/ecoscan
```

Cria a venv:

```bash
python3 -m venv venv
source venv/bin/activate
```

Instala as dependências - no Pi usamos o `tflite-runtime` em vez do TensorFlow completo, que é muito pesado:

```bash
pip install tflite-runtime opencv-python numpy
```

> Se o `tflite-runtime` não instalar por incompatibilidade de versão do Python, tenta instalar o TensorFlow completo mesmo:
> ```bash
> pip install tensorflow
> ```
> O script detecta automaticamente qual está disponível e usa o correto.

---

## Ajustando o script para a câmera do Pi

A câmera do Raspberry Pi pode ser reconhecida de forma diferente pelo OpenCV. Edita a linha 235 do `script2_app.py`:

**Se estiver usando a câmera oficial do Pi (ribbon cable):**
```python
cap = cv2.VideoCapture(0)
```

**Se não abrir, tenta:**
```python
cap = cv2.VideoCapture('/dev/video0')
```

**Para câmera USB:**
```python
cap = cv2.VideoCapture(0)   # ou 1 se tiver outra câmera conectada
```

---

## Rodando

```bash
python script2_app.py
```

Se estiver acessando via SSH sem monitor, o OpenCV não consegue abrir a janela. Nesse caso, conecta um monitor ao Pi ou usa uma sessão com suporte a display (VNC, X11 forwarding).

---

## Dicas de performance

O Raspberry Pi é mais lento que um notebook, então algumas coisas ajudam:

**Reduz a resolução da câmera** - edita as linhas 236-237 do `script2_app.py`:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)   # era 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)   # era 720
```

**Classifica menos vezes por segundo** - edita a linha 263:
```python
if frame_count % 10 == 0:   # era 5, aumenta para classificar menos
```

Essas duas mudanças juntas já deixam o app bem mais fluido no Pi.

---

## Observações

- Não precisa levar o dataset, os scripts de treino, nem o modelo `.keras`
- O modelo `.tflite` foi exportado com quantização, o que reduz o tamanho e acelera a inferência em hardware sem GPU
- Quanto melhor a iluminação do ambiente, melhor o resultado - o Pi Camera Module tem sensor pequeno e sofre mais em ambientes escuros
