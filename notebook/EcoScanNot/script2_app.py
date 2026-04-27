"""
SCRIPT 2 - APP EM TEMPO REAL
==============================
Classifica lixo em tempo real com camera.
Modo normal: camera ao vivo com classificacao automatica.
Modo lapiz: congela a imagem, voce desenha sobre o objeto,
            ENTER classifica so o que foi marcado.

Controles:
  [Q]     -> Sai
  [S]     -> Salva screenshot
  [L]     -> Ativa modo lapiz (congela camera)
  [R]     -> Cancela modo lapiz e volta a camera
  [ENTER] -> Classifica a area desenhada (so no modo lapiz)

Como usar:
  python script2_app.py

Dependencias:
  pip install tensorflow opencv-python numpy
"""

import cv2
import numpy as np
import os
import time
from collections import deque

try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# ── Configuracoes ──────────────────────────────────────────────────────────────
MODELO_PATH      = "modelo_lixo.tflite"
CLASSES_PATH     = "classes.txt"
IMG_SIZE         = (224, 224)
CONFIANCA_MINIMA = 0.55
SUAVIZACAO       = 8
RAIO_LAPIZ       = 14

# ── Dicionarios ────────────────────────────────────────────────────────────────
TRADUCAO = {
    "battery":     "Bateria",
    "biological":  "Organico",
    "brown-glass": "Vidro Marrom",
    "cardboard":   "Papelao",
    "clothes":     "Roupa",
    "green-glass": "Vidro Verde",
    "metal":       "Metal",
    "paper":       "Papel",
    "plastic":     "Plastico",
    "shoes":       "Sapato",
    "trash":       "Rejeito",
    "white-glass": "Vidro Branco",
}

DESCRICAO = {
    "battery":     "Descarte especial - nao misture com lixo comum",
    "biological":  "Lixo organico - compostavel",
    "brown-glass": "Vidro reciclavel - coleta seletiva",
    "cardboard":   "Papelao reciclavel - coleta seletiva",
    "clothes":     "Doacao ou descarte especial",
    "green-glass": "Vidro reciclavel - coleta seletiva",
    "metal":       "Metal reciclavel - coleta seletiva",
    "paper":       "Papel reciclavel - coleta seletiva",
    "plastic":     "Plastico reciclavel - coleta seletiva",
    "shoes":       "Doacao ou descarte especial",
    "trash":       "Rejeito - lixo comum",
    "white-glass": "Vidro reciclavel - coleta seletiva",
}

CORES = {
    "battery":     (45,  45,  220),
    "biological":  (50,  200, 80),
    "brown-glass": (30,  100, 160),
    "cardboard":   (30,  160, 255),
    "clothes":     (200, 100, 220),
    "green-glass": (50,  230, 140),
    "metal":       (180, 180, 190),
    "paper":       (60,  220, 220),
    "plastic":     (230, 180, 50),
    "shoes":       (80,  120, 180),
    "trash":       (100, 100, 110),
    "white-glass": (230, 230, 240),
}

ICONE_LIXEIRA = {
    "battery":     "[BATERIA]",
    "biological":  "[BIO]",
    "brown-glass": "[VIDRO]",
    "cardboard":   "[PAPEL]",
    "clothes":     "[ROUPA]",
    "green-glass": "[VIDRO]",
    "metal":       "[METAL]",
    "paper":       "[PAPEL]",
    "plastic":     "[PLAST]",
    "shoes":       "[ROUPA]",
    "trash":       "[LIXO]",
    "white-glass": "[VIDRO]",
}

# ── Carrega modelo e classes ───────────────────────────────────────────────────
if not os.path.exists(CLASSES_PATH):
    print("[ERRO] classes.txt nao encontrado.")
    exit()
if not os.path.exists(MODELO_PATH):
    print("[ERRO] modelo_lixo.tflite nao encontrado.")
    exit()

with open(CLASSES_PATH, encoding="utf-8") as f:
    classes = [linha.strip() for linha in f if linha.strip()]

interpreter = Interpreter(model_path=MODELO_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"[OK] Modelo carregado | Classes: {classes}")

# ── Classificacao ──────────────────────────────────────────────────────────────
def classificar(img):
    """Classifica uma imagem e retorna vetor de probabilidades."""
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0].copy()

def classificar_com_mascara(frame, mascara):
    """Recorta a regiao marcada, coloca fundo branco e classifica."""
    # Fundo branco — igual ao dataset de treino
    img_recortada = np.ones_like(frame, dtype=np.uint8) * 255
    img_recortada[mascara > 0] = frame[mascara > 0]

    # Bounding box da mascara para centralizar o objeto
    coords = cv2.findNonZero(mascara)
    if coords is not None:
        x, y, bw, bh = cv2.boundingRect(coords)
        x  = max(0, x)
        y  = max(0, y)
        bw = min(bw, frame.shape[1] - x)
        bh = min(bh, frame.shape[0] - y)
        if bw > 10 and bh > 10:
            img_recortada = img_recortada[y:y+bh, x:x+bw]
    return classificar(img_recortada)

# ── HUD ────────────────────────────────────────────────────────────────────────
def desenhar_painel(frame, classe, confianca, probs, fps, historico,
                    sem_objeto=False, modo_lapiz=False):
    h, w  = frame.shape[:2]
    cor   = CORES.get(classe, (180, 180, 180))
    nome  = TRADUCAO.get(classe, classe)
    desc  = DESCRICAO.get(classe, "")
    icone = ICONE_LIXEIRA.get(classe, "")

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (13, 13, 25), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.rectangle(frame, (0, 0), (w, 130), (18, 18, 35), -1)
    cor_linha = cor if confianca >= CONFIANCA_MINIMA else (50, 50, 70)
    cv2.rectangle(frame, (0, 128), (w, 131), cor_linha, -1)

    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 120), 1)

    # Badge do modo
    if modo_lapiz:
        cv2.rectangle(frame, (w - 230, 8), (w - 110, 32), (0, 120, 255), -1)
        cv2.putText(frame, "MODO LAPIZ", (w - 225, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    else:
        cv2.rectangle(frame, (w - 180, 8), (w - 110, 32), (0, 160, 80), -1)
        cv2.putText(frame, "AO VIVO", (w - 175, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    if sem_objeto:
        cv2.putText(frame, "Nenhum objeto detectado", (20, 65),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (60, 60, 80), 2)
        cv2.putText(frame, "Aponte a camera para um objeto", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 70), 1)
    elif confianca >= CONFIANCA_MINIMA:
        cv2.putText(frame, nome, (20, 65),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, cor, 2)
        cv2.putText(frame, icone, (w - 150, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        cv2.putText(frame, desc, (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 180), 1)
        cv2.putText(frame, f"{confianca*100:.1f}%", (20, 122),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 130), 1)
    else:
        cv2.putText(frame, "Identificando...", (20, 65),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (60, 60, 80), 2)
        cv2.putText(frame, "Mantenha o objeto parado na area", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 50, 70), 1)

    # Retangulo de foco (so no modo normal)
    if not modo_lapiz:
        cx, cy   = w // 2, h // 2 + 30
        tam      = min(w, h) // 3
        cor_foco = cor if confianca >= CONFIANCA_MINIMA else (40, 40, 60)
        comp, esp = 30, 3
        pts  = [(cx-tam,cy-tam),(cx+tam,cy-tam),(cx+tam,cy+tam),(cx-tam,cy+tam)]
        dirs = [(1,1),(-1,1),(-1,-1),(1,-1)]
        for (px, py), (dx, dy) in zip(pts, dirs):
            cv2.line(frame, (px, py), (px+dx*comp, py), cor_foco, esp)
            cv2.line(frame, (px, py), (px, py+dy*comp), cor_foco, esp)
        cv2.putText(frame, "APONTE O OBJETO AQUI", (cx-145, cy-tam-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_foco, 1)

    # Painel lateral
    px_start = max(w - 220, w // 2)
    cv2.rectangle(frame, (px_start-10, 140), (w, h-40), (18, 18, 35), -1)
    cv2.putText(frame, "PROBABILIDADES", (px_start, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 110), 1)

    top5 = np.argsort(probs)[::-1][:5]
    for i, idx in enumerate(top5):
        prob   = float(probs[idx])
        cls    = classes[idx]
        nome_c = TRADUCAO.get(cls, cls)
        cor_c  = CORES.get(cls, (120, 120, 120))
        y      = 185 + i * 42
        cv2.rectangle(frame, (px_start, y), (w-10, y+26), (28, 28, 45), -1)
        barra_w = int(prob * (w - 10 - px_start))
        if barra_w > 0:
            cv2.rectangle(frame, (px_start, y), (px_start+barra_w, y+26), cor_c, -1)
            sub = frame[y:y+26, px_start:px_start+barra_w]
            frame[y:y+26, px_start:px_start+barra_w] = (sub * 0.55).astype(np.uint8)
        cv2.putText(frame, nome_c, (px_start+5, y+17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 230), 1)
        cv2.putText(frame, f"{prob*100:.0f}%", (w-50, y+17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 230), 1)

    if historico:
        cv2.putText(frame, "HISTORICO", (px_start, h-140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 110), 1)
        for i, (hcls, hconf) in enumerate(list(historico)[-4:]):
            hnome = TRADUCAO.get(hcls, hcls)
            hcor  = CORES.get(hcls, (120, 120, 120))
            hy    = h - 120 + i * 22
            cv2.putText(frame, f"- {hnome} ({hconf*100:.0f}%)", (px_start, hy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, hcor, 1)

    # Rodape
    cv2.rectangle(frame, (0, h-38), (w, h), (18, 18, 35), -1)
    cv2.rectangle(frame, (0, h-40), (w, h-38), (40, 40, 60), -1)
    if modo_lapiz:
        cv2.putText(frame,
                    "[ENTER] Classificar   [R] Cancelar lapiz   [Q] Sair",
                    (15, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 80, 110), 1)
    else:
        cv2.putText(frame,
                    "[L] Modo Lapiz   [S] Screenshot   [Q] Sair   |   EcoScan v2.0",
                    (15, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 80, 110), 1)

# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("[ERRO] Nao foi possivel abrir a camera.")
    exit()

print("[OK] Camera aberta.")
print("  [L] ativar modo lapiz - desenhe sobre o objeto e pressione ENTER")
print("  [R] cancelar lapiz e voltar a camera")
print("  [Q] sair\n")

# ── Estado global ─────────────────────────────────────────────────────────────
historico_probs  = deque(maxlen=SUAVIZACAO)
historico_detect = deque(maxlen=10)

frame_count      = 0
t_anterior       = time.time()
fps              = 0.0
classe_atual     = "trash"
conf_atual       = 0.0
probs_atual      = np.zeros(len(classes))

modo_lapiz       = False
desenhando       = False
mascara_lapiz    = None
frame_congelado  = None

# ── Mouse callback ─────────────────────────────────────────────────────────────
def mouse_callback(event, x, y, flags, param):
    global desenhando, mascara_lapiz
    if not modo_lapiz or mascara_lapiz is None:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        desenhando = True
        cv2.circle(mascara_lapiz, (x, y), RAIO_LAPIZ, 255, -1)
    elif event == cv2.EVENT_MOUSEMOVE and desenhando:
        cv2.circle(mascara_lapiz, (x, y), RAIO_LAPIZ, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        desenhando = False
        cv2.circle(mascara_lapiz, (x, y), RAIO_LAPIZ, 255, -1)

cv2.namedWindow("EcoScan - Classificador de Residuos")
cv2.setMouseCallback("EcoScan - Classificador de Residuos", mouse_callback)

# ── Loop principal ─────────────────────────────────────────────────────────────
while cap.isOpened():

    if not modo_lapiz:
        # ── Modo normal: camera ao vivo ────────────────────────────────────────
        ret, frame_raw = cap.read()
        if not ret:
            break
        frame_raw = cv2.flip(frame_raw, 1)

        if frame_count % 5 == 0:
            probs_raw = classificar(frame_raw)
            historico_probs.append(probs_raw)
            probs_medio  = np.mean(historico_probs, axis=0)
            idx_medio    = int(np.argmax(probs_medio))
            classe_atual = classes[idx_medio]
            conf_atual   = float(probs_medio[idx_medio])
            probs_atual  = probs_medio
            if conf_atual >= CONFIANCA_MINIMA:
                if not historico_detect or historico_detect[-1][0] != classe_atual:
                    historico_detect.append((classe_atual, conf_atual))

        t_agora    = time.time()
        fps        = 1.0 / max(t_agora - t_anterior, 0.001)
        t_anterior = t_agora

        sem_objeto    = bool(np.mean(frame_raw) < 8)
        frame_display = frame_raw.copy()
        desenhar_painel(frame_display, classe_atual, conf_atual, probs_atual,
                        fps, historico_detect, sem_objeto=sem_objeto, modo_lapiz=False)

    else:
        # ── Modo lapiz: frame congelado + desenho ─────────────────────────────
        frame_display = frame_congelado.copy()

        # Overlay verde semitransparente onde o usuario ja desenhou
        if mascara_lapiz is not None and mascara_lapiz.max() > 0:
            overlay = frame_display.copy()
            overlay[mascara_lapiz > 0] = (0, 220, 80)
            cv2.addWeighted(overlay, 0.35, frame_display, 0.65, 0, frame_display)

        # Instrucao central
        h_d, w_d = frame_display.shape[:2]
        cv2.putText(frame_display,
                    "Desenhe sobre o objeto e pressione ENTER",
                    (w_d // 2 - 280, h_d - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 80), 2)

        desenhar_painel(frame_display, classe_atual, conf_atual, probs_atual,
                        fps, historico_detect, sem_objeto=False, modo_lapiz=True)

    cv2.imshow("EcoScan - Classificador de Residuos", frame_display)
    frame_count += 1

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("s"):
        nome_arq = f"screenshot_{int(time.time())}.png"
        cv2.imwrite(nome_arq, frame_display)
        print(f"[OK] Screenshot salvo: {nome_arq}")

    elif key == ord("l") and not modo_lapiz:
        # Ativa modo lapiz: congela o frame atual
        ret, frame_congelado = cap.read()
        if ret:
            frame_congelado = cv2.flip(frame_congelado, 1)
            mascara_lapiz   = np.zeros(frame_congelado.shape[:2], dtype=np.uint8)
            modo_lapiz      = True
            historico_probs.clear()
            print("[INFO] Modo lapiz ativado - desenhe sobre o objeto")

    elif key == ord("r"):
        # Cancela lapiz e volta a camera
        modo_lapiz      = False
        frame_congelado = None
        mascara_lapiz   = None
        historico_probs.clear()
        conf_atual      = 0.0
        probs_atual     = np.zeros(len(classes))
        print("[INFO] Voltando ao modo normal")

    elif key == 13 and modo_lapiz:
        # ENTER: classifica a area desenhada
        if mascara_lapiz is not None and mascara_lapiz.max() > 0:
            print("[INFO] Classificando area selecionada...")
            probs_raw    = classificar_com_mascara(frame_congelado, mascara_lapiz)
            idx          = int(np.argmax(probs_raw))
            classe_atual = classes[idx]
            conf_atual   = float(probs_raw[idx])
            probs_atual  = probs_raw
            if conf_atual >= CONFIANCA_MINIMA:
                if not historico_detect or historico_detect[-1][0] != classe_atual:
                    historico_detect.append((classe_atual, conf_atual))
            print(f"[OK] {TRADUCAO.get(classe_atual, classe_atual)} ({conf_atual*100:.1f}%)")
            # Volta ao modo normal
            modo_lapiz      = False
            frame_congelado = None
            mascara_lapiz   = None
        else:
            print("[AVISO] Desenhe sobre o objeto antes de pressionar ENTER")

cap.release()
cv2.destroyAllWindows()
print("[OK] App encerrado.")

import cv2
import numpy as np
import os
import time
from collections import deque

try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# ── Configuracoes ──────────────────────────────────────────────────────────────
MODELO_PATH      = "modelo_lixo.tflite"
CLASSES_PATH     = "classes.txt"
IMG_SIZE         = (224, 224)
CONFIANCA_MINIMA = 0.55
SUAVIZACAO       = 8

# ── Traducao das classes ───────────────────────────────────────────────────────
TRADUCAO = {
    "battery":     "Bateria",
    "biological":  "Organico",
    "brown-glass": "Vidro Marrom",
    "cardboard":   "Papelao",
    "clothes":     "Roupa",
    "green-glass": "Vidro Verde",
    "metal":       "Metal",
    "paper":       "Papel",
    "plastic":     "Plastico",
    "shoes":       "Sapato",
    "trash":       "Rejeito",
    "white-glass": "Vidro Branco",
}

DESCRICAO = {
    "battery":     "Descarte especial - nao misture com lixo comum",
    "biological":  "Lixo organico - compostavel",
    "brown-glass": "Vidro reciclavel - coleta seletiva",
    "cardboard":   "Papelao reciclavel - coleta seletiva",
    "clothes":     "Doacao ou descarte especial",
    "green-glass": "Vidro reciclavel - coleta seletiva",
    "metal":       "Metal reciclavel - coleta seletiva",
    "paper":       "Papel reciclavel - coleta seletiva",
    "plastic":     "Plastico reciclavel - coleta seletiva",
    "shoes":       "Doacao ou descarte especial",
    "trash":       "Rejeito - lixo comum",
    "white-glass": "Vidro reciclavel - coleta seletiva",
}

CORES = {
    "battery":     (45,  45,  220),
    "biological":  (50,  200, 80),
    "brown-glass": (30,  100, 160),
    "cardboard":   (30,  160, 255),
    "clothes":     (200, 100, 220),
    "green-glass": (50,  230, 140),
    "metal":       (180, 180, 190),
    "paper":       (60,  220, 220),
    "plastic":     (230, 180, 50),
    "shoes":       (80,  120, 180),
    "trash":       (100, 100, 110),
    "white-glass": (230, 230, 240),
}

ICONE_LIXEIRA = {
    "battery":     "[BATERIA]",
    "biological":  "[BIO]",
    "brown-glass": "[VIDRO]",
    "cardboard":   "[PAPEL]",
    "clothes":     "[ROUPA]",
    "green-glass": "[VIDRO]",
    "metal":       "[METAL]",
    "paper":       "[PAPEL]",
    "plastic":     "[PLAST]",
    "shoes":       "[ROUPA]",
    "trash":       "[LIXO]",
    "white-glass": "[VIDRO]",
}

# ── Carrega modelo e classes ───────────────────────────────────────────────────
if not os.path.exists(CLASSES_PATH):
    print("[ERRO] classes.txt nao encontrado.")
    exit()
if not os.path.exists(MODELO_PATH):
    print("[ERRO] modelo_lixo.tflite nao encontrado.")
    exit()

with open(CLASSES_PATH, encoding="utf-8") as f:
    classes = [linha.strip() for linha in f if linha.strip()]

interpreter = Interpreter(model_path=MODELO_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"[OK] Modelo carregado | Classes: {classes}")

# ── GrabCut ────────────────────────────────────────────────────────────────────
def aplicar_grabcut(frame, rect):
    """Aplica GrabCut no retangulo dado e retorna frame com fundo removido."""
    mask    = np.zeros(frame.shape[:2], dtype=np.uint8)
    bgd     = np.zeros((1, 65), dtype=np.float64)
    fgd     = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(frame, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        mask2  = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        result = frame * mask2[:, :, np.newaxis]
        return result, mask2
    except Exception:
        return frame, np.ones(frame.shape[:2], dtype=np.uint8)

def grabcut_com_mascara(frame, mascara_lapiz):
    """Aplica GrabCut usando mascara desenhada pelo usuario."""
    mask = np.full(frame.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    mask[mascara_lapiz > 0] = cv2.GC_FGD
    bgd  = np.zeros((1, 65), dtype=np.float64)
    fgd  = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(frame, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        mask2  = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        result = frame * mask2[:, :, np.newaxis]
        return result, mask2
    except Exception:
        return frame, np.ones(frame.shape[:2], dtype=np.uint8)

# ── Classificacao ──────────────────────────────────────────────────────────────
def classificar(img_segmentada):
    """Classifica a imagem ja segmentada."""
    img = cv2.resize(img_segmentada, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0].copy()

# ── HUD ────────────────────────────────────────────────────────────────────────
def desenhar_painel(frame, classe, confianca, probs, fps, historico,
                    sem_objeto=False, modo_lapiz=False):
    h, w  = frame.shape[:2]
    cor   = CORES.get(classe, (180, 180, 180))
    nome  = TRADUCAO.get(classe, classe)
    desc  = DESCRICAO.get(classe, "")
    icone = ICONE_LIXEIRA.get(classe, "")

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (13, 13, 25), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.rectangle(frame, (0, 0), (w, 130), (18, 18, 35), -1)
    cor_linha = cor if confianca >= CONFIANCA_MINIMA else (50, 50, 70)
    cv2.rectangle(frame, (0, 128), (w, 131), cor_linha, -1)

    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 120), 1)

    # Badge do modo atual
    if modo_lapiz:
        cv2.rectangle(frame, (w - 220, 8), (w - 110, 32), (0, 120, 255), -1)
        cv2.putText(frame, "MODO LAPIZ", (w - 215, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    else:
        cv2.rectangle(frame, (w - 220, 8), (w - 110, 32), (0, 180, 80), -1)
        cv2.putText(frame, "AUTO", (w - 195, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    if sem_objeto:
        cv2.putText(frame, "Nenhum objeto detectado", (20, 65),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (60, 60, 80), 2)
        cv2.putText(frame, "Aponte a camera para um objeto", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 70), 1)
    elif confianca >= CONFIANCA_MINIMA:
        cv2.putText(frame, nome, (20, 65),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, cor, 2)
        cv2.putText(frame, icone, (w - 150, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        cv2.putText(frame, desc, (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 180), 1)
        cv2.putText(frame, f"{confianca*100:.1f}%", (20, 122),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 130), 1)
    else:
        cv2.putText(frame, "Identificando...", (20, 65),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (60, 60, 80), 2)
        cv2.putText(frame, "Mantenha o objeto parado na area", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 50, 70), 1)

    # Retangulo de foco
    cx, cy   = w // 2, h // 2 + 30
    tam      = min(w, h) // 3
    cor_foco = cor if confianca >= CONFIANCA_MINIMA else (40, 40, 60)
    comp, esp = 30, 3
    pts  = [(cx-tam,cy-tam),(cx+tam,cy-tam),(cx+tam,cy+tam),(cx-tam,cy+tam)]
    dirs = [(1,1),(-1,1),(-1,-1),(1,-1)]
    for (px, py), (dx, dy) in zip(pts, dirs):
        cv2.line(frame, (px, py), (px+dx*comp, py), cor_foco, esp)
        cv2.line(frame, (px, py), (px, py+dy*comp), cor_foco, esp)
    cv2.putText(frame, "APONTE O OBJETO AQUI", (cx-145, cy-tam-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_foco, 1)

    # Painel lateral
    px_start = max(w - 220, w // 2)
    cv2.rectangle(frame, (px_start-10, 140), (w, h-40), (18, 18, 35), -1)
    cv2.putText(frame, "PROBABILIDADES", (px_start, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 110), 1)

    top5 = np.argsort(probs)[::-1][:5]
    for i, idx in enumerate(top5):
        prob   = float(probs[idx])
        cls    = classes[idx]
        nome_c = TRADUCAO.get(cls, cls)
        cor_c  = CORES.get(cls, (120, 120, 120))
        y      = 185 + i * 42
        cv2.rectangle(frame, (px_start, y), (w-10, y+26), (28, 28, 45), -1)
        barra_w = int(prob * (w - 10 - px_start))
        if barra_w > 0:
            cv2.rectangle(frame, (px_start, y), (px_start+barra_w, y+26), cor_c, -1)
            sub = frame[y:y+26, px_start:px_start+barra_w]
            frame[y:y+26, px_start:px_start+barra_w] = (sub * 0.55).astype(np.uint8)
        cv2.putText(frame, nome_c, (px_start+5, y+17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 230), 1)
        cv2.putText(frame, f"{prob*100:.0f}%", (w-50, y+17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 230), 1)

    if historico:
        cv2.putText(frame, "HISTORICO", (px_start, h-140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 110), 1)
        for i, (hcls, hconf) in enumerate(list(historico)[-4:]):
            hnome = TRADUCAO.get(hcls, hcls)
            hcor  = CORES.get(hcls, (120, 120, 120))
            hy    = h - 120 + i * 22
            cv2.putText(frame, f"- {hnome} ({hconf*100:.0f}%)", (px_start, hy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, hcor, 1)

    # Rodape
    cv2.rectangle(frame, (0, h-38), (w, h), (18, 18, 35), -1)
    cv2.rectangle(frame, (0, h-40), (w, h-38), (40, 40, 60), -1)
    if modo_lapiz:
        cv2.putText(frame,
                    "[ENTER] Classificar   [R] Resetar   [Q] Sair   [S] Screenshot",
                    (15, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 110), 1)
    else:
        cv2.putText(frame,
                    "[L] Modo Lapiz   [R] Resetar   [Q] Sair   [S] Screenshot   |   EcoScan v2.0",
                    (15, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 110), 1)

# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("[ERRO] Nao foi possivel abrir a camera.")
    exit()

print("[OK] Camera aberta.")
print("  [L] ativar modo lapiz - desenhe sobre o objeto e pressione ENTER")
print("  [R] resetar mascara para automatico")
print("  [Q] sair\n")

# ── Estado global ─────────────────────────────────────────────────────────────
historico_probs  = deque(maxlen=SUAVIZACAO)
historico_detect = deque(maxlen=10)

frame_count  = 0
t_anterior   = time.time()
fps          = 0.0
classe_atual = "trash"
conf_atual   = 0.0
probs_atual  = np.zeros(len(classes))

modo_lapiz      = False       # True = usuario esta desenhando
desenhando      = False       # True = botao do mouse pressionado
mascara_lapiz   = None        # Canvas onde o usuario desenha
frame_congelado = None        # Frame pausado para o usuario desenhar
RAIO_LAPIZ      = 12          # Espessura do lapis em pixels

# ── Callbacks do mouse ─────────────────────────────────────────────────────────
def mouse_callback(event, x, y, flags, param):
    global desenhando, mascara_lapiz
    if not modo_lapiz or mascara_lapiz is None:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        desenhando = True
    elif event == cv2.EVENT_MOUSEMOVE and desenhando:
        cv2.circle(mascara_lapiz, (x, y), RAIO_LAPIZ, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        desenhando = False
        cv2.circle(mascara_lapiz, (x, y), RAIO_LAPIZ, 255, -1)

cv2.namedWindow("EcoScan - Classificador de Residuos")
cv2.setMouseCallback("EcoScan - Classificador de Residuos", mouse_callback)

# ── Loop principal ─────────────────────────────────────────────────────────────
while cap.isOpened():
    ret, frame_raw = cap.read()
    if not ret:
        break

    frame_raw = cv2.flip(frame_raw, 1)
    h, w      = frame_raw.shape[:2]

    # Retangulo central para GrabCut automatico
    tam  = min(w, h) // 3
    cx   = w // 2
    cy   = h // 2 + 30
    rect = (cx - tam, cy - tam, tam * 2, tam * 2)

    if modo_lapiz:
        # ── Modo lapiz: frame congelado + desenho do usuario ──────────────────
        if frame_congelado is None:
            frame_congelado = frame_raw.copy()
            mascara_lapiz   = np.zeros(frame_raw.shape[:2], dtype=np.uint8)

        frame_display = frame_congelado.copy()

        # Mostra o que o usuario ja desenhou em verde semitransparente
        overlay_lapiz = frame_display.copy()
        overlay_lapiz[mascara_lapiz > 0] = (0, 220, 80)
        cv2.addWeighted(overlay_lapiz, 0.4, frame_display, 0.6, 0, frame_display)

        # Instrucao no centro
        cv2.putText(frame_display, "Desenhe sobre o objeto e pressione ENTER",
                    (cx - 280, cy + tam + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 2)

        fps_disp = fps
        desenhar_painel(frame_display, classe_atual, conf_atual, probs_atual,
                        fps_disp, historico_detect,
                        sem_objeto=False, modo_lapiz=True)
        cv2.imshow("EcoScan - Classificador de Residuos", frame_display)

    else:
        # ── Modo automatico: GrabCut no retangulo central ─────────────────────
        frame_congelado = None
        mascara_lapiz   = None

        if frame_count % 5 == 0:
            # Aplica GrabCut automatico
            img_seg, mask2 = aplicar_grabcut(frame_raw, rect)

            # Classifica a imagem segmentada
            probs_raw = classificar(img_seg)
            historico_probs.append(probs_raw)

            probs_medio  = np.mean(historico_probs, axis=0)
            idx_medio    = int(np.argmax(probs_medio))
            classe_atual = classes[idx_medio]
            conf_atual   = float(probs_medio[idx_medio])
            probs_atual  = probs_medio

            if conf_atual >= CONFIANCA_MINIMA:
                if not historico_detect or historico_detect[-1][0] != classe_atual:
                    historico_detect.append((classe_atual, conf_atual))

            # Mostra contorno do objeto segmentado
            contornos, _ = cv2.findContours(
                mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contornos:
                maior = max(contornos, key=cv2.contourArea)
                cor_cont = CORES.get(classe_atual, (100, 100, 100))
                cv2.drawContours(frame_raw, [maior], -1, cor_cont, 2)

        t_agora    = time.time()
        fps        = 1.0 / max(t_agora - t_anterior, 0.001)
        t_anterior = t_agora

        sem_objeto = bool(np.mean(frame_raw) < 8)
        frame_display = frame_raw.copy()
        desenhar_painel(frame_display, classe_atual, conf_atual, probs_atual,
                        fps, historico_detect,
                        sem_objeto=sem_objeto, modo_lapiz=False)
        cv2.imshow("EcoScan - Classificador de Residuos", frame_display)

    frame_count += 1
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("s"):
        nome_arq = f"screenshot_{int(time.time())}.png"
        cv2.imwrite(nome_arq, frame_display)
        print(f"[OK] Screenshot salvo: {nome_arq}")

    elif key == ord("l"):
        # Ativa modo lapiz
        modo_lapiz      = True
        frame_congelado = None
        mascara_lapiz   = None
        historico_probs.clear()
        print("[INFO] Modo lapiz ativado - desenhe sobre o objeto")

    elif key == ord("r"):
        # Reseta para modo automatico
        modo_lapiz      = False
        frame_congelado = None
        mascara_lapiz   = None
        historico_probs.clear()
        conf_atual      = 0.0
        probs_atual     = np.zeros(len(classes))
        print("[INFO] Resetado para modo automatico")

    elif key == 13 and modo_lapiz:
        # ENTER: aplica GrabCut com a mascara do lapiz e classifica
        if frame_congelado is not None and mascara_lapiz is not None:
            if mascara_lapiz.max() > 0:
                print("[INFO] Aplicando segmentacao com mascara do lapiz...")
                img_seg, _ = grabcut_com_mascara(frame_congelado, mascara_lapiz)
                probs_raw  = classificar(img_seg)
                historico_probs.clear()
                historico_probs.append(probs_raw)
                idx_medio    = int(np.argmax(probs_raw))
                classe_atual = classes[idx_medio]
                conf_atual   = float(probs_raw[idx_medio])
                probs_atual  = probs_raw
                if conf_atual >= CONFIANCA_MINIMA:
                    if not historico_detect or historico_detect[-1][0] != classe_atual:
                        historico_detect.append((classe_atual, conf_atual))
                print(f"[OK] Resultado: {TRADUCAO.get(classe_atual, classe_atual)} ({conf_atual*100:.1f}%)")
                # Volta ao modo automatico apos classificar
                modo_lapiz      = False
                frame_congelado = None
                mascara_lapiz   = None
            else:
                print("[AVISO] Desenhe sobre o objeto antes de pressionar ENTER")

cap.release()
cv2.destroyAllWindows()
print("[OK] App encerrado.")