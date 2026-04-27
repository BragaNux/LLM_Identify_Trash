"""
SCRIPT 0 - FUSAO DOS DATASETS
================================
Combina o garbage_classification (15k imagens, fundo branco)
com o original/RealWaste (19k imagens, mundo real)
numa pasta dataset_final/ pronta pra treinar.

Mapeamento:
  original/glass  -> dataset_final/brown-glass (1/3)
                  -> dataset_final/green-glass (1/3)
                  -> dataset_final/white-glass (1/3)
  Resto           -> direto pra classe equivalente

Como usar:
  python script0_fusao.py

Nao precisa de dependencias extras.
"""

import os
import shutil
import random

BASE          = r"C:\Users\Braga\Desktop\EcoScan"
DATASET_A     = os.path.join(BASE, "garbage_classification")   # fundo branco
DATASET_B     = os.path.join(BASE, "original")                 # mundo real
DESTINO       = os.path.join(BASE, "dataset_final")

# Mapeamento das classes do dataset B para as classes finais
MAPA_B = {
    "battery":    "battery",
    "biological": "biological",
    "cardboard":  "cardboard",
    "clothes":    "clothes",
    "glass":      ["brown-glass", "green-glass", "white-glass"],  # divide em 3
    "metal":      "metal",
    "paper":      "paper",
    "plastic":    "plastic",
    "shoes":      "shoes",
    "trash":      "trash",
}

# Classes finais (as mesmas do dataset A)
CLASSES_FINAIS = [
    "battery", "biological", "brown-glass", "cardboard",
    "clothes", "green-glass", "metal", "paper",
    "plastic", "shoes", "trash", "white-glass"
]

def copiar_imagens(origem, destino, prefixo=""):
    """Copia todas as imagens de origem pra destino com prefixo no nome."""
    os.makedirs(destino, exist_ok=True)
    extensoes = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    imagens = [f for f in os.listdir(origem)
               if f.lower().endswith(extensoes)]
    for img in imagens:
        src  = os.path.join(origem, img)
        nome = f"{prefixo}_{img}" if prefixo else img
        dst  = os.path.join(destino, nome)
        # Evita sobrescrever se o nome já existe
        if os.path.exists(dst):
            base, ext = os.path.splitext(nome)
            dst = os.path.join(destino, f"{base}_{random.randint(1000,9999)}{ext}")
        shutil.copy2(src, dst)
    return len(imagens)

# ── Cria as pastas do destino ──────────────────────────────────────────────────
print("[INFO] Criando estrutura do dataset_final...")
for cls in CLASSES_FINAIS:
    os.makedirs(os.path.join(DESTINO, cls), exist_ok=True)

total = 0

# ── Copia dataset A (garbage_classification) ───────────────────────────────────
print("\n[FASE 1] Copiando garbage_classification...")
for cls in os.listdir(DATASET_A):
    pasta_src = os.path.join(DATASET_A, cls)
    if not os.path.isdir(pasta_src):
        continue
    if cls not in CLASSES_FINAIS:
        print(f"  [SKIP] Classe '{cls}' nao existe no destino, pulando...")
        continue
    pasta_dst = os.path.join(DESTINO, cls)
    n = copiar_imagens(pasta_src, pasta_dst, prefixo="A")
    print(f"  {cls:15} -> {n} imagens")
    total += n

# ── Copia dataset B (original/RealWaste) ──────────────────────────────────────
print("\n[FASE 2] Copiando original (RealWaste)...")
for cls_b, destinos in MAPA_B.items():
    pasta_src = os.path.join(DATASET_B, cls_b)
    if not os.path.isdir(pasta_src):
        print(f"  [SKIP] Pasta '{cls_b}' nao encontrada em original/")
        continue

    extensoes = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    imagens = [f for f in os.listdir(pasta_src)
               if f.lower().endswith(extensoes)]

    # Se a classe mapeia pra uma lista (glass -> 3 tipos de vidro)
    if isinstance(destinos, list):
        random.shuffle(imagens)
        partes = [imagens[i::len(destinos)] for i in range(len(destinos))]
        for cls_dst, parte in zip(destinos, partes):
            pasta_dst = os.path.join(DESTINO, cls_dst)
            n = 0
            for img in parte:
                src  = os.path.join(pasta_src, img)
                nome = f"B_{img}"
                dst  = os.path.join(pasta_dst, nome)
                if os.path.exists(dst):
                    base, ext = os.path.splitext(nome)
                    dst = os.path.join(pasta_dst,
                                       f"{base}_{random.randint(1000,9999)}{ext}")
                shutil.copy2(src, dst)
                n += 1
            print(f"  {cls_b} -> {cls_dst:15}: {n} imagens")
            total += n
    else:
        pasta_dst = os.path.join(DESTINO, destinos)
        n = copiar_imagens(pasta_src, pasta_dst, prefixo="B")
        print(f"  {cls_b:15} -> {n} imagens")
        total += n

# ── Resumo final ───────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"[OK] Fusao concluida!")
print(f"[OK] Total de imagens: {total}")
print()
print("[OK] Distribuicao final:")
for cls in sorted(CLASSES_FINAIS):
    pasta = os.path.join(DESTINO, cls)
    n = len([f for f in os.listdir(pasta)
             if f.lower().endswith(('.jpg','.jpeg','.png','.webp','.bmp'))])
    barra = "#" * (n // 100)
    print(f"  {cls:15}: {n:5} imagens  {barra}")

print(f"\n[PRONTO] Rode o script1_treino.py com DATASET_PATH = dataset_final")