# -*- coding: utf-8 -*-
"""
PRÓXIMO PASSO: de "modelo treinado em Madrid" até "modelo aplicado em Amsterdam"
==================================================================================
Este script assume que já correste o Notebook 3 (Preprocessing) e que existe o
ficheiro '../data/preprocessed/preprocessed_data.pkl' com X_madrid, y_madrid,
X_amsterdam, y_amsterdam, feature_names, scaler, pixel_ids_madrid, pixel_ids_amsterdam.

Ordem lógica (responde à pergunta 4):
  PARTE A — Confirmar que o modelo generaliza bem DENTRO de Madrid (validação cruzada)
  PARTE B — Treinar o modelo final em Madrid (agora usando XGBoost em vez de Random Forest)
  PARTE C — Zero-shot: aplicar esse modelo em Amsterdam sem nenhum rótulo local
  PARTE D — Few-shot: usar poucos rótulos de Amsterdam para calcular protótipos por classe
  PARTE E — Diagnóstico completo (pergunta 7): matriz de confusão, mapa espacial,
            incerteza (entropia/KL) e importância de variáveis
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

# XGBoost precisa de: pip install xgboost
from xgboost import XGBClassifier

RANDOM_STATE = 42
CLASS_LABELS_MADRID    = {1: 'Pre-1960',  2: '1960-1984', 3: '1984-2004', 4: '2004-2024'}
CLASS_LABELS_AMSTERDAM = {1: 'Pre-1945',  2: '1945-1984', 3: '1984-2004', 4: '2004-2024'}


# ==================================================================================
# CARREGAR OS DADOS JÁ PRÉ-PROCESSADOS (saída do Notebook 3)
# ==================================================================================
with open('../data/preprocessed/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_madrid            = data['X_madrid']
y_madrid            = data['y_madrid']
X_amsterdam         = data['X_amsterdam']
y_amsterdam         = data['y_amsterdam']            # usado só para AVALIAR, não para treinar
feature_names       = data['feature_names']
pixel_ids_madrid    = data['pixel_ids_madrid']
pixel_ids_amsterdam = data['pixel_ids_amsterdam']

print(f"Madrid:    X={X_madrid.shape}  y com {len(np.unique(y_madrid))} classes")
print(f"Amsterdam: X={X_amsterdam.shape}  y com {len(np.unique(y_amsterdam))} classes")


# ==================================================================================
# PARTE A — VALIDAÇÃO CRUZADA EM MADRID (confirmar antes de avançar para Amsterdam)
# ==================================================================================
# Isto responde à tua intuição na pergunta 4: sim, o primeiro passo é CONFIRMAR
# que o modelo generaliza bem dentro da própria Madrid antes de sequer tentar
# transferi-lo para outra cidade. Se o modelo não é estável em Madrid, não faz
# sentido gastar esforço a adaptá-lo a Amsterdam.

N_FOLDS, N_REPEATS = 5, 5
XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=5,                 # classes 1..4 -> XGBoost precisa de rótulos 0..3 (ver abaixo)
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric='mlogloss',
)

def to_xgb_labels(y):
    """XGBoost exige classes 0-indexed; as nossas classes são 1..4."""
    return y - 1

def from_xgb_labels(y):
    return y + 1

rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
all_f1 = []
oof_true, oof_pred, oof_proba, oof_idx = [], [], [], []

print(f"\n[PARTE A] Validação cruzada em Madrid — {N_FOLDS}-fold x {N_REPEATS}-repeat "
      f"({N_FOLDS * N_REPEATS} ajustes)")
for fit_num, (tr_idx, val_idx) in enumerate(rskf.split(X_madrid, y_madrid), 1):
    X_tr, X_val = X_madrid[tr_idx], X_madrid[val_idx]
    y_tr, y_val = to_xgb_labels(y_madrid[tr_idx]), y_madrid[val_idx]

    clf = XGBClassifier(**XGB_PARAMS)
    clf.fit(X_tr, y_tr)

    y_pred_raw = clf.predict(X_val)
    y_pred     = from_xgb_labels(y_pred_raw)
    y_proba    = clf.predict_proba(X_val)

    f1 = f1_score(y_val, y_pred, average='macro')
    all_f1.append(f1)
    oof_true.extend(y_val.tolist())
    oof_pred.extend(y_pred.tolist())
    oof_proba.extend(y_proba.tolist())
    oof_idx.extend(val_idx.tolist())
    print(f"  Fit {fit_num:2d}/{N_FOLDS*N_REPEATS}: macro-F1 = {f1:.4f}")

print(f"\n  >> Macro-F1 médio (Madrid, XGBoost): {np.mean(all_f1):.4f} +/- {np.std(all_f1):.4f}")
print(f"     (compara com Random Forest original: 0.6179 +/- 0.0043)")

oof_true  = np.array(oof_true)
oof_pred  = np.array(oof_pred)
oof_proba = np.array(oof_proba)
oof_idx   = np.array(oof_idx)


# ==================================================================================
# PARTE B — TREINAR O MODELO FINAL EM MADRID (todos os dados, para usar em Amsterdam)
# ==================================================================================
print("\n[PARTE B] Treinando modelo final XGBoost com todos os dados de Madrid...")
model_final = XGBClassifier(**XGB_PARAMS)
model_final.fit(X_madrid, to_xgb_labels(y_madrid))
print("  Concluído.")


# ==================================================================================
# PARTE C — ZERO-SHOT: aplicar o modelo de Madrid em Amsterdam SEM rótulos locais
# ==================================================================================
print("\n[PARTE C] Zero-shot em Amsterdam (nenhum rótulo local usado)...")
y_pred_zero_raw = model_final.predict(X_amsterdam)
y_pred_zero     = from_xgb_labels(y_pred_zero_raw)
proba_zero      = model_final.predict_proba(X_amsterdam)

f1_zero  = f1_score(y_amsterdam, y_pred_zero, average='macro')
acc_zero = accuracy_score(y_amsterdam, y_pred_zero)
print(f"  Macro-F1 zero-shot: {f1_zero:.4f}   |  Accuracy: {acc_zero:.4f}")
print(classification_report(
    y_amsterdam, y_pred_zero,
    target_names=[f'Classe {c}' for c in sorted(np.unique(y_amsterdam))],
    digits=4,
))


# ==================================================================================
# PARTE D — FEW-SHOT: transferência por protótipos (nearest-centroid)
# ==================================================================================
# IMPORTANTE: esta etapa NÃO usa o XGBoost. Os protótipos são calculados
# diretamente sobre os 60 atributos padronizados (X). É por isso que o modelo
# escolhido no Stage 1 (RF ou XGBoost) só afeta o zero-shot — a transferência
# em si é sempre feita da mesma forma.

def prototype_predict(X_support, y_support, X_query):
    classes    = np.unique(y_support)
    prototypes = np.stack([X_support[y_support == c].mean(axis=0) for c in classes])
    dists      = np.linalg.norm(X_query[:, None, :] - prototypes[None, :, :], axis=2)
    return classes[dists.argmin(axis=1)]

def prototype_predict_proba(X_support, y_support, X_query):
    classes    = np.unique(y_support)
    prototypes = np.stack([X_support[y_support == c].mean(axis=0) for c in classes])
    dists      = np.linalg.norm(X_query[:, None, :] - prototypes[None, :, :], axis=2)
    neg_d      = -dists
    neg_d     -= neg_d.max(axis=1, keepdims=True)
    exp_d      = np.exp(neg_d)
    return classes, exp_d / exp_d.sum(axis=1, keepdims=True)

SHOTS_PER_CLASS = [5, 10, 25, 50, 100, 200]   # nº de pixels rotulados de Amsterdam, por classe
N_TRIALS        = 10                           # repetições para estimar variância
classes_ams     = np.unique(y_amsterdam)
rng             = np.random.default_rng(RANDOM_STATE)

print("\n[PARTE D] Few-shot prototype transfer em Amsterdam...")
proto_results = {n: [] for n in SHOTS_PER_CLASS}
for n_shots in SHOTS_PER_CLASS:
    for _ in range(N_TRIALS):
        support_idx = []
        for cls in classes_ams:
            cls_idx = np.where(y_amsterdam == cls)[0]
            chosen  = rng.choice(cls_idx, min(n_shots, len(cls_idx)), replace=False)
            support_idx.extend(chosen.tolist())
        support_idx = np.array(support_idx)
        query_mask  = np.ones(len(y_amsterdam), dtype=bool)
        query_mask[support_idx] = False

        y_pred = prototype_predict(X_amsterdam[support_idx], y_amsterdam[support_idx],
                                    X_amsterdam[query_mask])
        f1 = f1_score(y_amsterdam[query_mask], y_pred, average='macro', zero_division=0)
        proto_results[n_shots].append(f1)

    mn, sd = np.mean(proto_results[n_shots]), np.std(proto_results[n_shots])
    print(f"  {n_shots:4d} amostras/classe: macro-F1 = {mn:.4f} +/- {sd:.4f}")


# ==================================================================================
# PARTE E — DIAGNÓSTICO COMPLETO (pergunta 7)
# ==================================================================================

# --- E.1: Matriz de confusão (Madrid CV + Amsterdam zero-shot) -------------------
def plot_confusion(y_true, y_pred, labels_map, title, ax):
    classes = sorted(np.unique(y_true))
    cm      = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    xl = [f'C{c}\n{labels_map[c]}' for c in classes]
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(xl, fontsize=7)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(xl, fontsize=7)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f'{cm_norm[i,j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if cm_norm[i, j] > 0.6 else 'black')
    ax.set_xlabel('Previsto'); ax.set_ylabel('Real'); ax.set_title(title, fontsize=9, fontweight='bold')
    return im

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
plot_confusion(oof_true, oof_pred, CLASS_LABELS_MADRID,
               'Madrid — CV out-of-fold (XGBoost)', axes[0])
plot_confusion(y_amsterdam, y_pred_zero, CLASS_LABELS_AMSTERDAM,
               'Amsterdam — zero-shot', axes[1])
plt.tight_layout()
plt.savefig('diagnostico_matrizes_confusao.png', dpi=130)
plt.show()
print("\n[E.1] Matrizes de confusão guardadas em diagnostico_matrizes_confusao.png")
print("      -> Aqui confirma-se, ou não, a confusão clássica Classe 1 <-> Classe 2")
print("         (ambas anteriores a 1984, sem sinal direto de construção no satélite).")


# --- E.2: Mapa espacial 2D (localização geográfica de acertos/erros) -------------
def make_pixel_grid(pixel_ids, values, fill_value):
    px = pixel_ids[:, 0].astype(int); px -= px.min()
    py = pixel_ids[:, 1].astype(int); py -= py.min()
    H, W = py.max() + 1, px.max() + 1
    if values.ndim == 1:
        img = np.full((H, W), fill_value, dtype=np.float32); img[py, px] = values
    else:
        img = np.full((H, W, values.shape[1]), fill_value, dtype=np.float32); img[py, px] = values
    return img

def plot_spatial_errors(pixel_ids, y_true, y_pred, title):
    correct = (y_true == y_pred)
    vals = np.where(correct[:, None], np.array([[0, 0, 0]], dtype=np.float32),
                     np.array([[1, 0, 0]], dtype=np.float32))
    img = make_pixel_grid(pixel_ids, vals, fill_value=0.6)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(img); ax.axis('off')
    ax.set_title(title, fontweight='bold')
    ax.legend(handles=[mpatches.Patch(color='black', label='Correto'),
                        mpatches.Patch(color='red', label='Incorreto'),
                        mpatches.Patch(facecolor=(0.6, 0.6, 0.6), label='Sem edifício')],
               loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig('diagnostico_mapa_espacial_amsterdam.png', dpi=130)
    plt.show()

plot_spatial_errors(pixel_ids_amsterdam, y_amsterdam, y_pred_zero,
                     'Mapa espacial de erros — Amsterdam zero-shot')
print("[E.2] Mapa espacial guardado em diagnostico_mapa_espacial_amsterdam.png")
print("      -> Procura blocos/zonas onde os erros se concentram geograficamente")
print("         (indica bairros onde vale a pena priorizar rotulagem manual).")


# --- E.3: Incerteza (entropia de Shannon e KL da classe verdadeira) --------------
def uncertainty_maps(pixel_ids, proba, y_true, title_prefix):
    eps = 1e-10
    H_vals  = -(proba * np.log2(proba + eps)).sum(axis=1)
    p_true  = proba[np.arange(len(y_true)), y_true - 1]
    kl_vals = -np.log2(p_true + eps)

    H_img  = make_pixel_grid(pixel_ids, H_vals.astype(np.float32),  fill_value=np.nan)
    kl_img = make_pixel_grid(pixel_ids, kl_vals.astype(np.float32), fill_value=np.nan)
    cmap = plt.cm.hot_r.copy(); cmap.set_bad((0.6, 0.6, 0.6))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im0 = axes[0].imshow(H_img, cmap=cmap, vmin=0, vmax=2); axes[0].axis('off')
    axes[0].set_title(f'{title_prefix}\nEntropia de Shannon (claro=confiante)', fontsize=9)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(kl_img, cmap=cmap, vmin=0, vmax=2); axes[1].axis('off')
    axes[1].set_title(f'{title_prefix}\nKL da classe real (claro=acertou com confiança)', fontsize=9)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    plt.savefig('diagnostico_incerteza_amsterdam.png', dpi=130)
    plt.show()
    return H_vals, kl_vals

H_vals, kl_vals = uncertainty_maps(pixel_ids_amsterdam, proba_zero, y_amsterdam,
                                    'Amsterdam zero-shot')
print("[E.3] Mapas de incerteza guardados em diagnostico_incerteza_amsterdam.png")
print(f"      -> Entropia média: {H_vals.mean():.3f} bits (máx=2 bits, quanto menor mais confiante)")


# --- E.4: Importância de variáveis (feature importance) --------------------------
importances = model_final.feature_importances_
order = np.argsort(importances)[::-1]
TOP_N = 15
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(range(TOP_N), importances[order[:TOP_N]][::-1], color='steelblue')
ax.set_yticks(range(TOP_N))
ax.set_yticklabels([feature_names[i] for i in order[:TOP_N]][::-1], fontsize=8)
ax.set_xlabel('Importância (XGBoost)')
ax.set_title(f'Top {TOP_N} atributos mais importantes', fontweight='bold')
plt.tight_layout()
plt.savefig('diagnostico_importancia_variaveis.png', dpi=130)
plt.show()
print("[E.4] Importância de variáveis guardada em diagnostico_importancia_variaveis.png")
print("      Top 5:")
for i in order[:5]:
    print(f"        {feature_names[i]}: {importances[i]:.4f}")

print("\n=== FIM: modelo validado em Madrid, testado (zero-shot) e adaptado (few-shot) em Amsterdam ===")
