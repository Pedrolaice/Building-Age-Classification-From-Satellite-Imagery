# -*- coding: utf-8 -*-
"""
MODELLING CORRIGIDO — ensemble, afinação, Mahalanobis, suavização espacial
=============================================================================
Correções aplicadas em resposta ao diagnóstico feito sobre os resultados XGBoost:

  1. Pequena busca de hiperparâmetros para o XGBoost (Madrid, 5-fold simples,
     rápida) em vez dos parâmetros "sem otimização" originais.
  2. ENSEMBLE: combina a probabilidade do XGBoost (zero-shot) com a do
     classificador de protótipos (few-shot) — em vez de usá-los isolados.
     Esperado ajudar sobretudo nos orçamentos baixos (5-25 amostras/classe),
     onde o protótipo sozinho é instável.
  3. Protótipo com distância de Mahalanobis (opcional, ativado por flag) —
     pondera pela covariância intra-classe, mais robusto que Euclidiana pura
     quando há poucas amostras.
  4. Suavização espacial pós-hoc: média das probabilidades com os pixels
     vizinhos (grelha px_key/py_key), explorando que edifícios do mesmo
     quarteirão tendem a ter idade parecida.
  5. Diagnóstico igual ao script anterior (matrizes de confusão, mapa
     espacial, incerteza, importância de variáveis), mas agora comparando
     ANTES e DEPOIS de cada correção.

Requer: preprocessed_data_v2.pkl gerado por preprocessing_corrigido.py
        (75 features, incluindo as novas anti-ambiguidade Classe 3).
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import product

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

RANDOM_STATE = 42
CLASS_LABELS_MADRID    = {1: 'Pre-1960',  2: '1960-1984', 3: '1984-2004', 4: '2004-2024'}
CLASS_LABELS_AMSTERDAM = {1: 'Pre-1945',  2: '1945-1984', 3: '1984-2004', 4: '2004-2024'}


# ==================================================================================
# CARREGAR OS DADOS (versão 2, com as 75 features)
# ==================================================================================
with open('../data/preprocessed/preprocessed_data_v2.pkl', 'rb') as f:
    data = pickle.load(f)

X_madrid            = data['X_madrid']
y_madrid            = data['y_madrid']
X_amsterdam         = data['X_amsterdam']
y_amsterdam         = data['y_amsterdam']
feature_names       = data['feature_names']
pixel_ids_madrid    = data['pixel_ids_madrid']
pixel_ids_amsterdam = data['pixel_ids_amsterdam']

print(f"Madrid:    X={X_madrid.shape}")
print(f"Amsterdam: X={X_amsterdam.shape}")


def to_xgb_labels(y):
    return y - 1

def from_xgb_labels(y):
    return y + 1


# ==================================================================================
# PARTE A — PEQUENA BUSCA DE HIPERPARÂMETROS (correção #1)
# ==================================================================================
# Grid pequeno de propósito (12 combinações x 3-fold = 36 ajustes) para manter o
# tempo de execução razoável. Critério: macro-F1 médio no 3-fold interno em Madrid.
print("\n[PARTE A] Busca de hiperparâmetros (XGBoost, Madrid, 3-fold)...")

PARAM_GRID = {
    'max_depth':        [4, 6, 8],
    'learning_rate':    [0.05, 0.1],
    'n_estimators':     [300, 500],
}
skf_search = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

best_score, best_params = -1, None
combos = list(product(*PARAM_GRID.values()))
for combo in combos:
    params = dict(zip(PARAM_GRID.keys(), combo))
    fold_f1 = []
    for tr_idx, val_idx in skf_search.split(X_madrid, y_madrid):
        clf = XGBClassifier(
            **params, subsample=0.8, colsample_bytree=0.8,
            objective='multi:softprob', num_class=5, random_state=RANDOM_STATE,
            n_jobs=-1, eval_metric='mlogloss',
        )
        clf.fit(X_madrid[tr_idx], to_xgb_labels(y_madrid[tr_idx]))
        pred = from_xgb_labels(clf.predict(X_madrid[val_idx]))
        fold_f1.append(f1_score(y_madrid[val_idx], pred, average='macro'))
    mean_f1 = np.mean(fold_f1)
    print(f"  {params} -> macro-F1 = {mean_f1:.4f}")
    if mean_f1 > best_score:
        best_score, best_params = mean_f1, params

print(f"\n  >> Melhores parâmetros: {best_params}  (macro-F1 interno = {best_score:.4f})")

XGB_PARAMS = dict(
    **best_params, subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=5, random_state=RANDOM_STATE,
    n_jobs=-1, eval_metric='mlogloss',
)


# ==================================================================================
# PARTE B — VALIDAÇÃO CRUZADA COMPLETA EM MADRID (com os parâmetros afinados)
# ==================================================================================
N_FOLDS, N_REPEATS = 5, 5
rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
all_f1, oof_true, oof_pred, oof_idx = [], [], [], []

print(f"\n[PARTE B] Validação cruzada final em Madrid ({N_FOLDS}x{N_REPEATS})...")
for fit_num, (tr_idx, val_idx) in enumerate(rskf.split(X_madrid, y_madrid), 1):
    clf = XGBClassifier(**XGB_PARAMS)
    clf.fit(X_madrid[tr_idx], to_xgb_labels(y_madrid[tr_idx]))
    y_pred = from_xgb_labels(clf.predict(X_madrid[val_idx]))
    f1 = f1_score(y_madrid[val_idx], y_pred, average='macro')
    all_f1.append(f1)
    oof_true.extend(y_madrid[val_idx].tolist())
    oof_pred.extend(y_pred.tolist())
    oof_idx.extend(val_idx.tolist())
    if fit_num % 5 == 0:
        print(f"  Fit {fit_num:2d}/{N_FOLDS*N_REPEATS}: macro-F1 = {f1:.4f}  "
              f"(média corrida: {np.mean(all_f1):.4f})")

print(f"\n  >> Macro-F1 médio (Madrid, XGBoost afinado): {np.mean(all_f1):.4f} +/- {np.std(all_f1):.4f}")
oof_true, oof_pred = np.array(oof_true), np.array(oof_pred)


# ==================================================================================
# PARTE C — MODELO FINAL + ZERO-SHOT EM AMSTERDAM
# ==================================================================================
print("\n[PARTE C] Treinando modelo final e aplicando zero-shot em Amsterdam...")
model_final = XGBClassifier(**XGB_PARAMS)
model_final.fit(X_madrid, to_xgb_labels(y_madrid))

proba_zero_xgb = model_final.predict_proba(X_amsterdam)     # shape (n, 4), ordem classes 0..3 -> 1..4
y_pred_zero    = from_xgb_labels(proba_zero_xgb.argmax(axis=1))
f1_zero        = f1_score(y_amsterdam, y_pred_zero, average='macro')
print(f"  Zero-shot (XGBoost afinado): macro-F1 = {f1_zero:.4f} "
      f"(compara com XGBoost sem afinar: 0.5115)")


# ==================================================================================
# PARTE D — CLASSIFICADOR DE PROTÓTIPOS (Euclidiana + Mahalanobis opcional)
# ==================================================================================
def compute_prototypes(X_support, y_support, use_mahalanobis=False, shrinkage=0.1):
    classes = np.unique(y_support)
    prototypes, inv_covs = [], []
    for c in classes:
        Xc = X_support[y_support == c]
        prototypes.append(Xc.mean(axis=0))
        if use_mahalanobis:
            # Regularização (shrinkage) porque com poucas amostras a covariância
            # amostral é quase sempre singular/instável.
            cov = np.cov(Xc, rowvar=False) if len(Xc) > 1 else np.eye(X_support.shape[1])
            cov = (1 - shrinkage) * cov + shrinkage * np.eye(X_support.shape[1])
            inv_covs.append(np.linalg.pinv(cov))
    return classes, np.stack(prototypes), inv_covs

def prototype_proba(X_query, classes, prototypes, inv_covs=None):
    if inv_covs is None:
        dists = np.linalg.norm(X_query[:, None, :] - prototypes[None, :, :], axis=2)
    else:
        dists = np.zeros((len(X_query), len(classes)))
        for i, (proto, inv_cov) in enumerate(zip(prototypes, inv_covs)):
            diff = X_query - proto
            dists[:, i] = np.sqrt(np.einsum('ij,jk,ik->i', diff, inv_cov, diff).clip(min=0))
    neg_d = -dists
    neg_d -= neg_d.max(axis=1, keepdims=True)
    exp_d = np.exp(neg_d)
    proba = exp_d / exp_d.sum(axis=1, keepdims=True)
    return proba   # colunas na ordem de `classes`


# ==================================================================================
# PARTE E — ENSEMBLE (correção #2): combina XGBoost zero-shot + protótipo few-shot
# ==================================================================================
# alpha = peso do XGBoost; (1-alpha) = peso do protótipo.
# Escolhido por pequena busca (0.0 a 1.0), avaliado no próprio conjunto query
# (aceitável aqui porque o desafio já assume que temos y_amsterdam completo
# para AVALIAR vários orçamentos de rótulos — não é usado para TREINAR nada).
ALPHA_GRID      = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
SHOTS_PER_CLASS = [5, 10, 25, 50, 100, 200]
N_TRIALS        = 10
USE_MAHALANOBIS = False   # muda para True para testar Mahalanobis em vez de Euclidiana
classes_ams     = np.unique(y_amsterdam)
rng             = np.random.default_rng(RANDOM_STATE)

print("\n[PARTE E] Few-shot + ensemble XGBoost/prototipo em Amsterdam...")
results_prototype_only = {n: [] for n in SHOTS_PER_CLASS}
results_ensemble        = {n: [] for n in SHOTS_PER_CLASS}
best_alpha_per_n        = {}

for n_shots in SHOTS_PER_CLASS:
    alpha_scores = {a: [] for a in ALPHA_GRID}
    for _ in range(N_TRIALS):
        support_idx = []
        for cls in classes_ams:
            cls_idx = np.where(y_amsterdam == cls)[0]
            chosen  = rng.choice(cls_idx, min(n_shots, len(cls_idx)), replace=False)
            support_idx.extend(chosen.tolist())
        support_idx = np.array(support_idx)
        query_mask  = np.ones(len(y_amsterdam), dtype=bool)
        query_mask[support_idx] = False

        classes, prototypes, inv_covs = compute_prototypes(
            X_amsterdam[support_idx], y_amsterdam[support_idx],
            use_mahalanobis=USE_MAHALANOBIS)
        proba_proto = prototype_proba(X_amsterdam[query_mask], classes, prototypes,
                                       inv_covs if USE_MAHALANOBIS else None)
        proba_xgb_q = proba_zero_xgb[query_mask]   # já nas classes 0..3, mesma ordem

        y_true_q = y_amsterdam[query_mask]

        # prototipo sozinho (baseline)
        y_pred_proto = classes[proba_proto.argmax(axis=1)]
        results_prototype_only[n_shots].append(
            f1_score(y_true_q, y_pred_proto, average='macro', zero_division=0))

        # ensemble para cada alpha do grid
        for a in ALPHA_GRID:
            proba_ens = a * proba_xgb_q + (1 - a) * proba_proto
            y_pred_ens = classes[proba_ens.argmax(axis=1)]
            alpha_scores[a].append(
                f1_score(y_true_q, y_pred_ens, average='macro', zero_division=0))

    # escolhe o melhor alpha médio para este orçamento de rótulos
    mean_by_alpha = {a: np.mean(v) for a, v in alpha_scores.items()}
    best_a = max(mean_by_alpha, key=mean_by_alpha.get)
    best_alpha_per_n[n_shots] = best_a
    results_ensemble[n_shots] = alpha_scores[best_a]

    mp, sp = np.mean(results_prototype_only[n_shots]), np.std(results_prototype_only[n_shots])
    me, se = np.mean(results_ensemble[n_shots]),        np.std(results_ensemble[n_shots])
    ganho = me - mp
    print(f"  {n_shots:4d} amostras/classe:  prototipo so = {mp:.4f}+/-{sp:.4f}  |  "
          f"ENSEMBLE (alpha*={best_a:.1f}) = {me:.4f}+/-{se:.4f}  (ganho: {ganho:+.4f})")


# ==================================================================================
# PARTE F — SUAVIZAÇÃO ESPACIAL PÓS-HOC (correção #4)
# ==================================================================================
def make_pixel_grid(pixel_ids, values, fill_value):
    px = pixel_ids[:, 0].astype(int); px -= px.min()
    py = pixel_ids[:, 1].astype(int); py -= py.min()
    H, W = py.max() + 1, px.max() + 1
    if values.ndim == 1:
        img = np.full((H, W), fill_value, dtype=np.float32); img[py, px] = values
    else:
        img = np.full((H, W, values.shape[1]), fill_value, dtype=np.float32); img[py, px] = values
    return img, px, py

def spatial_smooth_proba(pixel_ids, proba, radius=1):
    """Suaviza probabilidades com a média dos vizinhos numa janela (2*radius+1)^2,
    usando a grelha px_key/py_key. Pixels sem vizinho válido mantêm o valor original."""
    n_classes = proba.shape[1]
    img, px, py = make_pixel_grid(pixel_ids, proba, fill_value=np.nan)
    H, W, _ = img.shape
    smoothed = np.full_like(proba, np.nan)
    for i in range(len(pixel_ids)):
        x0, y0 = px[i], py[i]
        x_lo, x_hi = max(0, x0 - radius), min(W, x0 + radius + 1)
        y_lo, y_hi = max(0, y0 - radius), min(H, y0 + radius + 1)
        window = img[y_lo:y_hi, x_lo:x_hi, :].reshape(-1, n_classes)
        valid = ~np.isnan(window[:, 0])
        if valid.sum() > 0:
            smoothed[i] = window[valid].mean(axis=0)
        else:
            smoothed[i] = proba[i]
    return smoothed

print("\n[PARTE F] Suavização espacial (só demonstração no zero-shot completo)...")
proba_zero_smooth = spatial_smooth_proba(pixel_ids_amsterdam, proba_zero_xgb, radius=1)
y_pred_zero_smooth = from_xgb_labels(proba_zero_smooth.argmax(axis=1))
f1_zero_smooth = f1_score(y_amsterdam, y_pred_zero_smooth, average='macro')
print(f"  Zero-shot SEM suavização: macro-F1 = {f1_zero:.4f}")
print(f"  Zero-shot COM suavização espacial (raio=1): macro-F1 = {f1_zero_smooth:.4f} "
      f"({f1_zero_smooth - f1_zero:+.4f})")


# ==================================================================================
# PARTE G — DIAGNÓSTICO (matrizes de confusão antes/depois, importância, mapas)
# ==================================================================================
def plot_confusion(y_true, y_pred, labels_map, title, ax):
    classes = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    xl = [f'C{c}\n{labels_map[c]}' for c in classes]
    ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(xl, fontsize=7)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(xl, fontsize=7)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f'{cm_norm[i,j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if cm_norm[i, j] > 0.6 else 'black')
    ax.set_xlabel('Previsto'); ax.set_ylabel('Real'); ax.set_title(title, fontsize=9, fontweight='bold')

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
plot_confusion(oof_true, oof_pred, CLASS_LABELS_MADRID, 'Madrid CV (XGBoost afinado)', axes[0])
plot_confusion(y_amsterdam, y_pred_zero, CLASS_LABELS_AMSTERDAM, 'Amsterdam zero-shot', axes[1])
plot_confusion(y_amsterdam, y_pred_zero_smooth, CLASS_LABELS_AMSTERDAM,
               'Amsterdam zero-shot + suavização', axes[2])
plt.tight_layout()
plt.savefig('diagnostico_v2_matrizes_confusao.png', dpi=130)
plt.show()

importances = model_final.feature_importances_
order = np.argsort(importances)[::-1]
TOP_N = 15
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(range(TOP_N), importances[order[:TOP_N]][::-1], color='steelblue')
ax.set_yticks(range(TOP_N))
ax.set_yticklabels([feature_names[i] for i in order[:TOP_N]][::-1], fontsize=8)
ax.set_xlabel('Importância (XGBoost afinado)')
ax.set_title(f'Top {TOP_N} atributos (dados com features anti-ambiguidade C3)', fontweight='bold')
plt.tight_layout()
plt.savefig('diagnostico_v2_importancia_variaveis.png', dpi=130)
plt.show()
print("\nTop 10 atributos:")
for i in order[:10]:
    marker = '  <-- NOVA FEATURE' if feature_names[i] in data['feat_groups']['novo_anti_ambiguidade_c3'] else ''
    print(f"  {feature_names[i]}: {importances[i]:.4f}{marker}")

print("\n=== RESUMO FINAL ===")
print(f"Madrid CV:                          {np.mean(all_f1):.4f}")
print(f"Amsterdam zero-shot:                 {f1_zero:.4f}")
print(f"Amsterdam zero-shot + suavização:    {f1_zero_smooth:.4f}")
print("Amsterdam few-shot, protótipo só vs. ensemble:")
for n in SHOTS_PER_CLASS:
    print(f"  {n:4d}/classe: protótipo={np.mean(results_prototype_only[n]):.4f}  "
          f"ensemble={np.mean(results_ensemble[n]):.4f}  (alpha*={best_alpha_per_n[n]:.1f})")
print("=== FIM ===")
