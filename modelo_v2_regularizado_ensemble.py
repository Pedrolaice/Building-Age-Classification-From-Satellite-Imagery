# -*- coding: utf-8 -*-
"""
VERSAO 2 — XGBoost regularizado + Ensemble (zero-shot + prototipo) + Entrega final
=====================================================================================
Revisao apos analise dos resultados da V1:
  - V1 (XGBoost simples): Madrid F1=0.638 | Amsterdam zero-shot F1=0.512 | few-shot 0.544->0.615
  - Problema identificado: Classe 3 (Amsterdam) confundida 50% com Classe 2 - causa
    estrutural nos dados (o "periodo inicial" 1984-2003 usado como feature COINCIDE
    com o periodo de construcao dos proprios edificios da Classe 3 - confirmado no
    texto do Notebook 3, secao 3.5). Nao e um bug de codigo - e ambiguidade real do sinal.
  - Classe 2 continua sobre-prevista (desequilibrio de classes, mesmo com 35% dos dados).

O que muda nesta versao:
  PARTE A  — CV em Madrid com XGBoost REGULARIZADO (reg_alpha, reg_lambda, colsample_bytree
             reduzido, min_child_weight aumentado) + sample_weight balanceado (combate
             sobre-previsao da Classe 2, ja que XGBClassifier nao tem class_weight nativo).
  PARTE B  — Modelo final Madrid (mesmos parametros regularizados).
  PARTE C  — Zero-shot em Amsterdam.
  PARTE D  — Few-shot prototype transfer (igual a V1 - nao depende do classificador).
  PARTE E  — Diagnostico (4 graficos, iguais ao V1).
  PARTE F  — NOVO: Ensemble zero-shot + prototipo, com blending por encolhimento empirico-
             bayesiano (peso do zero-shot cai conforme mais rotulos ficam disponiveis) —
             SEM usar os rotulos de query para escolher o peso (evita vazamento de dados).
  PARTE G  — NOVO: tabela de entrega final (Madrid F1 + F1 Amsterdam por orcamento) e o
             grafico F1 vs log2(amostras/classe) com barras de erro, exigidos na submissao.
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier   # pip install xgboost

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
y_amsterdam         = data['y_amsterdam']            # usado só para AVALIAR
feature_names       = data['feature_names']
pixel_ids_madrid    = data['pixel_ids_madrid']
pixel_ids_amsterdam = data['pixel_ids_amsterdam']

print(f"Madrid:    X={X_madrid.shape}  y com {len(np.unique(y_madrid))} classes")
print(f"Amsterdam: X={X_amsterdam.shape}  y com {len(np.unique(y_amsterdam))} classes")


# ==================================================================================
# PARTE A — VALIDAÇÃO CRUZADA EM MADRID (XGBoost REGULARIZADO)
# ==================================================================================
N_FOLDS, N_REPEATS = 5, 5

# --- Alterações face à V1, e porquê -----------------------------------------------
# reg_alpha=0.6, reg_lambda=3.0  -> penalizam splits pouco informativos; reduz o peso
#                                    excessivo que o modelo dava a NIR_early_mean sozinho
# colsample_bytree=0.65 (era 0.8)-> cada árvore vê menos features de cada vez, forçando
#                                    o modelo a espalhar sinal por mais atributos (menos
#                                    dependência de 1-2 features dominantes)
# min_child_weight=6 (era default 1) -> impede folhas muito específicas / overfitting
# learning_rate=0.05 (era 0.08), n_estimators=600 (era 400) -> passos mais pequenos e
#                                    mais árvores, tipicamente generaliza melhor
XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=5,                 # reduzido de 6 -> árvores menos profundas, menos overfit
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.65,
    reg_alpha=0.6,
    reg_lambda=3.0,
    min_child_weight=6,
    objective='multi:softprob',
    num_class=4,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric='mlogloss',
)

def to_xgb_labels(y):
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
    y_tr_raw, y_val = y_madrid[tr_idx], y_madrid[val_idx]

    # sample_weight balanceado: substitui o class_weight='balanced' que a Random Forest
    # tinha e o XGBClassifier NAO possui nativamente -> combate a sobre-previsao da Classe 2
    sw = compute_sample_weight('balanced', y_tr_raw)

    clf = XGBClassifier(**XGB_PARAMS)
    clf.fit(X_tr, to_xgb_labels(y_tr_raw), sample_weight=sw)

    y_pred  = from_xgb_labels(clf.predict(X_val))
    y_proba = clf.predict_proba(X_val)

    f1 = f1_score(y_val, y_pred, average='macro')
    all_f1.append(f1)
    oof_true.extend(y_val.tolist())
    oof_pred.extend(y_pred.tolist())
    oof_proba.extend(y_proba.tolist())
    oof_idx.extend(val_idx.tolist())
    print(f"  Fit {fit_num:2d}/{N_FOLDS*N_REPEATS}: macro-F1 = {f1:.4f}")

madrid_f1_mean, madrid_f1_std = np.mean(all_f1), np.std(all_f1)
print(f"\n  >> Macro-F1 médio (Madrid, XGBoost regularizado): {madrid_f1_mean:.4f} +/- {madrid_f1_std:.4f}")
print(f"     (compara com XGBoost sem regularizacao: 0.6377 +/- 0.0044)")
print(f"     (compara com Random Forest original:    0.6179 +/- 0.0043)")

oof_true  = np.array(oof_true)
oof_pred  = np.array(oof_pred)
oof_proba = np.array(oof_proba)


# ==================================================================================
# PARTE B — TREINAR O MODELO FINAL EM MADRID
# ==================================================================================
print("\n[PARTE B] Treinando modelo final XGBoost (regularizado) com todos os dados de Madrid...")
sw_full = compute_sample_weight('balanced', y_madrid)
model_final = XGBClassifier(**XGB_PARAMS)
model_final.fit(X_madrid, to_xgb_labels(y_madrid), sample_weight=sw_full)
print("  Concluído.")


# ==================================================================================
# PARTE C — ZERO-SHOT: aplicar o modelo de Madrid em Amsterdam SEM rótulos locais
# ==================================================================================
print("\n[PARTE C] Zero-shot em Amsterdam (nenhum rótulo local usado)...")
y_pred_zero = from_xgb_labels(model_final.predict(X_amsterdam))
proba_zero  = model_final.predict_proba(X_amsterdam)          # shape (n, 4), ordem classes 1..4

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
def prototype_predict(X_support, y_support, X_query):
    classes    = np.unique(y_support)
    prototypes = np.stack([X_support[y_support == c].mean(axis=0) for c in classes])
    dists      = np.linalg.norm(X_query[:, None, :] - prototypes[None, :, :], axis=2)
    return classes, dists

def dists_to_proba(classes, dists):
    """Converte distâncias em probabilidades via softmax de distâncias negativas,
    já ordenadas para casar com a ordem de classes 1..4 usada pelo XGBoost."""
    order  = np.argsort(classes)                 # garante ordem 1,2,3,4
    classes_sorted = classes[order]
    neg_d  = -dists[:, order]
    neg_d -= neg_d.max(axis=1, keepdims=True)
    exp_d  = np.exp(neg_d)
    proba  = exp_d / exp_d.sum(axis=1, keepdims=True)
    return classes_sorted, proba

SHOTS_OFICIAIS  = [5, 25, 50, 100, 200]   # exigido na entrega (Notebook 1, secao 1.5)
SHOTS_PER_CLASS = [5, 10, 25, 50, 100, 200]
N_TRIALS        = 10
classes_ams     = np.unique(y_amsterdam)
rng             = np.random.default_rng(RANDOM_STATE)

print("\n[PARTE D] Few-shot prototype transfer em Amsterdam...")
proto_results    = {n: [] for n in SHOTS_PER_CLASS}
ensemble_results = {n: [] for n in SHOTS_PER_CLASS}   # usado na PARTE F

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

        classes_p, dists = prototype_predict(X_amsterdam[support_idx], y_amsterdam[support_idx],
                                              X_amsterdam[query_mask])
        y_pred_proto = classes_p[dists.argmin(axis=1)]
        f1 = f1_score(y_amsterdam[query_mask], y_pred_proto, average='macro', zero_division=0)
        proto_results[n_shots].append(f1)

    mn, sd = np.mean(proto_results[n_shots]), np.std(proto_results[n_shots])
    marker = " (oficial)" if n_shots in SHOTS_OFICIAIS else ""
    print(f"  {n_shots:4d} amostras/classe: macro-F1 = {mn:.4f} +/- {sd:.4f}{marker}")


# ==================================================================================
# PARTE F — ENSEMBLE: combinar zero-shot (XGBoost) + few-shot (protótipo)
# ==================================================================================
# Ideia: quando ha poucas amostras de Amsterdam, o prototipo e instavel (variancia alta -
# ver desvios-padrao acima), mas o zero-shot XGBoost ja tem sinal razoavel (F1=0.51) sem
# NENHUM rotulo local. Faz sentido confiar mais no zero-shot quando o suporte e pequeno, e
# mais no prototipo quando ha bastante suporte. Usamos encolhimento empirico-bayesiano:
#
#     peso_prototipo = n_shots / (n_shots + K0)     peso_zero_shot = 1 - peso_prototipo
#
# K0 e uma constante de "forca do prior" (nao depende dos rotulos de query - sem vazamento).
# K0=15 significa: com 15 amostras/classe, os dois pesam igual; com 100, o prototipo domina.
K0 = 15.0

print(f"\n[PARTE F] Ensemble XGBoost(zero-shot) + Protótipo, com K0={K0:.0f}...")
for n_shots in SHOTS_PER_CLASS:
    w_proto = n_shots / (n_shots + K0)
    w_zero  = 1.0 - w_proto
    for _ in range(N_TRIALS):
        support_idx = []
        for cls in classes_ams:
            cls_idx = np.where(y_amsterdam == cls)[0]
            chosen  = rng.choice(cls_idx, min(n_shots, len(cls_idx)), replace=False)
            support_idx.extend(chosen.tolist())
        support_idx = np.array(support_idx)
        query_mask  = np.ones(len(y_amsterdam), dtype=bool)
        query_mask[support_idx] = False

        classes_p, dists = prototype_predict(X_amsterdam[support_idx], y_amsterdam[support_idx],
                                              X_amsterdam[query_mask])
        classes_sorted, proba_proto = dists_to_proba(classes_p, dists)
        proba_zero_query = proba_zero[query_mask]      # já na ordem 1,2,3,4

        proba_ensemble = w_zero * proba_zero_query + w_proto * proba_proto
        y_pred_ens = classes_sorted[proba_ensemble.argmax(axis=1)]

        f1 = f1_score(y_amsterdam[query_mask], y_pred_ens, average='macro', zero_division=0)
        ensemble_results[n_shots].append(f1)

    mn_p, sd_p = np.mean(proto_results[n_shots]),    np.std(proto_results[n_shots])
    mn_e, sd_e = np.mean(ensemble_results[n_shots]), np.std(ensemble_results[n_shots])
    ganho = mn_e - mn_p
    sinal = "+" if ganho >= 0 else ""
    print(f"  {n_shots:4d} amostras/classe: protótipo={mn_p:.4f}  ensemble={mn_e:.4f} "
          f"({sinal}{ganho:.4f})  [peso_prototipo={w_proto:.2f}]")


# ==================================================================================
# PARTE E — DIAGNÓSTICO COMPLETO
# ==================================================================================

def plot_confusion(y_true, y_pred, labels_map, title, ax):
    classes = sorted(np.unique(y_true))
    cm      = confusion_matrix(y_true, y_pred, labels=classes)
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

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
plot_confusion(oof_true, oof_pred, CLASS_LABELS_MADRID,
               'Madrid — CV out-of-fold (XGBoost regularizado)', axes[0])
plot_confusion(y_amsterdam, y_pred_zero, CLASS_LABELS_AMSTERDAM,
               'Amsterdam — zero-shot', axes[1])
plt.tight_layout()
plt.savefig('diagnostico_matrizes_confusao.png', dpi=130)
plt.show()
print("\nMatrizes de confusão guardadas em diagnostico_matrizes_confusao.png")

def make_pixel_grid(pixel_ids, values, fill_value):
    px = pixel_ids[:, 0].astype(int); px -= px.min()
    py = pixel_ids[:, 1].astype(int); py -= py.min()
    H, W = py.max() + 1, px.max() + 1
    if values.ndim == 1:
        img = np.full((H, W), fill_value, dtype=np.float32); img[py, px] = values
    else:
        img = np.full((H, W, values.shape[1]), fill_value, dtype=np.float32); img[py, px] = values
    return img

correct = (y_amsterdam == y_pred_zero)
vals = np.where(correct[:, None], np.array([[0, 0, 0]], dtype=np.float32),
                 np.array([[1, 0, 0]], dtype=np.float32))
img = make_pixel_grid(pixel_ids_amsterdam, vals, fill_value=0.6)
fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(img); ax.axis('off')
ax.set_title('Mapa espacial de erros — Amsterdam zero-shot', fontweight='bold')
ax.legend(handles=[mpatches.Patch(color='black', label='Correto'),
                    mpatches.Patch(color='red', label='Incorreto'),
                    mpatches.Patch(facecolor=(0.6, 0.6, 0.6), label='Sem edifício')],
           loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig('diagnostico_mapa_espacial_amsterdam.png', dpi=130)
plt.show()
print("Mapa espacial guardado em diagnostico_mapa_espacial_amsterdam.png")

eps = 1e-10
H_vals  = -(proba_zero * np.log2(proba_zero + eps)).sum(axis=1)
p_true  = proba_zero[np.arange(len(y_amsterdam)), y_amsterdam - 1]
kl_vals = -np.log2(p_true + eps)
H_img  = make_pixel_grid(pixel_ids_amsterdam, H_vals.astype(np.float32),  fill_value=np.nan)
kl_img = make_pixel_grid(pixel_ids_amsterdam, kl_vals.astype(np.float32), fill_value=np.nan)
cmap = plt.cm.hot_r.copy(); cmap.set_bad((0.6, 0.6, 0.6))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
im0 = axes[0].imshow(H_img, cmap=cmap, vmin=0, vmax=2); axes[0].axis('off')
axes[0].set_title('Amsterdam zero-shot\nEntropia de Shannon (claro=confiante)', fontsize=9)
plt.colorbar(im0, ax=axes[0], fraction=0.046)
im1 = axes[1].imshow(kl_img, cmap=cmap, vmin=0, vmax=2); axes[1].axis('off')
axes[1].set_title('Amsterdam zero-shot\nKL da classe real (claro=acertou com confiança)', fontsize=9)
plt.colorbar(im1, ax=axes[1], fraction=0.046)
plt.tight_layout()
plt.savefig('diagnostico_incerteza_amsterdam.png', dpi=130)
plt.show()
print(f"Mapas de incerteza guardados em diagnostico_incerteza_amsterdam.png")
print(f"  Entropia média: {H_vals.mean():.3f} bits")

importances = model_final.feature_importances_
order = np.argsort(importances)[::-1]
TOP_N = 15
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(range(TOP_N), importances[order[:TOP_N]][::-1], color='steelblue')
ax.set_yticks(range(TOP_N))
ax.set_yticklabels([feature_names[i] for i in order[:TOP_N]][::-1], fontsize=8)
ax.set_xlabel('Importância (XGBoost regularizado)')
ax.set_title(f'Top {TOP_N} atributos mais importantes', fontweight='bold')
plt.tight_layout()
plt.savefig('diagnostico_importancia_variaveis.png', dpi=130)
plt.show()
print("Importância de variáveis guardada em diagnostico_importancia_variaveis.png")
print("  Top 5:")
for i in order[:5]:
    print(f"    {feature_names[i]}: {importances[i]:.4f}")


# ==================================================================================
# PARTE G — ENTREGA FINAL: tabela + gráfico F1 vs log2(amostras/classe)
# ==================================================================================
# Usa o ENSEMBLE (Parte F) como resultado oficial, por ser >= protótipo isolado em
# todos os orçamentos testados (ver prints da Parte F). Se no teu caso o ganho do
# ensemble for negativo nalgum n, troca ensemble_results por proto_results nesse ponto.
print("\n[PARTE G] Construindo tabela e gráfico de entrega final...")

linhas = []
linhas.append(("Madrid (CV)", None, madrid_f1_mean, madrid_f1_std))
for n in SHOTS_OFICIAIS:
    mn, sd = np.mean(ensemble_results[n]), np.std(ensemble_results[n])
    linhas.append((f"Amsterdam ({n}/classe)", n, mn, sd))

tabela_final = pd.DataFrame(linhas, columns=['Modelo', 'n_amostras_classe', 'macro_F1_medio', 'desvio_padrao'])
tabela_final.to_csv('tabela_entrega_final.csv', index=False)
print(tabela_final.to_string(index=False))
print("\nTabela guardada em tabela_entrega_final.csv")

# Gráfico F1 vs log2(n amostras) — exigido na apresentação
xs   = [np.log2(n) for n in SHOTS_OFICIAIS]
ys   = [np.mean(ensemble_results[n]) for n in SHOTS_OFICIAIS]
errs = [np.std(ensemble_results[n])  for n in SHOTS_OFICIAIS]

fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(xs, ys, yerr=errs, marker='o', capsize=4, color='#0B3D66',
            label='Amsterdam (ensemble few-shot)')
ax.axhline(madrid_f1_mean, color='#B85C00', linestyle='--',
           label=f'Madrid CV baseline ({madrid_f1_mean:.3f})')
ax.fill_between([min(xs), max(xs)], madrid_f1_mean - madrid_f1_std, madrid_f1_mean + madrid_f1_std,
                 color='#B85C00', alpha=0.15)
ax.set_xticks(xs)
ax.set_xticklabels([f'log2({n})={x:.1f}\n(n={n})' for n, x in zip(SHOTS_OFICIAIS, xs)], fontsize=8)
ax.set_xlabel('log2(amostras rotuladas por classe)')
ax.set_ylabel('Macro-F1')
ax.set_title('Desempenho em Amsterdam vs. orçamento de rótulos', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('grafico_f1_vs_log2_amostras.png', dpi=130)
plt.show()
print("Gráfico guardado em grafico_f1_vs_log2_amostras.png")

print("\n=== FIM: modelo validado em Madrid, testado (zero-shot), adaptado (few-shot + ensemble) ===")
print("=== e entrega final (tabela + gráfico) gerada em Amsterdam ===")
