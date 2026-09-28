# -*- coding: utf-8 -*-
"""
PREPROCESSING CORRIGIDO — resolve a ambiguidade estrutural da Classe 3
=========================================================================
DIAGNÓSTICO QUE MOTIVA ESTA CORREÇÃO
-------------------------------------------------------------------------
Nas matrizes de confusão (XGBoost), a Classe 3 (construída 1984-2004) é a mais
mal servida: em Amsterdam zero-shot, 50% dos seus pixels são confundidos com
a Classe 2. A causa raiz está na definição das features '<banda>_early_mean' /
'<banda>_early_std' (janela 1984-2003):

  - Para as Classes 1 e 2 (edifícios já existentes antes de 1984), a janela
    early mostra um edifício ESTÁVEL e maduro -> sinal limpo.
  - Para a Classe 3 (construída algures entre 1984 e 2004), o evento de
    construção cai DENTRO da própria janela early -> a média/desvio-padrão
    de 1984-2003 mistura "terreno antigo" com "obra em curso", ficando
    ambígua e parecida com qualquer outra classe.

  Precisamente por isso 'NIR_early_mean' é a feature #1 em importância no
  XGBoost (peso 0.067) — o modelo depende muito de uma feature que é
  sistematicamente ambígua para uma das 4 classes que precisa de separar.

CORREÇÃO APLICADA (2 grupos de features novas, +18 no total)
-------------------------------------------------------------------------
1. SPLIT DA JANELA EARLY EM DUAS METADES (early1: 1984-1993, early2: 1994-2003)
   + feature de "salto" (early2_mean - early1_mean) por banda.
   -> Um pixel realmente estável (Classes 1/2) não deve ter uma diferença
      grande entre as duas metades. Um pixel cuja construção caiu dentro do
      próprio período early (parte da Classe 3) deve mostrar um salto maior.
   Custo: +6 features (early_shift por banda).

2. MAIOR SALTO ANO-A-ANO ABSOLUTO (max |Δ| por banda, em toda a série)
   -> O desvio-padrão do YoY (já existente) DILUI um único evento abrupto
      (construção) ao longo de 40 anos de ruído normal. O MÁXIMO absoluto,
      em vez da média/desvio, captura melhor um evento único e pontual,
      que é exatamente a assinatura de uma obra de construção.
   Custo: +6 features (d_<banda>_max_abs).

3. ANO DO MAIOR SALTO, NORMALIZADO (posição temporal relativa do pico de
   mudança, 0=início da série, 1=fim) -> ajuda a distinguir se o "salto"
   aconteceu perto de 1984-2004 (típico de Classe 3) ou noutra altura.
   Custo: +6 features (d_<banda>_peak_year_norm) -- na prática calculamos só
   para as 3 bandas mais importantes (NIR, Red, Green) para não inflar
   demasiado a dimensionalidade: +3 features.

Total: 60 (original) + 6 + 6 + 3 = 75 features.

Resto do pipeline (carregar, assign_age_class, flatten, gap-fill, índices
espectrais, padronização) é EXATAMENTE o mesmo do Notebook 3 original —
só a função build_pixel_features() e a lista FEATURE_COLS mudam.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ── Constantes (iguais ao notebook original) ──────────────────────────────
BASE_BANDS = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
INDEX_COLS = ['NDVI', 'NDBI', 'UI', 'MNDWI', 'BSI']
BLUE_MAX   = 15_000

CLASS_EVENTS = {'Amsterdam': 1945, 'Madrid': 1960}
CLASS_LABELS = {
    'Amsterdam': {1: 'Pre-1945', 2: '1945-1984', 3: '1984-2004', 4: '2004-2024'},
    'Madrid':    {1: 'Pre-1960', 2: '1960-1984', 3: '1984-2004', 4: '2004-2024'},
}

EARLY_CUTOFF = 2003
LATE_START   = 2004
EARLY1_END   = 1993          # NOVO: primeira metade do período early (1984-1993)
EARLY2_START = 1994          # NOVO: segunda metade do período early (1994-2003)
PID = ['px_key', 'py_key']

PEAK_BANDS = ['NIR', 'Red', 'Green']   # NOVO: bandas usadas para a feature de "ano do pico"


# ===========================================================================
# 3.1 a 3.5 — EXATAMENTE IGUAIS AO NOTEBOOK 3 ORIGINAL (sem alterações)
# ===========================================================================
def assign_age_class(data, events=CLASS_EVENTS):
    out = data.copy()
    out['age_class'] = pd.NA
    for city, ev in events.items():
        mask = out['city'] == city
        if mask.any():
            bins = [-np.inf, ev, 1984, 2004, np.inf]
            out.loc[mask, 'age_class'] = pd.cut(
                out.loc[mask, 'weighted_mean_year'], bins=bins, labels=[1, 2, 3, 4]
            ).astype('Int64')
    return out


def get_best_obs_band(data, band):
    col = np.full(len(data), np.nan)
    for slot in [1, 2, 3]:
        bc, qc = f'{band}_{slot}', f'qa_valid_{slot}'
        if bc not in data.columns or qc not in data.columns:
            continue
        valid = data[qc].fillna(False).values
        col = np.where(np.isnan(col) & valid, data[bc].fillna(np.nan).values, col)
    return col


def make_flat_view(data):
    keep = [c for c in ['pixel_id', 'city', 'year', 'px_key', 'py_key',
                         'weighted_mean_year', 'age_class'] if c in data.columns]
    out = data[keep].copy()
    for band in BASE_BANDS:
        out[band] = get_best_obs_band(data, band)
    out = out.dropna(subset=BASE_BANDS)
    out = out[out['Blue'] <= BLUE_MAX].copy()
    return out


def period_coverage(flat):
    pixels    = flat[PID].drop_duplicates()
    has_early = flat[flat['year'] <= EARLY_CUTOFF].drop_duplicates(PID).assign(has_early_data=1.0)
    has_late  = flat[flat['year'] >= LATE_START].drop_duplicates(PID).assign(has_late_data=1.0)
    return (pixels
            .merge(has_early[PID + ['has_early_data']], on=PID, how='left')
            .merge(has_late[PID  + ['has_late_data']],  on=PID, how='left')
            .fillna(0.0))


def fill_time_series_gaps(flat):
    year_min, year_max = flat['year'].min(), flat['year'].max()
    all_years = list(range(year_min, year_max + 1))
    pixels = flat[PID + ['city', 'weighted_mean_year', 'age_class']].drop_duplicates(PID)
    grid = pixels.assign(_k=1).merge(pd.DataFrame({'year': all_years, '_k': 1}), on='_k').drop('_k', axis=1)
    merged = grid.merge(flat[PID + ['year'] + BASE_BANDS], on=PID + ['year'], how='left')
    merged = merged.sort_values(PID + ['year'])

    def interp(s):
        return s.interpolate(method='linear').bfill().ffill()

    for band in BASE_BANDS:
        merged[band] = merged.groupby(PID)[band].transform(interp)
    return merged


def compute_spectral_indices(df):
    df = df.copy()
    B, G, R = df['Blue'], df['Green'], df['Red']
    N, S1, S2 = df['NIR'], df['SWIR1'], df['SWIR2']
    eps = 1e-6
    df['NDVI']  = (N - R)  / (N + R  + eps)
    df['NDBI']  = (S1 - N) / (S1 + N + eps)
    df['UI']    = (S2 - N) / (S2 + N + eps)
    df['MNDWI'] = (G - S1) / (G + S1 + eps)
    df['BSI']   = ((S1 + R) - (N + B)) / ((S1 + R) + (N + B) + eps)
    return df


# ===========================================================================
# 3.6 — FEATURE ENGINEERING (VERSÃO CORRIGIDA)
# ===========================================================================
def build_pixel_features_v2(flat):
    s = flat[flat['age_class'].notna()].copy()
    s = s.sort_values(PID + ['year'])

    # ---- (Original) Overall stats: bandas + índices --------------------------
    grp = s.groupby(PID + ['age_class'])
    overall = pd.concat([
        grp[BASE_BANDS + INDEX_COLS].mean().add_suffix('_mean'),
        grp[BASE_BANDS + INDEX_COLS].std().add_suffix('_std'),
    ], axis=1).reset_index()
    overall['age_class'] = overall['age_class'].astype(int)

    # ---- (Original) Early / Late period stats --------------------------------
    early_grp = s[s['year'] <= EARLY_CUTOFF].groupby(PID)
    early = pd.concat([
        early_grp[BASE_BANDS].mean().add_suffix('_early_mean'),
        early_grp[BASE_BANDS].std().add_suffix('_early_std'),
    ], axis=1).reset_index()

    late_grp = s[s['year'] >= LATE_START].groupby(PID)
    late = pd.concat([
        late_grp[BASE_BANDS].mean().add_suffix('_late_mean'),
        late_grp[BASE_BANDS].std().add_suffix('_late_std'),
    ], axis=1).reset_index()

    # ---- (Original) Year-on-year mean/std -------------------------------------
    for b in BASE_BANDS:
        s[f'd_{b}'] = s.groupby(PID)[b].diff()
    dcols = [f'd_{b}' for b in BASE_BANDS]
    diff_grp = s.groupby(PID)[dcols]
    diff = pd.concat([
        diff_grp.mean().add_suffix('_mean'),
        diff_grp.std().add_suffix('_std'),
    ], axis=1).reset_index()

    yr = s.groupby(PID)['weighted_mean_year'].first().reset_index()

    # ---- NOVO 1: Split da janela early em duas metades + "salto" --------------
    e1_grp = s[(s['year'] <= EARLY1_END)].groupby(PID)
    e2_grp = s[(s['year'] >= EARLY2_START) & (s['year'] <= EARLY_CUTOFF)].groupby(PID)
    e1_mean = e1_grp[BASE_BANDS].mean().add_suffix('_e1mean')
    e2_mean = e2_grp[BASE_BANDS].mean().add_suffix('_e2mean')
    early_split = e1_mean.join(e2_mean, how='outer').reset_index()
    for b in BASE_BANDS:
        c1, c2 = f'{b}_e1mean', f'{b}_e2mean'
        early_split[f'{b}_early_shift'] = early_split[c2] - early_split[c1]
    early_shift_cols = [f'{b}_early_shift' for b in BASE_BANDS]
    early_split = early_split[PID + early_shift_cols].fillna(0.0)

    # ---- NOVO 2: Maior salto ano-a-ano absoluto (em toda a série) -------------
    max_abs_grp = s.groupby(PID)[dcols].apply(lambda g: g.abs().max())
    max_abs_grp.columns = [f'{c}_max_abs' for c in max_abs_grp.columns]
    max_abs_grp = max_abs_grp.reset_index()

    # ---- NOVO 3: Ano (normalizado) do maior salto, para as bandas mais úteis --
    def peak_year_norm(g):
        out = {}
        yrs = g['year'].values
        span = (yrs.max() - yrs.min()) if yrs.max() > yrs.min() else 1
        for b in PEAK_BANDS:
            col = f'd_{b}'
            if g[col].notna().any():
                idx = g[col].abs().idxmax()
                out[f'{b}_peak_year_norm'] = (g.loc[idx, 'year'] - yrs.min()) / span
            else:
                out[f'{b}_peak_year_norm'] = np.nan
        return pd.Series(out)

    peak_year = s.groupby(PID).apply(peak_year_norm).reset_index()
    peak_cols = [f'{b}_peak_year_norm' for b in PEAK_BANDS]
    peak_year[peak_cols] = peak_year[peak_cols].fillna(0.5)   # sem sinal -> assume meio da série

    # ---- Merge de tudo ----------------------------------------------------------
    result = overall
    for extra in [early, late, diff, yr, early_split, max_abs_grp, peak_year]:
        result = result.merge(extra, on=PID, how='left')
    return result, early_shift_cols, [f'{c}_max_abs' for c in dcols], peak_cols


# ===========================================================================
# PIPELINE PRINCIPAL
# ===========================================================================
def run_preprocessing():
    DATA_DIR = '../data'
    df_madrid    = pd.read_parquet(f'{DATA_DIR}/madrid_train.parquet')
    df_amsterdam = pd.read_parquet(f'{DATA_DIR}/amsterdam_data.parquet')

    df_madrid    = assign_age_class(df_madrid)
    df_amsterdam = assign_age_class(df_amsterdam)

    print('Achatando observações...')
    flat_madrid    = make_flat_view(df_madrid)
    flat_amsterdam = make_flat_view(df_amsterdam)

    indicators_madrid    = period_coverage(flat_madrid)
    indicators_amsterdam = period_coverage(flat_amsterdam)

    print('Preenchendo lacunas temporais...')
    flat_madrid    = fill_time_series_gaps(flat_madrid)
    flat_amsterdam = fill_time_series_gaps(flat_amsterdam)

    flat_madrid    = compute_spectral_indices(flat_madrid)
    flat_amsterdam = compute_spectral_indices(flat_amsterdam)

    print('Construindo features por pixel (versão corrigida, com anti-ambiguidade Classe 3)...')
    features_madrid, shift_cols, maxabs_cols, peak_cols = build_pixel_features_v2(flat_madrid)
    features_amsterdam, _, _, _                          = build_pixel_features_v2(flat_amsterdam)

    features_madrid    = features_madrid.merge(indicators_madrid,    on=PID, how='left')
    features_amsterdam = features_amsterdam.merge(indicators_amsterdam, on=PID, how='left')

    # ── Listas de colunas de features (originais + novas) ──────────────────
    FEAT_OVERALL = [f'{b}_mean' for b in BASE_BANDS] + [f'{b}_std' for b in BASE_BANDS]
    FEAT_INDICES = [f'{i}_mean' for i in INDEX_COLS] + [f'{i}_std' for i in INDEX_COLS]
    FEAT_EARLY   = [f'{b}_early_mean' for b in BASE_BANDS] + [f'{b}_early_std' for b in BASE_BANDS]
    FEAT_LATE    = [f'{b}_late_mean' for b in BASE_BANDS] + [f'{b}_late_std' for b in BASE_BANDS]
    FEAT_YOY     = [f'd_{b}_mean' for b in BASE_BANDS] + [f'd_{b}_std' for b in BASE_BANDS]
    FEAT_IND     = ['has_early_data', 'has_late_data']
    FEAT_NOVO    = shift_cols + maxabs_cols + peak_cols   # 6 + 6 + 3 = 15 novas features

    FEATURE_COLS = (FEAT_OVERALL + FEAT_INDICES + FEAT_EARLY + FEAT_LATE +
                     FEAT_YOY + FEAT_IND + FEAT_NOVO)

    print(f'\nTotal de features: {len(FEATURE_COLS)} '
          f'(60 originais + {len(FEAT_NOVO)} novas anti-ambiguidade Classe 3)')

    for feat in [features_madrid, features_amsterdam]:
        feat[FEATURE_COLS] = feat[FEATURE_COLS].fillna(0.0)

    X_madrid    = features_madrid[FEATURE_COLS].values.astype(np.float64)
    y_madrid    = features_madrid['age_class'].values.astype(int)
    X_amsterdam = features_amsterdam[FEATURE_COLS].values.astype(np.float64)
    y_amsterdam = features_amsterdam['age_class'].values.astype(int)

    scaler = StandardScaler()
    X_madrid    = scaler.fit_transform(X_madrid)
    X_amsterdam = scaler.transform(X_amsterdam)
    print('Padronizado (scaler ajustado apenas em Madrid).')

    output_dir = Path('../data/preprocessed')
    output_dir.mkdir(exist_ok=True, parents=True)
    data = {
        'X_madrid': X_madrid, 'y_madrid': y_madrid,
        'X_amsterdam': X_amsterdam, 'y_amsterdam': y_amsterdam,
        'feature_names': FEATURE_COLS, 'scaler': scaler,
        'pixel_ids_madrid':    features_madrid[PID].values,
        'pixel_ids_amsterdam': features_amsterdam[PID].values,
        'feat_groups': {
            'overall': FEAT_OVERALL, 'indices': FEAT_INDICES, 'early': FEAT_EARLY,
            'late': FEAT_LATE, 'yoy': FEAT_YOY, 'indicators': FEAT_IND,
            'novo_anti_ambiguidade_c3': FEAT_NOVO,
        },
    }
    import pickle
    save_path = output_dir / 'preprocessed_data_v2.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'Guardado em: {save_path}')
    return data


if __name__ == '__main__':
    run_preprocessing()
