"""
silhouette_variance_analysis.py
====================================
Within-category vs between-category silhouette variance analysis.

Hypothesis: if silhouettes are missing discriminative cues, GIC (Good Image Categories
for distraction) should show lower within/between distance ratio (silhouettes are
undifferentiated) compared to GSC (Good Silhouette Categories, silhouettes are
distinctive).

Metrics per category:
  - Mean within-category cosine distance (same category silhouettes)
  - Mean between-category cosine distance (different category silhouettes)
  - Ratio: within/between (lower = less discriminative silhouettes)
  - Silhouette score (sklearn) — standard cluster quality metric

Uses avgpool (penultimate layer) embeddings loaded from a JSON file.
Keys expected: cat4.png -> [2048-dim vector]
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import cosine, pdist, squareform
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# FILE PATH  ← update this to your avgpool JSON file
# ══════════════════════════════════════════════════════════════════════════════
EMBEDDING_PATH = "./path/to/your/embedding/file/", # path to silhouette embedding json file
# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
CATEGORIES = [
    'airplane', 'bear', 'bicycle', 'bird', 'boat', 'bottle',
    'car', 'cat', 'chair', 'clock', 'dog', 'elephant',
    'keyboard', 'knife', 'oven', 'truck'
]

GSC_CATS = ['airplane', 'bicycle', 'chair', 'bottle', 'knife']   # Good Silhouette Categories
GIC_CATS = ['car', 'keyboard', 'oven', 'truck', 'boat']          # Good Image Categories
OTHER_CATS = [c for c in CATEGORIES if c not in GSC_CATS and c not in GIC_CATS]

def get_group(cat):
    if cat in GSC_CATS:  return 'GSC'
    if cat in GIC_CATS:  return 'GIC'
    return 'other'

GROUP_COLORS = {'GSC': '#4CAF50', 'GIC': '#F44336', 'other': '#9E9E9E'}
group_order  = ['GSC', 'GIC', 'other']

# ══════════════════════════════════════════════════════════════════════════════
# LOAD & PARSE
# ══════════════════════════════════════════════════════════════════════════════
print('Loading avgpool embeddings...')
print(f'  File: {EMBEDDING_PATH}')
with open(EMBEDDING_PATH) as f:
    raw = json.load(f)
print(f'  {len(raw)} keys found in file')

sil_emb = {}
for name, vec in raw.items():
    stem = name.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
    # Strip any condition suffixes so this works with any key style
    for suffix in ('_noise_filled', '_hq_outline', '_edges'):
        stem = stem.replace(suffix, '')
    cat = ''.join([c for c in stem if not c.isdigit()])
    if cat in CATEGORIES:
        sil_emb[name] = {'vec': np.array(vec, dtype=np.float32), 'category': cat}

print(f'  {len(sil_emb)} embeddings retained after category filter')
cat_counts = pd.Series([v['category'] for v in sil_emb.values()]).value_counts()
print(cat_counts.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# BUILD PER-CATEGORY EMBEDDING MATRICES
# ══════════════════════════════════════════════════════════════════════════════
cat_vecs = {cat: [] for cat in CATEGORIES}
for name, meta in sil_emb.items():
    cat_vecs[meta['category']].append(meta['vec'])
cat_vecs = {cat: np.stack(vecs) for cat, vecs in cat_vecs.items() if vecs}

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE WITHIN AND BETWEEN CATEGORY DISTANCES
# ══════════════════════════════════════════════════════════════════════════════
print('\nComputing pairwise distances...')

all_vecs  = []
all_cats  = []
all_names = []
for name, meta in sorted(sil_emb.items()):
    all_vecs.append(meta['vec'])
    all_cats.append(meta['category'])
    all_names.append(name)

X = np.stack(all_vecs)
dist_matrix = squareform(pdist(X, metric='cosine'))
cat_array = np.array(all_cats)

rows = []
for cat in CATEGORIES:
    if cat not in cat_vecs:
        continue
    idx_cat   = np.where(cat_array == cat)[0]
    idx_other = np.where(cat_array != cat)[0]

    within_dists = []
    for i in range(len(idx_cat)):
        for j in range(i+1, len(idx_cat)):
            within_dists.append(dist_matrix[idx_cat[i], idx_cat[j]])

    between_dists = []
    for i in idx_cat:
        for j in idx_other:
            between_dists.append(dist_matrix[i, j])

    within_mean  = np.mean(within_dists)
    between_mean = np.mean(between_dists)
    ratio        = within_mean / between_mean

    rows.append({
        'category':     cat,
        'group':        get_group(cat),
        'within_mean':  within_mean,
        'between_mean': between_mean,
        'ratio':        ratio,
        'within_std':   np.std(within_dists),
        'between_std':  np.std(between_dists),
        'n':            len(idx_cat),
    })

df = pd.DataFrame(rows).sort_values('ratio')

print('\n── Within vs Between category cosine distance ────────────────────────────')
print(f'{"Category":12s}  {"Group":6s}  {"Within":>8s}  {"Between":>8s}  '
      f'{"Ratio":>8s}  {"n":>4s}')
print('-' * 58)
for _, row in df.iterrows():
    print(f'{row["category"]:12s}  {row["group"]:6s}  '
          f'{row["within_mean"]:8.4f}  {row["between_mean"]:8.4f}  '
          f'{row["ratio"]:8.4f}  {int(row["n"]):4d}')

# ══════════════════════════════════════════════════════════════════════════════
# SKLEARN SILHOUETTE SCORE PER CATEGORY
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Sklearn silhouette score (higher = more distinct cluster) ─────────────')
labels = np.array([CATEGORIES.index(c) for c in all_cats])
sample_scores = silhouette_samples(X, labels, metric='cosine')

sil_scores = {}
for cat in CATEGORIES:
    idx = np.where(cat_array == cat)[0]
    if len(idx) > 0:
        sil_scores[cat] = np.mean(sample_scores[idx])

for cat, score in sorted(sil_scores.items(), key=lambda x: x[1], reverse=True):
    grp = get_group(cat)
    bar = '█' * int((score + 0.5) * 30)
    print(f'  {cat:12s} ({grp:5s}): {score:+.4f} {bar}')

# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL TEST: GSC vs GIC ratio
# ══════════════════════════════════════════════════════════════════════════════
gsc_ratios = df[df['group'] == 'GSC']['ratio'].values
gic_ratios = df[df['group'] == 'GIC']['ratio'].values
stat, p = mannwhitneyu(gsc_ratios, gic_ratios, alternative='less')
print(f'\nMann-Whitney U (GSC ratio < GIC ratio): U={stat:.1f}, p={p:.4f}')
print(f'GSC mean ratio: {gsc_ratios.mean():.4f}')
print(f'GIC mean ratio: {gic_ratios.mean():.4f}')

# ══════════════════════════════════════════════════════════════════════════════
# SORT ORDER: group-first (GSC, GIC, other), then ratio within group
# ══════════════════════════════════════════════════════════════════════════════
group_order_map = {'GSC': 0, 'GIC': 1, 'other': 2}
df['group_order'] = df['group'].map(group_order_map)
df_sorted = df.sort_values(['group_order', 'ratio'])
cats_ord  = df_sorted['category'].tolist()
groups    = df_sorted['group'].tolist()
x = np.arange(len(cats_ord))

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: Within vs Between bar chart per category
# ══════════════════════════════════════════════════════════════════════════════
w = 0.35
fig, ax = plt.subplots(figsize=(14, 7))
ax.bar(x - w/2, df_sorted['within_mean'],  w,
       color=[GROUP_COLORS[g] for g in groups], alpha=0.85,
       edgecolor='white', label='within-category')
ax.bar(x + w/2, df_sorted['between_mean'], w,
       color=[GROUP_COLORS[g] for g in groups], alpha=0.40,
       edgecolor='white', hatch='//', label='between-category')

ax.set_xticks(x)
ax.set_xticklabels(cats_ord, rotation=45, ha='right')
ax.set_ylabel('Mean cosine distance')
ax.set_title('Within vs between-category cosine distance for silhouette embeddings\n'
             'Smaller gap = less discriminative silhouette')

pos = 0
for grp in group_order:
    grp_cats = [c for c in cats_ord if get_group(c) == grp]
    if not grp_cats:
        continue
    n = len(grp_cats)
    ax.axvline(pos - 0.5, color='gray', linewidth=0.8, linestyle='--')
    ax.text(pos + n/2 - 0.5, ax.get_ylim()[1] * 0.97,
            grp, ha='center', fontsize=12,
            color=GROUP_COLORS[grp], fontweight='bold')
    pos += n

handles = [
    mpatches.Patch(color='gray', alpha=0.85, label='within-category (solid)'),
    mpatches.Patch(color='gray', alpha=0.40, hatch='//', label='between-category (hatched)'),
] + [mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in group_order]
ax.legend(handles=handles, fontsize=12, bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig('silhouette_within_between.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved -> silhouette_within_between.png')

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: Ratio plot
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))
ax.bar(x, df_sorted['ratio'],
       color=[GROUP_COLORS[g] for g in groups], alpha=0.85, edgecolor='white')
ax.axhline(df_sorted['ratio'].mean(), color='black', linewidth=1,
           linestyle='--', alpha=0.5,
           label=f'mean={df_sorted["ratio"].mean():.3f}')
ax.set_xticks(x)
ax.set_xticklabels(cats_ord, rotation=45, ha='right')
ax.set_ylabel('Within / Between distance ratio')
ax.set_title('Silhouette discriminability ratio (within/between)\n'
             'Lower = more distinctive silhouette | Higher = ambiguous silhouette')

pos = 0
for grp in group_order:
    grp_cats = [c for c in cats_ord if get_group(c) == grp]
    if not grp_cats:
        continue
    n = len(grp_cats)
    ax.axvline(pos - 0.5, color='gray', linewidth=0.8, linestyle='--')
    ax.text(pos + n/2 - 0.5, ax.get_ylim()[1] * 0.97,
            grp, ha='center', fontsize=12,
            color=GROUP_COLORS[grp], fontweight='bold')
    pos += n

handles = [mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in group_order]
ax.legend(handles=handles + [plt.Line2D([0], [0], color='black', linestyle='--',
          label='overall mean')], fontsize=12)
plt.tight_layout()
plt.savefig('silhouette_ratio.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved -> silhouette_ratio.png')

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3: Silhouette score per category
# ══════════════════════════════════════════════════════════════════════════════
sil_df = pd.DataFrame({'category': list(sil_scores.keys()),
                        'score':    list(sil_scores.values())})
sil_df['group'] = sil_df['category'].apply(get_group)
sil_df = sil_df.sort_values('score', ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
bar_colors = [GROUP_COLORS[sil_df.loc[i, 'group']] for i in sil_df.index]
ax.bar(range(len(sil_df)), sil_df['score'],
       color=bar_colors, alpha=0.85, edgecolor='white')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(range(len(sil_df)))
ax.set_xticklabels(sil_df['category'], rotation=45, ha='right')
ax.set_ylabel('Mean silhouette score')
ax.set_title('Sklearn silhouette score per category\n'
             'Higher = silhouette embeddings form a more distinct cluster')
handles = [mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in group_order]
ax.legend(handles=handles, fontsize=12)
plt.tight_layout()
plt.savefig('silhouette_cluster_score.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved -> silhouette_cluster_score.png')

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
print('\nAll saved.')
import os, shutil
os.makedirs('results/plots', exist_ok=True)
for f in ['silhouette_within_between.png', 'silhouette_ratio.png',
          'silhouette_cluster_score.png']:
    if os.path.exists(f):
        shutil.move(f, f'results/plots/{f}')
