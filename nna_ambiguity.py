"""
nearest_neighbou.py
====================================
Nearest neighbour analysis for stimuli.

Uses raw avgpool (penultimate layer) embeddings loaded from separate JSON files,
one per condition. Each JSON is expected to have the format:
    { "cat4.png": [2048-dim vector], ... }
    { "cat4_edges.png": [2048-dim vector], ... }
    { "cat4_hq_outline.png": [2048-dim vector], ... }

For each edges / outline image:
  - Find its nearest neighbour among ALL other images of the same condition
  - Check if NN is same category or different
  - Separate plots for distraction and good categories

Produces 2 plots:
  Plot 1 — NN confusion grid (3 conditions × 5 distraction cats)
  Plot 2 — NN confusion grid (3 conditions × 5 good cats)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# FILE PATHS  ← update these to point to your avgpool JSON files, one per condition
# ══════════════════════════════════════════════════════════════════════════════
EMBEDDING_PATHS = {
    'silhouette': "./path/to/your/embedding/file/", # path to silhouette embedding json file
    'edges':      "./path/to/your/embedding/file/",
    'outline':    "./path/to/your/embedding/file/",
}

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
CATEGORIES = [
    'airplane', 'bear', 'bicycle', 'bird', 'boat', 'bottle',
    'car', 'cat', 'chair', 'clock', 'dog', 'elephant',
    'keyboard', 'knife', 'oven', 'truck'
]

Glob_Suff_Cats        = ['airplane', 'bicycle', 'chair', 'bottle', 'knife']
Glob_Ins_Cats = ['car', 'keyboard', 'oven', 'truck', 'boat']

def get_group(cat):
    if cat in Glob_Suff_Cats:        return 'GSC'
    if cat in Glob_Ins_Cats: return 'GIC'
    return 'other'

GROUP_COLORS = {'GSC': '#4CAF50', 'GIC': '#F44336', 'other': '#9E9E9E'}

COND_COLORS = {
    'silhouette': '#000000',
    'edges':      '#000000',
    'outline':    '#000000',
}

# ══════════════════════════════════════════════════════════════════════════════
# LOAD & PARSE
# Each JSON file covers a single condition; keys encode condition implicitly
# via their suffix (or lack thereof).
# ══════════════════════════════════════════════════════════════════════════════
def parse_key(name):
    """
    Returns (shape_spec, category) from a filename key.
    Handles:
      cat4.png              -> silhouette
      cat4_edges.png        -> edges
      cat4_hq_outline.png   -> outline
      cat4_noise_filled.png -> noise_filled
    The caller already knows the condition from which file was loaded.
    """
    stem = name.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
    for suffix in ('_noise_filled', '_hq_outline', '_edges'):
        stem = stem.replace(suffix, '')
    category = ''.join([c for c in stem if not c.isdigit()])
    return stem, category   # shape_spec, category


print('Loading avgpool embeddings...')
cond_emb = {}
for cond, path in EMBEDDING_PATHS.items():
    print(f'  Loading {cond} from:\n    {path}')
    with open(path) as f:
        raw = json.load(f)
    print(f'    {len(raw)} keys found in file')
    cond_emb[cond] = {}
    for name, vec in raw.items():
        shape_spec, category = parse_key(name)
        if category not in CATEGORIES:
            continue
        cond_emb[cond][name] = {
            'vec':        np.array(vec, dtype=np.float32),
            'category':   category,
            'shape_spec': shape_spec,
        }
    print(f'    {len(cond_emb[cond])} embeddings retained after category filter')

# ══════════════════════════════════════════════════════════════════════════════
# NEAREST NEIGHBOUR FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def find_nn(query_name, query_vec, candidate_dict, query_shape_spec):
    """Find nearest neighbour by cosine similarity, excluding same shape_spec."""
    best_sim  = -1
    best_name = None
    for name, meta in candidate_dict.items():
        if name == query_name:
            continue
        if meta['shape_spec'] == query_shape_spec:
            continue
        sim = 1 - cosine(query_vec, meta['vec'])
        if sim > best_sim:
            best_sim  = sim
            best_name = name
    return best_name, best_sim


def compute_nn_df(source_emb, candidate_emb):
    """
    For each image in source_emb, find its NN in candidate_emb.
    Returns DataFrame with category, nn_category, correct_nn, etc.
    """
    rows = []
    for name, meta in source_emb.items():
        nn_name, nn_sim = find_nn(
            name, meta['vec'], candidate_emb, meta['shape_spec']
        )
        if nn_name is None:
            continue
        nn_cat = candidate_emb[nn_name]['category']
        rows.append({
            'image':       name,
            'category':    meta['category'],
            'shape_spec':  meta['shape_spec'],
            'group':       get_group(meta['category']),
            'nn_name':     nn_name,
            'nn_category': nn_cat,
            'nn_sim':      nn_sim,
            'correct_nn':  nn_cat == meta['category'],
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE NN FOR EACH CONDITION
# (each condition's pool used as both query and candidate)
# ══════════════════════════════════════════════════════════════════════════════
print('\nComputing nearest neighbours...')
nn_dfs = {}
for cond in ['silhouette', 'edges', 'outline']:
    print(f'  {cond}...')
    nn_dfs[cond] = compute_nn_df(cond_emb[cond], cond_emb[cond])
    print(f'    {len(nn_dfs[cond])} images analysed')

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY PRINTOUT
# ══════════════════════════════════════════════════════════════════════════════
print('\n── % correct NN per category per condition ───────────────────────────────')
print(f'{"Category":12s} {"Group":12s}', end='')
for cond in ['silhouette', 'edges', 'outline']:
    print(f'  {cond:>12s}', end='')
print()
print('-' * 56)

for cat in CATEGORIES:
    grp = get_group(cat)
    print(f'{cat:12s} {grp:12s}', end='')
    for cond in ['silhouette', 'edges', 'outline']:
        sub = nn_dfs[cond][nn_dfs[cond]['category'] == cat]
        pct = sub['correct_nn'].mean() * 100 if len(sub) > 0 else 0
        print(f'  {pct:11.1f}%', end='')
    print()

# ══════════════════════════════════════════════════════════════════════════════
# HELPER: confusion grid (3 conditions × N categories)
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_grid(cat_list, group_label, filename):
    fig, axes = plt.subplots(3, len(cat_list),
                              figsize=(len(cat_list) * 4, 14))
    # ensure axes is always 2D
    if len(cat_list) == 1:
        axes = axes.reshape(3, 1)

    for col, cat in enumerate(cat_list):
        for row, cond in enumerate(['silhouette', 'edges', 'outline']):
            ax  = axes[row, col]
            sub = nn_dfs[cond][nn_dfs[cond]['category'] == cat]
            nn_counts  = sub['nn_category'].value_counts()
            colors_bar = [GROUP_COLORS[get_group(c)] for c in nn_counts.index]
            ax.bar(range(len(nn_counts)), nn_counts.values,
                   color=colors_bar, alpha=0.85, edgecolor='white')
            ax.set_xticks(range(len(nn_counts)))
            ax.set_xticklabels(nn_counts.index, rotation=45, ha='right', fontsize=12)
            ax.set_title(f'{cat} — {cond}', fontsize=12,
                         color=COND_COLORS[cond], fontweight='bold')
            ax.set_ylabel('Count', fontsize=12)

    plt.suptitle(f'{group_label}: NN confusion across conditions\n'
                 f'Green=GSC, Red=GIC, Grey=other',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved -> {filename}')


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# Plot 1 — Confusion grid: distraction categories
plot_confusion_grid(
    Glob_Ins_Cats,
    'GIC categories',
    'nn_confusion_GIC.png'
)

# Plot 2 — Confusion grid: good categories
plot_confusion_grid(
    Glob_Suff_Cats,
    'GSC categories',
    'nn_confusion_GSC.png'
)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
print('\nAll saved.')
import os, shutil
os.makedirs('results/plots', exist_ok=True)
for f in ['nn_confusion_GIC.png',
          'nn_confusion_GSC.png']:
    if os.path.exists(f):
        shutil.move(f, f'results/plots/{f}')
