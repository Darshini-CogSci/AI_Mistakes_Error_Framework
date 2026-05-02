"""
t-SNE of avgpool embeddings for images labelled knife (true/false)
across silhouette, edges, outline, and noise-filled conditions.
No normal images used. Condition = colour, true/false knife = marker.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


AVGPOOL_FILES = {
    'silhouette':   "./path/to/your/embedding/file/", # a file containing silhouette condition embeddings for avgpool layer
    'edges':"./path/to/your/embedding/file/", 
    'outline':"./path/to/your/embedding/file/",
    'noise_filled':"./path/to/your/embedding/file/", # Add files for more conditions for comparision here
}

TRIAL_CSVS = {
    'silhouette':   "./path/to/your/trial/data/", # a file Resnet50 trial data with final top label response for all the images of silhouette condition
    'edges':        "./path/to/your/trial/data/",
    'outline':      "./path/to/your/trial/data/",
    'noise_filled': "./path/to/your/trial/data/", # Add files for more conditions for comparision here
}

def get_true_category(imagename):
    stem = imagename.replace('.png','').replace('.jpg','').replace('.jpeg','')
    for suffix in ['_edges', '_hq_outline', '_noise_filled']:
        stem = stem.replace(suffix, '')
    return ''.join([c for c in stem if not c.isdigit()])

# ── Collect knife-labelled images from all four conditions ────────────────────
print('Loading trial data and avgpool embeddings...')
knife_rows = []

for condition, csv_path in TRIAL_CSVS.items():
    with open(AVGPOOL_FILES[condition]) as f:
        avgpool_emb = {k: np.array(v) for k, v in json.load(f).items()}

    df = pd.read_csv(csv_path)
    knife_df = df[df['object_response'] == 'knife']
    print(f'  {condition}: {len(knife_df)} knife responses')

    for _, row in knife_df.iterrows():
        img_name = row['imagename']
        true_cat = row['category'] if 'category' in df.columns and pd.notna(row.get('category')) \
                   else get_true_category(img_name)

        vec = avgpool_emb.get(img_name)
        if vec is None:
            # Try matching by stem, ignoring condition suffixes in embedding keys
            stem = img_name.rsplit('.', 1)[0]  # e.g. airplane1
            for key, v in avgpool_emb.items():
                key_stem = key.rsplit('.', 1)[0]
                # Strip known suffixes from embedding key
                for suffix in ['_edges', '_hq_outline', '_noise_filled']:
                    key_stem = key_stem.replace(suffix, '')
                if key_stem == stem:
                    vec = v
                    break
        if vec is None:
            print(f'    WARNING: {img_name} not found')
            continue

        is_true = (true_cat == 'knife')
        knife_rows.append({
            'imagename':     img_name,
            'condition':     condition,
            'true_category': true_cat,
            'is_true_knife': is_true,
            'label':         'true knife' if is_true else 'false knife',
            'embedding':     vec
        })

kdf = pd.DataFrame(knife_rows)
print(f'\nTotal: {len(kdf)} knife images')
print(kdf.groupby(['condition', 'label']).size().unstack(fill_value=0))

# ── t-SNE ─────────────────────────────────────────────────────────────────────
X = np.stack(kdf['embedding'].values)
print(f'\navgpool dim: {X.shape[1]}, vectors: {len(X)}')
print('Running PCA...')
X_pca = PCA(n_components=min(50, X.shape[0]-1, X.shape[1]), random_state=42).fit_transform(X)
print('Running t-SNE...')
X_tsne = TSNE(n_components=2, perplexity=min(30, len(X)//4),
              random_state=42, max_iter=1000).fit_transform(X_pca)
print('Done')

# ── Plot ───────────────────────────────────────────────────────────────────────
cond_colors = {
    'silhouette':   '#2196F3',
    'edges':        "#FF2600",
    'outline':      "#27B05E",
    'noise_filled': "#D5E91E",
}

fig, ax = plt.subplots(figsize=(12, 9))

for i, (_, row) in enumerate(kdf.iterrows()):
    x, y    = X_tsne[i, 0], X_tsne[i, 1]
    color   = cond_colors[row['condition']]
    is_true = row['is_true_knife']
    ax.scatter(x, y,
               c=color,
               marker='X' if is_true else 'o',
               s=55,
               alpha=0.82,
               edgecolors='black' if is_true else 'none',
               linewidths=1.0,
               zorder=3)

cond_handles = [mpatches.Patch(color=cond_colors[c], label=c.replace('_','-'))
                for c in cond_colors]
label_handles = [
    plt.Line2D([0],[0], marker='X', color='gray', linestyle='none',
               markersize=8, label='true knife (X, black edge)'),
    plt.Line2D([0],[0], marker='o', color='gray', linestyle='none',
               markersize=8, markeredgecolor='black',
               label='false knife (o)'),
]
leg1 = ax.legend(handles=cond_handles, bbox_to_anchor=(1.01, 1),
                 loc='upper left', fontsize=12, title='Condition')
ax.add_artist(leg1)
ax.legend(handles=label_handles, bbox_to_anchor=(1.01, 0.45),
          loc='upper left', fontsize=12, title='Label type')

ax.set_title('avgpool embeddings: knife true/false labels across conditions\n'
             'Circles = true knife | X (black edge) = false knife', fontsize=12)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
plt.tight_layout()

import os
os.makedirs('results/plots', exist_ok=True)
plt.savefig('results/plots/avgpool_knife_tsne.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved -> results/plots/avgpool_knife_tsne.png')
