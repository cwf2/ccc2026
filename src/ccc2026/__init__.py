'''ccc2026
'''

# import statements
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from ccc2026 import config

#
# data
#


# load tokens
token_file = os.path.join(config.DATA_DIR, "tokens.tsv")
tokens = pd.read_csv(token_file, delimiter="\t", dtype=str)

# force Triphiodorus pref to string
tokens.loc[tokens["work"]=="Sack of Troy", "pref"] = " "

# all work/pref combos - useful for dropdown
all_prefs = {}
for work in tokens["work"].unique():
    all_prefs[work] = [pref for pref in tokens.loc[tokens["work"]==work, "pref"].unique()]

#
# feature selection
#

# corpus-wide count for all non-punctuation lemmas, from most frequent to least
corpus_lemma_count = tokens["lemma"].value_counts()

# a list of the top lemmas
top_lemmas = corpus_lemma_count.head(100).index

# get corpus-wide POS counts
corpus_pos_count = tokens["pos"].value_counts()

# select all pos features
all_pos = corpus_pos_count.index

# melt morph column values
morph = tokens.melt(
        value_vars=["verbform", "mood", "tense", "voice", "person", "number", "case", "gender"],
        ignore_index = False,
    ).dropna()

# agg morphs as list, add to token table
tokens["morph"] = morph.groupby(morph.index).agg(morph = ("value", list))

# get corpus wide morph counts
corpus_morph_count = tokens["morph"].explode().value_counts()

# select all but anomalously low
top_morph = corpus_morph_count[corpus_morph_count > 1000].index

# (truncated) feature counts bundled for export
feature_count = {
    "lemma": corpus_lemma_count[:100],
    "pos": corpus_pos_count,
    "morph": corpus_morph_count[corpus_morph_count > 1000],
}

#
# training
#

def run_training(feature_set, sample_size=1000, seed=1):
    ''' train on a feature set, return trained models
    '''
    
    # Narratological groups
    nr_mask = tokens["speaker"].isna()
    sp_mask = tokens["speaker"].notna() & tokens["speaker"].ne("Odysseus-Apologue")

    # default group is "other"
    nara_group_ids = pd.Series("oth", index=tokens.index)
    nara_group_ids[nr_mask] = "nar"
    nara_group_ids[sp_mask] = "spk"

    # Authorship groups
    auth_group_ids = tokens["work"].str.slice(0,4)

    # Combined two-factor group
    group_ids = auth_group_ids + "-" + nara_group_ids

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Sample labels
    sample_ids = pd.Series(index=tokens.index)
    for group in group_ids.unique():
        n_toks = sum(group_ids==group)
        sample_ids.loc[group_ids==group] = rng.permutation(n_toks) // sample_size
    sample_ids = group_ids + "-" + sample_ids.map(lambda f: f"{int(f):03d}")

    # Calculate sample sizes
    tokens_per_sample = tokens.groupby(sample_ids).size()

    # use a different scaler for each feat class
    scalers = {}

    # collect output vectors for each feat class
    parts = []

    for col, features in feature_set.items():

        # Generate feature tallies
        raw = (tokens[col]
            .explode()
            .where(lambda x: x.isin(features))
            .pipe(pd.get_dummies)
            .groupby(level=0).agg("sum")
            .groupby(sample_ids).agg("sum")
        )
        
        # normalize as freq / 1000 words
        normalized = raw.div(tokens_per_sample, axis=0) * 1000

        # scale
        scaler = StandardScaler()
        scaled = pd.DataFrame(
            data = scaler.fit_transform(normalized),
            columns = [f"{col}_{f}" for f in features],
            index = normalized.index,
        )
        scalers[col] = scaler
        parts.append(scaled)

    # combine all vectors into one table
    composite = pd.concat(parts, axis=1)
    
    # principal components analysis
    pca_model = PCA(n_components=3)

    pca = pd.DataFrame(
        data = pca_model.fit_transform(composite),
        columns = ["PC1", "PC2", "PC3"],
        index = composite.index,
    )

    # logistic regression on nar/spk only
    mask = ~pca.index.str.contains("-oth-")
    X = pca.loc[mask, ["PC1", "PC2"]].values
    y = pca.index[mask].str.contains("spk").astype(int)
    clf = LogisticRegression()
    clf.fit(X, y)

    # used for labeling PCA graphs
    nara_label = nara_group_ids.groupby(sample_ids).agg("first").values
    auth_label = auth_group_ids.groupby(sample_ids).agg("first").values

    # difference of z-scores
    diff = (
            composite.loc[nara_label=="spk"].agg("mean") -
            composite.loc[nara_label=="nar"].agg("mean")
        ).sort_values()
 
    # bundle outputs
    return dict(
        feature_set = feature_set,
        sample_size = sample_size,
        seed = seed,
        nara_label = nara_label,
        auth_label = auth_label,
        scalers = scalers,
        pca_model = pca_model,
        scaled = scaled,
        pca = pca,
        clf = clf,
        diff = diff,
    )


def plot_training(training, show_decision_boundary=True):
    '''plot first two principal components of training data - return figure'''

    work_order = ["Ilia", "Odys", "Argo", "Post", "Sack", "Dion"]
    class_order = ["nar", "spk", "oth"]

    # plot using Seaborn
    g = sns.relplot(data=training["pca"],
        x = "PC1",
        y = "PC2",
        hue = training["auth_label"],
        hue_order = work_order,
        style = training["nara_label"],
        style_order = class_order,
        palette = "deep",
        legend = "full",
    )

    # set figure size
    g.figure.set_size_inches(7,4)

    # get text of default legend
    all_handles = g._legend.legend_handles
    all_labels = [t.get_text() for t in g._legend.get_texts()]
    # remove default legend
    g._legend.remove()

    # divide legend text into hue, style labels
    hue_handles, hue_labels = [], []
    style_handles, style_labels = [], []
    for h, l in zip(all_handles, all_labels):
        if l in work_order:
            hue_handles.append(h); hue_labels.append(l)
        elif l in class_order:
            style_handles.append(h); style_labels.append(l)

    # generate two legends, one for hue and one for style
    leg1 = g.ax.legend(hue_handles, hue_labels, title="Work",
                     bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    g.ax.add_artist(leg1)  # keep leg1 when adding leg2
    g.ax.legend(style_handles, style_labels, title="Text Class",
              bbox_to_anchor=(1.01, 0.5), loc="upper left", frameon=False)

    # add decision boundary
    if show_decision_boundary:
    
        # get the axis for the existing plot
        xlim = g.ax.get_xlim()
    
        # solve for y at each x endpoint: coef[0]*x + coef[1]*y + intercept = 0
        #   => y = -(coef[0]*x + intercept) / coef[1]
        w = training["clf"].coef_[0]
        b = training["clf"].intercept_[0]
        xs = np.array(xlim)
        ys = -(w[0] * xs + b) / w[1]
        
        g.ax.plot(xs, ys, "k--", linewidth=1)
        g.ax.set_xlim(xlim)  # restore limits so the line doesn't expand the plot

    return g.figure

#
# rolling samples
#

def rolling_samples(training, window_size=500, min_ratio=0.7):
    '''calculate rolling samples across the corpus'''

    # how many tokens in each sample
    tokens_per_sample = (
        tokens["lemma"]
        .rolling(window=window_size, center=True,
                 min_periods=int(window_size * min_ratio))
        .agg("count")
        .fillna(0)
        .astype(int)
    )

    # calculate one normalized, scaled feature vector per col
    parts = []
    for col, features in training["feature_set"].items():

        # raw count per sample
        raw = (tokens[col]
            .explode()
            .where(lambda x: x.isin(features))
            .pipe(pd.get_dummies)
            .groupby(level=0).agg("sum")
            .rolling(window=window_size, center=True, min_periods=int(window_size * 0.7))
            .agg("sum")
            .fillna(0)
            .astype(int)
        )

        # normalize by 1000 tokens
        normalized = raw.div(tokens_per_sample, axis=0) * 1000

        # scaled
        scaled = pd.DataFrame(
            data = training["scalers"][col].transform(normalized),
            columns = [f"{col}_{f}" for f in features],
            index = raw.index,
        )
        
        parts.append(scaled)

    # combine all vectors in one table
    composite = pd.concat(parts, axis=1)

    # remove NaN rows
    valid = composite.dropna()

    # transform vectors using pre-fitted pca model
    pca = pd.DataFrame(
        data = training["pca_model"].transform(valid),
        columns = ["PC1", "PC2", "PC3"],
        index = valid.index,
    )
    
    # calculate speechiness score
    X = pca[["PC1", "PC2"]].values
    proj = X @ training["clf"].coef_.T + training["clf"].intercept_
    
    speech_score = pd.DataFrame(dict(
            work = tokens.loc[pca.index, "work"],
            pref = tokens.loc[pca.index, "pref"],
            line = tokens.loc[pca.index, "line"],
            speech_id = tokens.loc[pca.index, "speech_id"],
            score = proj.flatten(), 
        ),
        index = pca.index)

    return dict(
        window_size = window_size,
        min_ratio = min_ratio,
        speech_score = speech_score,
    )


def plot_rolling(data, work, pref):
    '''Project rolling window using the trained PCA model'''

    # select just the data from one work/book
    mask = (data["speech_score"]["work"] == work) & (data["speech_score"]["pref"] == pref)

    # plot using Seaborn
    g = sns.relplot(
        x = data["speech_score"][mask].index,
        y = data["speech_score"][mask].score,
        kind = "line",
    )

    # give the graph a title
    g.set(
        title = work + " " + pref,
        xlabel = "line",
    )

    # set output size
    g.figure.set_size_inches(8, 4)

    # set x-tics
    xmin = data["speech_score"][mask].index.min()
    xmax = data["speech_score"][mask].index.max()
    ymin = data["speech_score"]["score"].min()
    ymax = data["speech_score"]["score"].max()

    x_ticks = []
    x_tick_labels = []
    for idx in data["speech_score"][mask].index:
        ln = data["speech_score"].loc[idx, "line"]
        if int(ln) % 50 == 0:
            if ln not in x_tick_labels:
                x_ticks.append(idx)
                x_tick_labels.append(ln)
    g.ax.plot([xmin,xmax], [0,0], "k--", lw=1)
    g.set(
        xticks = x_ticks,
        xticklabels = x_tick_labels,
        xlim = (xmin, xmax),
        ylim = (ymin, ymax),
    )

    # plot speeches as vertical grey bars
    speech_data = data["speech_score"].dropna(subset=["speech_id"])
    for sid, group in speech_data.groupby("speech_id"):
        g.ax.axvspan(group.index.min(), group.index.max(), alpha=0.15, color="gray", linewidth=0)

    return g.figure


