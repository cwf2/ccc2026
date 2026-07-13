'''ccc2026
'''

# import statements
import os
import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import uva_common
from ccc2026.config import CONFIG

#
# data
#

# populated by setup() — everything below is None until it's been called.
# Nothing here happens automatically on import: setup() does network access
# (downloading from OSF) and disk writes (directories, cached files), which
# shouldn't be side effects of a bare `import ccc2026`.
tokens = None
all_prefs = None
corpus_lemma_count = None
top_lemmas = None
corpus_pos_count = None
top_pos = None
corpus_morph_count = None
top_morph = None
feature_count = None


def _split_locus(locus):
    '''Split a urn's locus suffix into (pref, line).

    The rightmost "."-separated segment is always the raw line number;
    texts with no book subdivision (e.g. Sack of Troy) get "00" rather
    than the old " " placeholder hack.
    '''
    if "." in locus:
        pref, line = locus.rsplit(".", 1)
    else:
        pref, line = "00", locus
    return pref, line


def setup(force_download=False):
    '''Ensure local directories exist, download any missing token tables
    from OSF, and load/prepare the corpus-wide `tokens` table and derived
    feature lists. Call this once near the top of a notebook before using
    anything else in this package.
    '''
    global tokens, all_prefs
    global corpus_lemma_count, top_lemmas, corpus_pos_count, top_pos
    global corpus_morph_count, top_morph, feature_count

    # ensure non-version-controlled local directories exist
    for path in [CONFIG["data_dir"], CONFIG["plot_dir"]]:
        os.makedirs(path, exist_ok=True)

    # each text is archived on OSF as its own CSV under data/tokens/; download
    # any that aren't already present locally
    tokens_dir = os.path.join(CONFIG["data_dir"], "tokens")
    os.makedirs(tokens_dir, exist_ok=True)
    for name, file in CONFIG["texts"]:
        if force_download or (not os.path.exists(os.path.join(tokens_dir, file))):
            uva_common.download(file, node_id="tokens", local_dir=tokens_dir)

    # concatenate all per-text token tables into one corpus-wide frame
    tokens = pd.concat(
        [pd.read_csv(f, dtype=str) for f in sorted(glob.glob(os.path.join(tokens_dir, "*.csv")))],
        ignore_index=True,
    )

    # recover "pref" (book) and "line" from the urn's locus suffix
    _locus = tokens["urn"].str.split(":").str[-1]
    tokens[["pref", "line"]] = pd.DataFrame(_locus.map(_split_locus).tolist(), index=tokens.index)

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

    # drop very infrequent tags
    pos_cutoff = 100
    top_pos = corpus_pos_count[corpus_pos_count > pos_cutoff].index

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
        "lemma": top_lemmas,
        "pos": top_pos,
        "morph": top_morph,
    }

#
# training
#

def run_training(feature_set, sample_size=1000, seed=1, z_cap=None, min_ratio=0.7):
    ''' train on a feature set, return trained models

    Sampling is stratified but, by default (z_cap=None), proportional to
    each (author, class) group's raw token count — each group is chopped
    into as many non-overlapping sample_size chunks as it has tokens for,
    so a large group (e.g. currently Dionysiaca-narrative, ~19% of all
    composite samples) supplies far more training samples than a small one
    (e.g. Sack of Troy-speech, ~0.2%), letting it disproportionately shape
    the PCA axes and decision boundary.

    z_cap, if given, caps any group whose token count exceeds z_cap
    standard deviations above the mean of the 12 real narration/speech
    groups (the Odysseus-Apologue "oth" category is excluded from that
    mean/sd, since it isn't used to train the classifier either, but is
    still eligible to be capped itself if it ever became large enough to
    qualify). Groups under the cap are unaffected — same non-overlapping
    chunks as the uncapped default, no invented samples for small groups
    (a group already too small to reach the cap can't be capped further;
    there's no equivalent "floor" option, since manufacturing samples for
    an under-represented group like Sack of Troy's 909-token speech class —
    the entire corpus of what Triphiodorus wrote — would only produce
    heavily-overlapping, non-independent samples that don't add real
    information).

    Every group also has its own leftover remainder after full-size
    chunking (n_toks % sample_size tokens). min_ratio controls what happens
    to it: kept as one extra, undersized chunk if it's at least
    min_ratio * sample_size tokens, dropped otherwise. A composite sample's
    per-feature relative-frequency estimate has standard error proportional
    to 1/sqrt(n), so an undersized chunk of ratio r has sqrt(1/r) times a
    full sample's noise — min_ratio=0.7 (the default, matching
    rolling_samples'/rolling_dialogism's own min_ratio) bounds that at
    about 1.19x, i.e. roughly 20% more per-feature noise than a full-size
    sample, which is mild enough not to meaningfully distort a PCA fit
    dominated by hundreds of full-precision samples. min_ratio=1.0 always
    drops the remainder (nothing below a full sample survives); a lower
    min_ratio keeps noisier remainders, trading fidelity for keeping every
    group represented (e.g. Sack of Troy-speech's 909-token remainder, at
    ratio 0.909, clears any min_ratio up to that value).
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

    # cap threshold, computed from the 12 real nar/spk groups only
    cap_tokens = None
    if z_cap is not None:
        group_counts = group_ids.value_counts()
        real_group_counts = group_counts[~group_counts.index.str.endswith("-oth")]
        cap_tokens = real_group_counts.mean() + z_cap * real_group_counts.std(ddof=0)

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Sample labels. A group's usable pool is its raw token count, or
    # cap_tokens if smaller (see above). That pool divides into
    # n_full_chunks *full* sample_size chunks, plus a leftover remainder —
    # kept as one extra, undersized chunk if it clears min_ratio, dropped
    # otherwise (see min_ratio docs above). Either way, excluded tokens get
    # no chunk label (NaN) and so drop out of every downstream
    # groupby(sample_ids).
    sample_ids = pd.Series(index=tokens.index, dtype=object)
    for group in group_ids.unique():
        group_mask = group_ids == group
        n_toks = sum(group_mask)
        perm = rng.permutation(n_toks)
        keep_tokens = cap_tokens if (cap_tokens is not None and n_toks > cap_tokens) else n_toks
        n_full_chunks = int(keep_tokens // sample_size)
        remainder = keep_tokens - n_full_chunks * sample_size
        effective_max = keep_tokens if remainder >= min_ratio * sample_size else n_full_chunks * sample_size
        chunk_id = np.where(perm < effective_max, perm // sample_size, np.nan)
        sample_ids.loc[group_mask] = chunk_id
    sample_ids = sample_ids.dropna()
    sample_ids = group_ids.loc[sample_ids.index] + "-" + sample_ids.map(lambda f: f"{int(f):03d}")

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
        z_cap = z_cap,
        cap_tokens = cap_tokens,
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
    '''calculate rolling samples across the corpus, one book at a time so
    windows don't cross book boundaries — grouping on (work, pref) together,
    since pref alone repeats across works (e.g. Iliad and Odyssey book 1)'''

    book_groups = [tokens["work"], tokens["pref"]]

    # how many tokens in each sample
    tokens_per_sample = (
        tokens["lemma"]
        .groupby(book_groups)
        .rolling(window=window_size, center=True,
                 min_periods=int(window_size * min_ratio))
        .agg("count")
        .reset_index(level=[0, 1], drop=True)
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
            .groupby(book_groups)
            .rolling(window=window_size, center=True, min_periods=int(window_size * 0.7))
            .agg("sum")
            .reset_index(level=[0, 1], drop=True)
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


