'''dialogism.py - weighted log-odds "dialogism" method (Burns 2026)

A port of the weighted log-odds ratio method (Monroe, Colaresi & Quinn 2008,
"Fightin' Words", as implemented in the R package tidylo) used by Patrick
Burns in "Measuring Dialogism in Latin Epic" to identify lexical and
grammatical features that distinguish speech from narrative, and to build a
composite per-token "dialogism" score.

This is an alternative to the PCA + logistic regression pipeline in
ccc2026.__init__ (run_training / rolling_samples), developed to replicate
Burns' method for comparison. See "Compare speechiness methods.ipynb".
'''

import numpy as np
import pandas as pd

import ccc2026


def weighted_log_odds(col):
    '''Weighted log-odds ratio (speech vs. narrative) for every value of `col`,
    weighted by an uninformative Dirichlet prior (Monroe, Colaresi & Quinn 2008),
    following the same method used by Burns 2026 via the R package tidylo.

    Requires ccc2026.setup() to have been called first.
    '''
    tokens = ccc2026.tokens

    # Narratological groups - same definition used throughout the package
    nr_mask = tokens["speaker"].isna()
    sp_mask = tokens["speaker"].notna() & tokens["speaker"].ne("Odysseus-Apologue")

    # explode handles both scalar columns (lemma, pos) and list columns (morph)
    exploded = tokens[col].explode().dropna()
    is_nar = nr_mask.loc[exploded.index]
    is_spk = sp_mask.loc[exploded.index]

    y_nar = exploded[is_nar].value_counts()
    y_spk = exploded[is_spk].value_counts()
    counts = pd.DataFrame({"y_nar": y_nar, "y_spk": y_spk}).fillna(0)

    # uninformative Dirichlet prior: each feature's prior pseudo-count is its
    # own marginal (background) frequency across both groups combined
    alpha = counts["y_nar"] + counts["y_spk"]
    alpha0 = alpha.sum()
    n_nar = counts["y_nar"].sum()
    n_spk = counts["y_spk"].sum()

    omega_nar = (counts["y_nar"] + alpha) / (n_nar + alpha0 - counts["y_nar"] - alpha)
    omega_spk = (counts["y_spk"] + alpha) / (n_spk + alpha0 - counts["y_spk"] - alpha)

    delta = np.log(omega_spk) - np.log(omega_nar)
    variance = 1 / (counts["y_spk"] + alpha) + 1 / (counts["y_nar"] + alpha)
    z = delta / np.sqrt(variance)

    return pd.DataFrame({
        "count_nar": counts["y_nar"],
        "count_spk": counts["y_spk"],
        "delta": delta,
        "z": z,
    }).sort_values("z", ascending=False)


def make_lexicon(wlo):
    '''Scale a weighted_log_odds table's z-scores to [0, 1] (1 = most speech-like)'''
    z = wlo["z"]
    return (z - z.min()) / (z.max() - z.min())


def build_lexicons():
    '''Build the lexical (lemma) and grammatical (POS + morph) lexicons'''

    lex_lemma = make_lexicon(weighted_log_odds("lemma"))
    lex_grammar = pd.concat([
        make_lexicon(weighted_log_odds("pos")),
        make_lexicon(weighted_log_odds("morph")),
    ])

    return dict(lemma=lex_lemma, grammar=lex_grammar)


def token_dialogism_score(lexicons):
    '''Composite per-token dialogism score.

    The lexical score is a token's lemma's lexicon value (0 if
    out-of-vocabulary); the grammatical score is the mean of the lexicon
    values for the token's POS tag and all of its morph tags. The composite
    score is the average of the two.

    Requires ccc2026.setup() to have been called first.
    '''
    tokens = ccc2026.tokens

    lex_lemma = lexicons["lemma"]
    lex_grammar = lexicons["grammar"]

    # lexical score: lemma's lexicon value, excluded from the composite mean
    # (like grammar's OOV handling below) rather than filled with 0 — 0 is
    # the most narrative-like end of the [0,1] scale, not a neutral "no
    # signal" value, so an OOV lemma shouldn't be scored as if it were
    # strong narrative evidence
    lexical_score = tokens["lemma"].map(lex_lemma)

    # grammatical score: mean of the POS tag + all morph tags' lexicon values
    def grammar_score_for_row(pos, morph):
        vals = []
        if pd.notna(pos) and pos in lex_grammar.index:
            vals.append(lex_grammar[pos])
        if isinstance(morph, list):
            vals.extend(lex_grammar[m] for m in morph if m in lex_grammar.index)
        return np.mean(vals) if vals else np.nan

    grammatical_score = tokens.apply(lambda row: grammar_score_for_row(row["pos"], row["morph"]), axis=1)

    return pd.concat(
        [lexical_score.rename("lex"), grammatical_score.rename("gram")], axis=1
    ).mean(axis=1, skipna=True)


def rolling_dialogism(score, window_size=500, min_ratio=0.7):
    '''Calculate a rolling mean of the per-token dialogism score, one book at
    a time so windows don't cross book boundaries — grouping on (work, pref)
    together, since pref alone repeats across works (e.g. Iliad and Odyssey
    book 1).

    Returns the same shape as ccc2026.rolling_samples's output (a dict with
    window_size, min_ratio, and a speech_score DataFrame with work, pref,
    line, speech_id, and score columns) so that ccc2026.plot_rolling can be
    reused unmodified on either method's output.

    Requires ccc2026.setup() to have been called first.
    '''
    tokens = ccc2026.tokens

    book_groups = [tokens["work"], tokens["pref"]]
    rolled = (
        score
        .groupby(book_groups)
        .rolling(window=window_size, center=True, min_periods=int(window_size * min_ratio))
        .mean()
        .reset_index(level=[0, 1], drop=True)
        .dropna()
    )

    speech_score = pd.DataFrame(dict(
            work = tokens.loc[rolled.index, "work"],
            pref = tokens.loc[rolled.index, "pref"],
            line = tokens.loc[rolled.index, "line"],
            speech_id = tokens.loc[rolled.index, "speech_id"],
            score = rolled,
        ),
        index = rolled.index)

    return dict(
        window_size = window_size,
        min_ratio = min_ratio,
        speech_score = speech_score,
    )
