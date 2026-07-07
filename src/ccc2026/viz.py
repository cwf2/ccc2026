'''viz.py - display/visualization helpers shared across both scoring methods

These operate on the output of ccc2026's PCA pipeline and/or dialogism.py's
Burns-method implementation, rather than belonging to either one specifically
— so they live here instead of in __init__.py or dialogism.py.

Ported from "Regions of interest.ipynb" (build_display_column) and
"Compare speechiness methods.ipynb" (plot_overlay, generalized from that
notebook's single-book z-scored overlay). The original notebook versions are
left in place rather than repointed at this module, since they're useful as
a record of how these were developed.
'''

import pandas as pd
from matplotlib import pyplot as plt


def build_display_column(tokens, lexicons, lemma_cutoff=0.5, grammar_cutoff=0.7,
                          lemma_color="red", grammar_color="green"):
    '''Wrap each token's text in a colored <span> if its lemma or any of its
    grammatical features (POS + morph) exceeds the given lexicon cutoff.
    Lemma matches take priority over grammar matches for a given token.

    tokens   — a token DataFrame with "text", "lemma", "pos", and morph columns
    lexicons — a dict with "lemma" and "grammar" lexicons, e.g. from
               dialogism.build_lexicons()
    '''
    lemma_lex = set(lexicons["lemma"].loc[lexicons["lemma"] > lemma_cutoff].index)
    grammar_lex = set(lexicons["grammar"].loc[lexicons["grammar"] > grammar_cutoff].index)

    morph_cols = ["pos", "verbform", "mood", "tense", "voice", "person", "number", "case", "gender"]
    is_lemma_hit = tokens["lemma"].isin(lemma_lex)
    is_grammar_hit = tokens[morph_cols].isin(grammar_lex).any(axis=1)

    text = tokens["text"]
    display_col = text.copy()
    display_col = display_col.where(~is_grammar_hit, f'<span style="color:{grammar_color}">' + text + "</span>")
    display_col = display_col.where(~is_lemma_hit, f'<span style="color:{lemma_color}">' + text + "</span>")
    return display_col


def plot_overlay(tokens, work, pref, pca_roll, dialogism_roll):
    '''Z-scored overlay of the PCA and dialogism rolling scores for one book,
    plotted in document order so the two methods' agreement/divergence can be
    read against the actual sequence of the poem.

    pca_roll, dialogism_roll — rolling score Series indexed like tokens
    (e.g. from ccc2026.rolling_samples(...)["speech_score"]["score"] and
    dialogism.rolling_dialogism(...)["speech_score"]["score"])
    '''
    mask = (tokens["work"] == work) & (tokens["pref"] == pref)
    idx = tokens.index[mask]

    book = pd.DataFrame({
        "dialogism": dialogism_roll.reindex(idx),
        "pca": pca_roll.reindex(idx),
        "line": tokens.loc[idx, "line"],
    }).dropna()

    # standardize both scores so they're comparable on one axis
    book["dialogism_z"] = (book["dialogism"] - book["dialogism"].mean()) / book["dialogism"].std()
    book["pca_z"] = (book["pca"] - book["pca"].mean()) / book["pca"].std()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(book.index, book["pca_z"], label="PCA + logistic regression (standardized)")
    ax.plot(book.index, book["dialogism_z"], label="weighted log-odds dialogism (standardized)")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_title(f"{work} {pref}")
    ax.set_xlabel("token index")
    ax.legend()
    return fig
