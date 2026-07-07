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


def _in_line_range(line_series, first_line=None, last_line=None):
    '''Boolean mask: does each raw "line" label fall within [first_line, last_line]?

    Compares by the leading digit run, so letter-suffixed or compound labels
    (e.g. "568a", or Nonnus' transposed "74_75") are treated as their leading
    line number — fine for picking a rough passage, not meant as a precise
    citation match.
    '''
    if first_line is None and last_line is None:
        return pd.Series(True, index=line_series.index)
    line_num = line_series.str.extract(r"^(\d+)")[0].astype(int)
    mask = pd.Series(True, index=line_series.index)
    if first_line is not None:
        mask &= line_num >= first_line
    if last_line is not None:
        mask &= line_num <= last_line
    return mask


def plot_overlay(tokens, work, pref, pca_roll, dialogism_roll, first_line=None, last_line=None):
    '''Z-scored overlay of the PCA and dialogism rolling scores for one book
    (or a line range within it), plotted in document order so the two
    methods' agreement/divergence can be read against the actual sequence
    of the poem.

    pca_roll, dialogism_roll — rolling score Series indexed like tokens
    (e.g. from ccc2026.rolling_samples(...)["speech_score"]["score"] and
    dialogism.rolling_dialogism(...)["speech_score"]["score"])

    first_line, last_line — optional; restrict the plotted range to these
    lines. Standardization is always computed over the whole book first, so
    a zoomed-in passage still reads relative to the book as a whole rather
    than being re-centered on itself.
    '''
    mask = (tokens["work"] == work) & (tokens["pref"] == pref)
    idx = tokens.index[mask]

    book = pd.DataFrame({
        "dialogism": dialogism_roll.reindex(idx),
        "pca": pca_roll.reindex(idx),
        "line": tokens.loc[idx, "line"],
    }).dropna()

    # standardize both scores so they're comparable on one axis — using the
    # whole book's stats, before any line-range restriction
    book["dialogism_z"] = (book["dialogism"] - book["dialogism"].mean()) / book["dialogism"].std()
    book["pca_z"] = (book["pca"] - book["pca"].mean()) / book["pca"].std()

    book = book[_in_line_range(book["line"], first_line, last_line)]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(book.index, book["pca_z"], label="PCA + logistic regression (standardized)")
    ax.plot(book.index, book["dialogism_z"], label="weighted log-odds dialogism (standardized)")
    ax.axhline(0, color="k", ls="--", lw=1)
    title = f"{work} {pref}"
    if first_line is not None or last_line is not None:
        title += f" ({first_line if first_line is not None else 'start'}-{last_line if last_line is not None else 'end'})"
    ax.set_title(title)
    ax.set_xlabel("token index")
    ax.legend()
    return fig


def highlighted_excerpt(tokens, work, pref, first_line=None, last_line=None, display_col="display"):
    '''Build an HTML excerpt of a book (or a line range within it), one row
    per verse line with its highlighted text.

    Requires tokens[display_col] to already be populated, e.g. via
    build_display_column() — highlighting is corpus-wide and relatively
    expensive, so it's meant to be computed once and then sliced by book/
    range repeatedly, not recomputed per excerpt.
    '''
    mask = (tokens["work"] == work) & (tokens["pref"] == pref)
    book_tokens = tokens.loc[mask]
    book_tokens = book_tokens[_in_line_range(book_tokens["line"], first_line, last_line)]

    lines = book_tokens.groupby("line_id", sort=False).agg(
        line=("line", "first"),
        text=(display_col, " ".join),
    )

    rows = "\n".join(
        f'<div><b>{row["line"]}</b>&nbsp;&nbsp;{row["text"]}</div>'
        for _, row in lines.iterrows()
    )
    style = '<style>.excerpt div { margin-bottom: 0.3em; }</style>'
    return style + f'<div class="excerpt">{rows}</div>'
