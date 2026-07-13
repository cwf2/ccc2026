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

import html

import colorcet as cc
import pandas as pd
from matplotlib import pyplot as plt


def build_display_column(tokens, lexicons, lemma_cutoff=0.5, grammar_cutoff=0.7,
                          lemma_color="red", grammar_color="green"):
    '''Wrap each token's text in a colored <span> if its lemma or any of its
    grammatical features (POS + morph) exceeds the given lexicon cutoff.
    Lemma matches take priority over grammar matches for a given token, but
    the span's title attribute always lists every matched feature (lemma and
    grammar both), so a lemma-colored token doesn't hide a grammar match it
    also happened to have.

    tokens   — a token DataFrame with "text", "lemma", "pos", and morph columns
    lexicons — a dict with "lemma" and "grammar" lexicons, e.g. from
               dialogism.build_lexicons()
    '''
    lemma_lex = set(lexicons["lemma"].loc[lexicons["lemma"] > lemma_cutoff].index)
    grammar_lex = set(lexicons["grammar"].loc[lexicons["grammar"] > grammar_cutoff].index)

    morph_cols = ["pos", "verbform", "mood", "tense", "voice", "person", "number", "case", "gender"]
    grammar_hits = tokens[morph_cols].isin(grammar_lex)
    is_grammar_hit = grammar_hits.any(axis=1)
    is_lemma_hit = tokens["lemma"].isin(lemma_lex)

    # restrict the row-wise apply to only the rows that actually have a
    # grammar hit — over the full corpus, running it on every row (rather
    # than just the ~20% that match) was the dominant cost by two orders
    # of magnitude (25s vs 5s)
    matched_grammar = pd.Series([[]] * len(tokens), index=tokens.index, dtype=object)
    matched_grammar.loc[is_grammar_hit] = (
        tokens.loc[is_grammar_hit, morph_cols]
        .where(grammar_hits.loc[is_grammar_hit])
        .apply(lambda row: [f"{col}={val}" for col, val in row.dropna().items()], axis=1)
    )

    def tooltip_for(idx):
        parts = []
        if is_lemma_hit.at[idx]:
            parts.append(f"lemma={tokens.at[idx, 'lemma']}")
        parts.extend(matched_grammar.at[idx])
        return html.escape(", ".join(parts))

    needs_tooltip = is_lemma_hit | is_grammar_hit
    tooltip_col = pd.Series("", index=tokens.index)
    tooltip_col.loc[needs_tooltip] = tokens.index[needs_tooltip].to_series().apply(tooltip_for)

    text = tokens["text"]
    display_col = text.copy()

    grammar_span = f'<span style="color:{grammar_color}" title="' + tooltip_col + '">' + text + "</span>"
    display_col = display_col.where(~is_grammar_hit, grammar_span)

    lemma_span = f'<span style="color:{lemma_color}" title="' + tooltip_col + '">' + text + "</span>"
    display_col = display_col.where(~is_lemma_hit, lemma_span)

    return display_col


_MORPH_COLS = ["pos", "verbform", "mood", "tense", "voice", "person", "number", "case", "gender"]


def top_grammar_features(lexicons, n=15):
    '''The top N grammatical features by corpus-wide lexicon weight, ranked
    once and meant to stay fixed regardless of what line range is in view —
    so a feature checklist built from this doesn't reorder or lose entries
    as the user scrolls/zooms.
    '''
    return list(lexicons["grammar"].sort_values(ascending=False).index[:n])


def feature_palette(features):
    '''Map each feature to a color, in the order given, using colorcet's
    glasbey palette — colors are chosen to greedily maximize their distance
    from every prior color, so the first colors assigned (i.e. the
    highest-ranked features, if `features` comes from top_grammar_features)
    are the most mutually distinctive, and later ones fill in progressively
    finer distinctions. glasbey has 256 colors, far more than any realistic
    top N here.
    '''
    return dict(zip(features, cc.glasbey))


def feature_tally(tokens, features):
    '''Count of tokens matching each feature, within whatever subset of
    tokens is passed in (e.g. the currently-displayed line range) — meant to
    be recomputed per range, unlike top_grammar_features's fixed ranking.
    '''
    counts = {feat: int((tokens[_MORPH_COLS] == feat).any(axis=1).sum()) for feat in features}
    return pd.Series(counts)


def build_feature_display_column(tokens, lexicons, selected_features, feature_colors,
                                  lemma_cutoff=0.5, lemma_color="red"):
    '''Wrap each token's text in a <span> highlighting lexical and/or checked
    grammatical feature matches, meant to be called on a small slice of
    tokens (e.g. one book or a zoomed line range) rather than the whole
    corpus — see build_display_column's docstring for why the full-corpus
    version is comparatively expensive.

    Lexical matches use a foreground color (lemma_color); grammar matches use
    background-color instead, so the two never visually conflict and a token
    can show both at once. A token matching more than one checked feature is
    colored by whichever comes first in selected_features (its lexicon rank),
    but the tooltip lists every match regardless of which one "won" the color.

    selected_features — ordered list of grammar feature values (e.g. from a
        checklist built on top_grammar_features()); first in the list wins
        ties for background color.
    feature_colors — dict mapping each selected feature to a CSS color.
        lemma_color should not appear among these values, since it's
        reserved for lexical matches.
    '''
    lemma_lex = set(lexicons["lemma"].loc[lexicons["lemma"] > lemma_cutoff].index)
    is_lemma_hit = tokens["lemma"].isin(lemma_lex)

    bg_color = pd.Series(None, index=tokens.index, dtype=object)
    matched_features = pd.Series([[]] * len(tokens), index=tokens.index, dtype=object)
    for feat in selected_features:
        hit = (tokens[_MORPH_COLS] == feat).any(axis=1)
        matched_features.loc[hit] = matched_features.loc[hit].apply(lambda fs, feat=feat: fs + [feat])
        still_unset = hit & bg_color.isna()
        bg_color.loc[still_unset] = feature_colors[feat]

    has_bg = bg_color.notna()
    needs_span = has_bg | is_lemma_hit

    def style_and_title(idx):
        style_parts = []
        title_parts = []
        if has_bg.at[idx]:
            style_parts.append(f"background-color:{bg_color.at[idx]}")
        if is_lemma_hit.at[idx]:
            style_parts.append(f"color:{lemma_color}")
            title_parts.append(f"lemma={tokens.at[idx, 'lemma']}")
        title_parts.extend(f"grammar={feat}" for feat in matched_features.at[idx])
        style = html.escape("; ".join(style_parts))
        title = html.escape(", ".join(title_parts))
        return style, title

    text = tokens["text"]
    display_col = text.copy()
    if needs_span.any():
        styled = tokens.index[needs_span].to_series().apply(style_and_title)
        style_col = styled.apply(lambda pair: pair[0])
        title_col = styled.apply(lambda pair: pair[1])
        span = '<span style="' + style_col + '" title="' + title_col + '">' + text.loc[needs_span] + "</span>"
        display_col.loc[needs_span] = span

    return display_col


def in_line_range(line_series, first_line=None, last_line=None):
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


def _line_ticks(book, target_ticks=8):
    '''Choose x-tick positions (token index) and labels (line number) for the
    currently-plotted range, picking a "nice" step so both a whole book and
    a short zoomed-in range end up with a legible, non-overlapping set of
    ticks — a fixed step (e.g. always every 50 lines, as ccc2026.plot_rolling
    does) leaves short ranges with few or no ticks at all.
    '''
    line_num = book["line"].str.extract(r"^(\d+)")[0].astype(int)
    span = max(line_num.max() - line_num.min(), 1)

    candidates = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]
    step = next((c for c in candidates if span / c <= target_ticks), candidates[-1])

    is_tick = (line_num % step == 0)
    ticks = book.loc[is_tick].copy()
    ticks["line_num"] = line_num[is_tick]
    ticks = ticks[~ticks["line_num"].duplicated()]
    return ticks.index, ticks["line_num"].astype(str)


def plot_overlay(tokens, work, pref, pca_roll=None, dialogism_roll=None, first_line=None, last_line=None,
                  min_label_frac=0.05, ylim=None, speech_boundaries=True):
    '''Z-scored overlay of the PCA and/or dialogism rolling scores for one
    book (or a line range within it), plotted in document order so the two
    methods' agreement/divergence can be read against the actual sequence
    of the poem. Speech regions are shaded; a speaker label is added to any
    shaded span wide enough to hold readable text.

    pca_roll, dialogism_roll — rolling score Series indexed like tokens
    (e.g. from ccc2026.rolling_samples(...)["speech_score"]["score"] and
    dialogism.rolling_dialogism(...)["speech_score"]["score"]). Pass None
    for either one to suppress that metric and plot only the other; pass
    None for both to render just the axes, line-number ticks, and speech-
    boundary shading with no data lines — e.g. for building a figure up in
    layers on a slide (axes-only, then +one metric, then +both).

    first_line, last_line — optional; restrict the plotted range to these
    lines. Standardization is always computed over the whole book first, so
    a zoomed-in passage still reads relative to the book as a whole rather
    than being re-centered on itself.

    min_label_frac — only label a shaded speech span with its speaker if the
    span covers at least this fraction of the visible x-range, to avoid
    clutter when many/short speeches are in view at once (e.g. zoomed out
    to a whole book).

    ylim — optional (ymin, ymax) override. With no data, the y-axis would
    otherwise auto-scale to a tight default range around the y=0 reference
    line, which won't match a later call that does have data — pin this to
    the same value across a build-up sequence so the axes don't jump
    between layers.

    speech_boundaries — if False, suppress the gray speech-region shading
    and speaker labels, but keep the line-number x-ticks/xlim exactly as
    they'd be with shading on — so a "signal only" layer and a "signal +
    boundaries" layer share identical axes, differing only in the shading.
    '''
    mask = (tokens["work"] == work) & (tokens["pref"] == pref)
    idx = tokens.index[mask]

    # only include a metric's column (and put it in the dropna subset) when
    # that metric was actually requested — including a None-valued column
    # unconditionally would broadcast to NaN for every row and make
    # dropna(subset=[...]) drop the entire book regardless of the other
    # metric's real data
    data = {
        "line": tokens.loc[idx, "line"],
        "speech_id": tokens.loc[idx, "speech_id"],
        "speaker": tokens.loc[idx, "speaker"],
    }
    dropna_cols = []
    if dialogism_roll is not None:
        data["dialogism"] = dialogism_roll.reindex(idx)
        dropna_cols.append("dialogism")
    if pca_roll is not None:
        data["pca"] = pca_roll.reindex(idx)
        dropna_cols.append("pca")

    book = pd.DataFrame(data)
    if dropna_cols:
        book = book.dropna(subset=dropna_cols)

    # standardize both scores so they're comparable on one axis — using the
    # whole book's stats, before any line-range restriction
    if dialogism_roll is not None:
        book["dialogism_z"] = (book["dialogism"] - book["dialogism"].mean()) / book["dialogism"].std()
    if pca_roll is not None:
        book["pca_z"] = (book["pca"] - book["pca"].mean()) / book["pca"].std()

    book = book[in_line_range(book["line"], first_line, last_line)]

    fig, ax = plt.subplots(figsize=(9, 4))
    if dialogism_roll is not None:
        ax.plot(book.index, book["dialogism_z"], label="weighted log-odds dialogism (standardized)")
    if pca_roll is not None:
        ax.plot(book.index, book["pca_z"], label="PCA + logistic regression (standardized)")
    ax.axhline(0, color="k", ls="--", lw=1)
    title = f"{work} {pref}"
    if first_line is not None or last_line is not None:
        title += f" ({first_line if first_line is not None else 'start'}-{last_line if last_line is not None else 'end'})"
    ax.set_title(title)
    ax.set_xlabel("line")
    if dialogism_roll is not None or pca_roll is not None:
        ax.legend()
    if ylim is not None:
        ax.set_ylim(ylim)

    # line-number x-ticks always apply, regardless of speech_boundaries —
    # only the gray shading/speaker labels below are optional
    if len(book):
        xmin, xmax = book.index.min(), book.index.max()
        total_span = max(xmax - xmin, 1)
        ymax = ax.get_ylim()[1]

        tick_positions, tick_labels = _line_ticks(book)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(xmin, xmax)

        # shade speech regions, and label wide-enough spans with their speaker
        if speech_boundaries:
            speech_data = book.dropna(subset=["speech_id"])
            for sid, group in speech_data.groupby("speech_id"):
                lo, hi = group.index.min(), group.index.max()
                ax.axvspan(lo, hi, alpha=0.15, color="gray", linewidth=0)

                speaker = group["speaker"].iloc[0]
                if pd.notna(speaker) and (hi - lo) / total_span >= min_label_frac:
                    ax.text((lo + hi) / 2, ymax * 0.92, speaker,
                             ha="center", va="top", fontsize=8, clip_on=True)

    return fig


def _excerpt_row_html(row):
    classes = "line"
    meta_text = ""
    if pd.notna(row["speech_id"]):
        classes += " speech"
        if pd.notna(row["speaker"]):
            meta_text = row["speaker"]
            if pd.notna(row["addressee"]):
                meta_text += f" &rarr; {row['addressee']}"

    # always emit the meta column (blank on narration rows) so the fixed
    # width reserves the same space either way, keeping verse text flush
    # to one left margin regardless of whether a line has an annotation
    meta = f'<span class="meta">{meta_text}</span>'
    return (
        f'<div class="{classes}">{meta}'
        f'<b class="locus">{row["line"]}</b>'
        f'<span class="text">{row["text"]}</span></div>'
    )


def highlighted_excerpt(tokens, work, pref, first_line=None, last_line=None, display_col="display"):
    '''Build an HTML excerpt of a book (or a line range within it), one row
    per verse line with its highlighted text, speaker/addressee (if any),
    and light shading on speech lines — matching plot_overlay's shaded
    speech regions.

    Requires tokens[display_col] to already be populated, e.g. via
    build_display_column() — highlighting is corpus-wide and relatively
    expensive, so it's meant to be computed once and then sliced by book/
    range repeatedly, not recomputed per excerpt.
    '''
    mask = (tokens["work"] == work) & (tokens["pref"] == pref)
    book_tokens = tokens.loc[mask]
    book_tokens = book_tokens[in_line_range(book_tokens["line"], first_line, last_line)]

    lines = book_tokens.groupby("line_id", sort=False).agg(
        line=("line", "first"),
        speaker=("speaker", "first"),
        addressee=("addressee", "first"),
        speech_id=("speech_id", "first"),
        text=(display_col, " ".join),
    )

    rows = "\n".join(_excerpt_row_html(row) for _, row in lines.iterrows())
    style = '''<style>
    .excerpt div.line {
        display: flex; align-items: baseline;
        margin-bottom: 0.3em; padding: 0.1em 0.3em;
    }
    .excerpt div.line.speech { background-color: rgba(128, 128, 128, 0.15); }
    .excerpt .meta {
        flex: 0 0 12em; text-align: right; padding-right: 0.75em;
        color: #666; font-size: 0.85em;
    }
    .excerpt .locus { flex: 0 0 auto; margin-right: 0.5em; }
    .excerpt .text { flex: 1 1 auto; }
    </style>'''
    return style + f'<div class="excerpt">{rows}</div>'
