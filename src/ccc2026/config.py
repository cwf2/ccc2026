'''config.py - global values'''

CONFIG = {

    # local directory for working data
    "data_dir": "data",

    # local directory for figure output
    "plot_dir": "plot",

    # texts to analyze, remote token files
    "texts": [
        ("Iliad", "tlg0012.tlg001.perseus-grc2.csv"),
        ("Odyssey", "tlg0012.tlg002.perseus-grc2.csv"),
        ("Argonautica", "tlg0001.tlg001.perseus-grc2.csv"),
        ("Posthomerica", "tlg2046.tlg001.perseus-grc2.csv"),
        ("Sack of Troy", "tlg0647.tlg001.perseus-grc2.csv"),
        ("Dionysiaca", "tlg2045.tlg001.perseus-grc2.csv"),
    ],
}