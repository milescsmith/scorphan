from pathlib import Path

import muon as mu
import stackprinter

import scorphan as so

stackprinter.set_excepthook()

mdata = mu.read(Path().home().joinpath("workspace", "scorphan", "tests", "5k_pbmc.h5mu"))
so.pp.neighbors(mdata, modality_weights={"rna":0.4, "prot":0.6}, key_added="wnn", method="umap", verbose=True)
