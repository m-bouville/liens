"""
Single source of truth for architectural constants shared across
models/, training/, evaluation/, and tests/.

LATENT_SPATIAL_SIZE previously existed as 8 INDEPENDENT hardcoded
copies of the literal 8 (see the project's own magic-number audit,
this file's whole reason for existing): encoder.py's depth calculation,
decoder.py's depth calculation AND its separate runtime shape check,
latent_dynamics.py's and stats_head.py's default constructor
parameters, training/datasets.py's augmentation phase calculation,
evaluation/check_parameter_dependence.py's reference-line calculation,
and tests/conftest.py's FakeEncoder fixture -- silently correct only as
long as nobody ever needed to change it, with a shape-mismatch (not a
clear error) as the failure mode for any one missed on an update.

This is now a DEFAULT, not a hardcoded assumption: the actual bottleneck
size for any given model is a real, configurable value (see Encoder/
Decoder's own latent_spatial_size constructor parameter, and the
'latent_size' stage-parameters-file key, both defaulting to this
constant when not specified) -- LATENT_SPATIAL_SIZE is what "not
specified" falls back to, not a ceiling on what's possible.
"""

LATENT_SPATIAL_SIZE = 8
