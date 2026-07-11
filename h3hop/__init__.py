"""h3hop — distill GLM-5.2 -> Step-3.7-Flash and build a shippable 'bad-weight'.

The package splits into a LIGHT track (stdlib-only, runs anywhere) and a HEAVY track
(torch/trl/repeng/vllm, GPU node). Heavy imports are done lazily inside functions so
this package imports cleanly without the ML stack installed.
"""

__version__ = "0.1.0"
