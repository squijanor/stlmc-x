"""Custom counterexample generation strategies.

Standalone algorithms that reuse only the STLMC falsification encoding
(`generation.encode`) and an incremental solver (`generation.oracle`). Selected
by the [common] ``generation`` config value and dispatched here.
"""


def make_generation_algorithm(kind: str, config):
    """Return the generation Algorithm for a ``generation`` config value.

    ``config`` is accepted for strategies that need it at construction; the
    current strategies read their parameters in ``run``.
    """
    if kind == "pathenum":
        from .pathenum import DiscretePathEnum

        return DiscretePathEnum()
    if kind in ("box", "compose"):
        raise NotImplementedError("generation strategy '{}' is not implemented yet".format(kind))
    raise ValueError("unknown generation strategy: {!r}".format(kind))