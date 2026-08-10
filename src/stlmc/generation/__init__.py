"""Custom counterexample generation strategies.

Standalone algorithms that reuse only the STLMC falsification encoding
(`generation.encode`), an incremental solver (`generation.oracle`) and the
driver-contract helpers in `generation.common` -- configuration reading and
verdict scoping. They share no mechanism and do not import one another. Selected
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
    if kind == "box":
        from .box import RegionBoxDiscovery

        return RegionBoxDiscovery()
    if kind == "compose":
        raise NotImplementedError(
            "generation strategy 'compose' is not implemented yet")
    raise ValueError(f"unknown generation strategy: {kind!r}")