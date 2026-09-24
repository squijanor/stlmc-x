"""Custom counterexample generation strategies.

Two strategies -- discrete path enumeration (`pathenum`) and region box
discovery (`box`) -- built on shared generation infrastructure: the STLMC
falsification encoding (`generation.encode`), an incremental solver
(`generation.oracle`), the driver-contract helpers in `generation.common`
(configuration reading and verdict scoping), the linear word feasibility filter
(`generation.feasibility`) and the reduced-query pivot reconstruction
(`generation.reduced`). Neither strategy imports the other. Selected by the
[common] ``generation`` config value and dispatched here.
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
            "generation strategy 'compose' is not implemented yet"
        )
    raise ValueError(f"unknown generation strategy: {kind!r}")
