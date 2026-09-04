"""Zixi.Reasoning — a minimal reflective cognitive state machine.

Six primitives, three operations, two timescales, no database.

    [FACT] [STATE] [REASONING] [REFLECT]  — cognition (truth zone)
    [ASSUME] [LAB]                        — hypothesis zone (unverified)
    [[WikiLink]]                          — association
    ->[STATE]                             — cognition (changes active state)
    =>[[Node]]                            — learning (submits crystallization)

    ACTIVE.md                             — fast / active memory (snapshot)
    memory/*.md                           — slow / crystallized memory (nodes)

The model does not change. The past changes the future.

    M_{t+1} = C(M_t, Reflect(E_t, S_t))
"""

__version__ = "0.2.0"
