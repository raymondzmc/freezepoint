"""The loop: generate -> verify -> refine. Every candidate goes through harness.ledger before it is scored;
every data access goes through harness.pit. Arms: frozen, refine, bestofn (see configs/)."""
