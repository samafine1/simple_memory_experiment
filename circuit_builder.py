from __future__ import annotations

from typing import Dict
import numpy as np
import stim


def _row_support(row) -> np.ndarray:
    """Indices of 1s in a sparse or dense row."""
    return row.indices if hasattr(row, "indices") else np.where(np.asarray(row).ravel() == 1)[0]


def build_memory_circuit(inner, *, rounds, p_data, p_meas, basis) -> stim.Circuit:
    """Phenomenological memory experiment for a CSS code.
    Warning: A bare ancilla is used for syndrome extraction,
    which is distance preserving for HGP codes, but not generally for CSS codes.
    """
    basis = basis.upper()
    checks = inner.hz if basis == "Z" else inner.hx
    lx = inner.lx
    lz = inner.lz
    n, m, k = checks.shape[1], checks.shape[0], lz.shape[0]

    c = stim.Circuit()
    c.append("R", range(n + m))
    if basis == "X":
        c.append("H", range(n))
    c.append("TICK")

    meas_count = 0
    last_meas: Dict[int, int] = {}

    for _ in range(rounds):
        if p_data > 0:
            c.append("DEPOLARIZE1", range(n), p_data)
        c.append("TICK")
        for i in range(m):
            anc = n + i
            c.append("R", [anc])
            if basis == "X":
                c.append("H", [anc])
            c.append("TICK")
            for q in _row_support(checks[i]):
                c.append("CX", [q, anc] if basis == "Z" else [anc, q])
            if basis == "X":
                c.append("H", [anc])
            if p_meas > 0:
                c.append("X_ERROR", [anc], p_meas)
            c.append("M", [anc])
            prev = last_meas.get(i)
            if prev is None:
                c.append("DETECTOR", [stim.target_rec(-1)])
            else:
                c.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-(meas_count - prev + 1))])
            last_meas[i] = meas_count
            meas_count += 1
            c.append("TICK")

    if basis == "X":
        c.append("H", range(n))
    c.append("M", range(n))
    for i in range(m):
        data_rec_targets = [stim.target_rec(-(n - q)) for q in _row_support(checks[i])]
        anc_rec_target = stim.target_rec(-(n + meas_count - last_meas[i]))
        c.append("DETECTOR", [anc_rec_target] + data_rec_targets)

    for j in range(k):
        op = lz[j] if basis == "Z" else lx[j]
        c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-(n - q)) for q in _row_support(op)], j)
    return c