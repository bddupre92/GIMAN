"""
derooij_bridge.jl — thin re-export layer over de Rooij's ude.jl regularizers.

PURPOSE
-------
De Rooij et al. 2025 define two physiology-informed regularizers inline inside the
`get_ude_loss()` closure in `minimal-model/ude.jl`:

    nonnegatives = sum(abs2, min.(0, ra)) * λ_nonneg
    auc_regularizer = abs(trapz(0:480, ra) - 1.) * λ_AUC

This bridge file re-implements those two regularizer expressions as named,
callable top-level functions, enabling juliacall to invoke them from Python
without modifying the vendored ude.jl (hard constraint).

FIDELITY
--------
The functions below are LITERAL EXTRACTIONS of the regularizer expressions from
ude.jl get_ude_loss(), parameterised over the prediction array.  The only change
is that λ is moved to the Python side (we pass pre-multiplied strengths or apply
λ=1 and scale in Python) so the functions return the raw penalty before λ scaling.

Reference (do not modify the vendored file):
  paper12_phys_gimin/baselines/derooij_2025/minimal-model/ude.jl
  Upstream SHA: e460ee00150a1b82d1cc0e5301c9839f662898e4

Usage from Python via juliacall
--------------------------------
    from juliacall import Main as jl
    bridge_path = Path("...") / "derooij_bridge.jl"
    bridge_env  = Path("...") / "julia_bridge_env"
    jl.seval(f'import Pkg; Pkg.activate("{bridge_env}")')
    jl.include(str(bridge_path))
    nonneg = float(jl.derooij_nonneg_loss(predictions_array))
    auc    = float(jl.derooij_auc_loss(predictions_array, times_array))
"""

# Activate the minimal bridge environment (Trapz only).
# The caller should activate BEFORE including this file; we guard here so
# the file is also safe to include standalone.
import Pkg
using Trapz

"""
    derooij_nonneg_loss(ra::AbstractArray) -> Float64

Non-negativity regularizer from de Rooij et al. 2025, ude.jl:

    nonnegatives = sum(abs2, min.(0, ra))

Returns the UNSCALED penalty (λ_nonneg = 1). Caller multiplies by λ.

Physiology: `ra` represents a rate (meal-appearance or learned NN output) that
must be non-negative. Any negative value is penalised quadratically.
"""
function derooij_nonneg_loss(ra::AbstractArray)::Float64
    return sum(abs2, min.(0.0, ra))
end


"""
    derooij_auc_loss(ra::AbstractArray, times::AbstractArray) -> Float64

AUC regularizer from de Rooij et al. 2025, ude.jl:

    auc_regularizer = abs(trapz(0:480, ra) - 1.)

In the original code `times` is always `0:480` (480-minute glucose experiment).
For the Python bridge we accept `times` as an argument so the function is general,
enabling use with arbitrary time grids (e.g. PPMI visit intervals).

Returns the UNSCALED penalty (λ_AUC = 1). Caller multiplies by λ.

Requires: Trapz.jl (loaded via the julia_bridge_env environment).
"""
function derooij_auc_loss(ra::AbstractArray, times::AbstractArray)::Float64
    return abs(trapz(times, ra) - 1.0)
end
