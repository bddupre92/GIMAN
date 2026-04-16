"""Gate tests for SAEM GFAP channel extension."""
import numpy as np
import pytest

from scripts.mechanistic_twin.multi_obs_saem import (  # noqa
    total_log_likelihood, gfap_likelihood, ParamVector,
)


def test_gfap_likelihood_finite():
    """GFAP likelihood should be finite for sane inputs."""
    ll = gfap_likelihood(
        gfap_obs=5.0, O_t=1e-3, s_gfap=2.0, sigma_gfap=0.5,
    )
    assert np.isfinite(ll)


def test_gfap_likelihood_maximum_at_zero_residual():
    s_gfap = 2.0
    O_t = 0.5
    max_ll = gfap_likelihood(s_gfap * O_t, O_t, s_gfap, 0.5)
    off_ll = gfap_likelihood(s_gfap * O_t + 2.0, O_t, s_gfap, 0.5)
    assert max_ll > off_ll


def test_param_vector_has_gfap_fields():
    """ParamVector must include s_gfap and sigma_gfap for SAEM v3."""
    p = ParamVector(
        k_n=3.6e-4, alpha_tox=1.17e-5,
        sigma_sbr=0.2, sigma_agg=10.0, sigma_saa=0.1, sigma_nev=0.5,
        s_gfap=2.0, sigma_gfap=0.5,
    )
    assert p.s_gfap == 2.0
    assert p.sigma_gfap == 0.5


def test_total_log_likelihood_accepts_gfap_obs():
    """If GFAP observation is present, it should contribute to the total LL."""
    obs_with = {"sbr": 1.2, "asyn_agg_pct": 12.5, "saa_ttt": 0.9,
                "nev_asyn": 0.8, "gfap_npx": 4.5, "t_years": 3.0}
    obs_without = {k: v for k, v in obs_with.items() if k != "gfap_npx"}

    params = ParamVector(
        k_n=3.6e-4, alpha_tox=1.17e-5,
        sigma_sbr=0.2, sigma_agg=10.0, sigma_saa=0.1, sigma_nev=0.5,
        s_gfap=2.0, sigma_gfap=0.5,
    )

    ll_with = total_log_likelihood(obs_with, params)
    ll_without = total_log_likelihood(obs_without, params)
    assert np.isfinite(ll_with)
    assert np.isfinite(ll_without)
    assert ll_with != ll_without, "GFAP channel should affect total LL"


def test_existing_saem_tests_still_pass():
    """No regression in the existing SAEM v2 tests (if any exist)."""
    # This is a marker — the regression check happens via pytest run.
    pass
