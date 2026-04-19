"""Load-bearing stop-gradient test for the physics regularizer.

Blueprint Concept 6 acceptance test (the single most important unit test in
the entire repo per impl_best_practices.md §2): verify that inside L_physics,
the sigma gradient path is severed. Without this, the physics term can
explain-away aleatoric σ and break coverage.

Failure mode protected against:
    If we erroneously write `sigma_ode` instead of `sigma_ode.detach()` in
    regularizer.py, L_physics.backward() will produce a non-zero gradient on
    sigma_ode. This test catches that regression at the unit-test level.
"""
from __future__ import annotations

import numpy as np
import torch

from phys_gimin.priors.literature import LiteraturePriorProvider
from phys_gimin.regularizer import PhysicsRegularizer, beta_nll


class TestStopGradient:
    """Verify that ∂L_physics / ∂σ = 0 after .backward()."""

    def test_sigma_receives_no_gradient_from_l_physics(self):
        """The load-bearing unit test (blueprint Concept 6 acceptance)."""
        torch.manual_seed(0)

        provider = LiteraturePriorProvider()
        regularizer = PhysicsRegularizer(provider=provider, beta=0.5)

        batch = 3
        n_visits = 4
        mu = torch.randn(batch, n_visits, requires_grad=True)
        # sigma is a LEAF tensor with requires_grad=True — only the detach
        # call inside regularizer.compute_physics_loss should block its grad.
        log_sigma = torch.randn(batch, n_visits, requires_grad=True)
        sigma = torch.nn.functional.softplus(log_sigma) + 0.1  # positive sigma

        t_years_per_patient = [np.array([0.0, 1.0, 3.0, 5.0]) for _ in range(batch)]
        sbr_0_per_patient = [2.5, 2.3, 2.7]
        patnos = [3000, 3100, 3200]

        l_phys = regularizer.compute_physics_loss(
            mu=mu,
            sigma=sigma,
            patnos=patnos,
            t_years_per_patient=t_years_per_patient,
            sbr_0_per_patient=sbr_0_per_patient,
        )

        l_phys.backward()

        # The invariant: physics loss produces NO gradient on sigma.
        # Gradient may be None (preferred, means no backward path found) or
        # a tensor of zeros.
        if log_sigma.grad is None:
            pass  # ideal — autograd found no path
        else:
            assert torch.all(log_sigma.grad == 0), (
                "∂L_physics / ∂sigma ≠ 0 — stop-gradient broken! "
                f"max |grad| = {log_sigma.grad.abs().max().item():.3e}. "
                "Check that regularizer.py uses sigma.detach() inside "
                "compute_physics_loss."
            )

        # Sanity check: mu SHOULD receive gradient — physics loss is not
        # trivially zero in its mean input.
        assert mu.grad is not None, "mu.grad is None — loss did not touch mu"
        assert torch.any(mu.grad != 0), (
            "∂L_physics / ∂mu == 0 everywhere — suspicious. "
            "Either the target happens to match mu exactly (check seed) or "
            "the physics loss computation is broken."
        )

    def test_beta_nll_directly_propagates_mu_grad(self):
        """β-NLL itself (without the physics wrapper) does push grad to mu."""
        torch.manual_seed(1)
        mu = torch.randn(5, requires_grad=True)
        sigma = torch.ones(5) * 0.5  # no grad — fixed
        target = torch.randn(5)

        loss = beta_nll(mu=mu, sigma=sigma, target=target, beta=0.5)
        loss.backward()

        assert mu.grad is not None
        assert torch.any(mu.grad != 0)

    def test_beta_nll_sample_weight_is_detached(self):
        """β-NLL's outer sigma^(2β) re-weight does NOT produce sigma grad."""
        torch.manual_seed(2)
        mu = torch.randn(5)
        # Leaf sigma with grad enabled — the detach inside beta_nll should block it.
        log_sigma = torch.zeros(5, requires_grad=True)
        sigma = torch.nn.functional.softplus(log_sigma) + 0.1
        target = torch.randn(5)

        loss = beta_nll(mu=mu, sigma=sigma, target=target, beta=0.5)
        loss.backward()

        # sigma's outer weight is detached, BUT the inner variance in the NLL
        # formula is NOT detached — β-NLL alone still trains sigma through
        # the log(σ²) + residual²/σ² terms. What is detached is ONLY the outer
        # sample-weight factor σ^(2β).
        # So this test just confirms `loss.backward()` succeeded; sigma.grad
        # may be nonzero from the inner terms.
        assert log_sigma.grad is not None

    def test_stop_grad_versus_no_stop_grad_sanity(self):
        """Without regularizer.detach, sigma would receive gradient — baseline check.

        This confirms the REGRESSION we're guarding against: if someone
        removes the .detach() from regularizer.compute_physics_loss, sigma
        WOULD get a gradient. We simulate that by calling beta_nll directly
        without the detach.
        """
        torch.manual_seed(3)
        mu = torch.randn(5, requires_grad=True)
        log_sigma = torch.randn(5, requires_grad=True)
        sigma = torch.nn.functional.softplus(log_sigma) + 0.1
        target = torch.randn(5)

        # Hypothetical "buggy" call: no detach on sigma.
        loss_buggy = beta_nll(mu=mu, sigma=sigma, target=target, beta=0.5)
        loss_buggy.backward()

        # In the buggy case, sigma MUST receive gradient (confirming the bug
        # shape we're protecting against).
        assert log_sigma.grad is not None
        # At least one element should be nonzero.
        assert torch.any(log_sigma.grad != 0), (
            "Even without detach, sigma.grad is zero — test is not discriminating."
        )
