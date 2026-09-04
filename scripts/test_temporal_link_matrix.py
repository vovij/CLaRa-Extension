"""
Test suite for the Temporal Link Matrix (TLM) implementation
(temporal_link_matrix.py).

This suite exercises TemporalLinkMatrix, multi_hop_propagate, and
order_by_hop -- the differentiable temporal-link multi-hop retrieval
extension to CLaRa described in Chapter tlm of the report. It is organised
around, and verifies one-to-one, the five claims made in the report's
"Correctness tests" paragraph (Section tlm-impl):

    "The module's standalone test suite verifies: shape correctness of L;
    row-stochasticity on valid rows and all-zero rows for padded candidates;
    finiteness of the propagated scores in the presence of -inf-masked padding;
    compatibility of the propagated scores with the straight-through top-k; and
    that gradients from a downstream loss reach M, q, and the link-module
    parameters, confirming the end-to-end differentiability claim."

Test classes, each covering one claim:
    TestShapeCorrectness          -> shape correctness of L
    TestRowStochasticity          -> row-stochasticity on valid rows and
                                      all-zero rows for padded candidates
    TestPropagationFiniteness     -> finiteness of the propagated scores in
                                      the presence of -inf-masked padding
    TestTopKCompatibility         -> compatibility of the propagated scores
                                      with the straight-through top-k
    TestEndToEndDifferentiability -> gradients from a downstream loss reach
                                      M, q, and the link-module parameters

Known gap: the "all-zero rows for padded candidates" half of the
row-stochasticity claim is not currently covered. The module's forward()
only masks the destination column for padded candidates, not the source
row, so a padded candidate's outgoing row is actually a uniform
distribution over valid candidates (row sum == 1), not all-zero as the
docstring and write-up claim.

Run with:
    pytest test_temporal_link_matrix.py -v
"""
import pytest
import torch
import torch.nn.functional as F

from temporal_link_matrix import (
    TemporalLinkMatrix,
    multi_hop_propagate,
    order_by_hop,
)


# Shared fixtures

@pytest.fixture
def seeded():
    torch.manual_seed(0)


@pytest.fixture
def small_pool(seeded):
    """A small candidate pool with a partially padded batch element,
    matching the shapes used in temporal_link_matrix.py's own __main__ block."""
    B, D, E, rank = 2, 100, 512, 64
    M = torch.randn(B, D, E, requires_grad=True)
    q = torch.randn(B, E, requires_grad=True)
    cand_mask = torch.ones(B, D, dtype=torch.bool)
    cand_mask[0, 80:] = False  # batch element 0 has 20 padded candidates
    link_module = TemporalLinkMatrix(hidden_dim=E, rank=rank)
    return {
        "B": B, "D": D, "E": E, "rank": rank,
        "M": M, "q": q, "cand_mask": cand_mask, "link_module": link_module,
    }


def differentiable_topk(logits, k, temperature=1.0):
    """Straight-through top-k, identical to the reference implementation used
    in temporal_link_matrix.py's own __main__ self-test and in
    modeling_clara.py's _forward_stage2_batch."""
    Bb, N = logits.shape
    perturbed = logits / max(temperature, 1e-6)
    _, topk_idx = perturbed.topk(k, dim=-1)
    K_hard = torch.zeros(Bb, k, N, device=logits.device, dtype=logits.dtype)
    K_hard.scatter_(2, topk_idx.unsqueeze(-1), 1.0)
    K_soft = torch.zeros_like(K_hard)
    taken = torch.zeros(Bb, N, device=logits.device, dtype=logits.dtype)
    for j in range(k):
        mask = (1.0 - taken.detach())
        masked = perturbed + (mask + 1e-8).log()
        pj = F.softmax(masked, dim=-1).float()
        K_soft[:, j, :] = pj
        taken = torch.clamp(taken + K_hard[:, j, :], max=1.0)
    W = K_hard + (K_soft - K_soft.detach())
    return W, topk_idx


# Claim 1: shape correctness of L

class TestShapeCorrectness:

    def test_output_shape_is_B_D_D(self, small_pool):
        link_module = small_pool["link_module"]
        L = link_module(small_pool["M"], small_pool["q"], small_pool["cand_mask"])
        B, D = small_pool["B"], small_pool["D"]
        assert L.shape == (B, D, D)

    @pytest.mark.parametrize("B,D,E,rank", [(1, 10, 32, 8), (3, 50, 128, 16), (4, 1, 16, 4)])
    def test_shape_scales_with_batch_and_pool_size(self, seeded, B, D, E, rank):
        M = torch.randn(B, D, E)
        q = torch.randn(B, E)
        cand_mask = torch.ones(B, D, dtype=torch.bool)
        link_module = TemporalLinkMatrix(hidden_dim=E, rank=rank)
        L = link_module(M, q, cand_mask)
        assert L.shape == (B, D, D)

    def test_hidden_dim_mismatch_raises(self, seeded):
        """The module asserts E must match hidden_dim -- this guards against
        the exact wiring bug flagged when comparing the write-up's E=4096
        worked example against modeling_clara.py's flattened_doc_dim."""
        link_module = TemporalLinkMatrix(hidden_dim=512, rank=64)
        wrong_E_M = torch.randn(2, 10, 256)  # E=256, module expects 512
        q = torch.randn(2, 256)
        cand_mask = torch.ones(2, 10, dtype=torch.bool)
        with pytest.raises(AssertionError):
            link_module(wrong_E_M, q, cand_mask)


# Claim 2: row-stochasticity on valid rows and all-zero rows for padded candidates

class TestRowStochasticity:

    def test_valid_rows_sum_to_one(self, small_pool):
        link_module = small_pool["link_module"]
        L = link_module(small_pool["M"], small_pool["q"], small_pool["cand_mask"])
        row_sums = L.sum(-1)
        assert torch.allclose(
            row_sums[small_pool["cand_mask"]],
            torch.ones_like(row_sums[small_pool["cand_mask"]]),
            atol=1e-4,
        )

    def test_no_nans_in_output(self, small_pool):
        link_module = small_pool["link_module"]
        L = link_module(small_pool["M"], small_pool["q"], small_pool["cand_mask"])
        assert not torch.isnan(L).any()

    def test_self_loops_receive_zero_mass(self, small_pool):
        """Eq. tlm-softmax masks L_ii to -inf before the softmax, so the
        diagonal should be exactly zero for every valid candidate."""
        link_module = small_pool["link_module"]
        L = link_module(small_pool["M"], small_pool["q"], small_pool["cand_mask"])
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)

    def test_fully_valid_pool_still_row_stochastic(self, seeded):
        """No padding at all -- every row across every batch element must
        sum to one."""
        B, D, E, rank = 3, 20, 64, 8
        M = torch.randn(B, D, E)
        q = torch.randn(B, E)
        cand_mask = torch.ones(B, D, dtype=torch.bool)
        link_module = TemporalLinkMatrix(hidden_dim=E, rank=rank)
        L = link_module(M, q, cand_mask)
        row_sums = L.sum(-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


# Claim 3: finiteness of the propagated scores in the presence of -inf-masked padding

class TestPropagationFiniteness:

    def test_propagated_scores_are_finite_with_padding(self, small_pool):
        link_module = small_pool["link_module"]
        M, q, cand_mask = small_pool["M"], small_pool["q"], small_pool["cand_mask"]
        L = link_module(M, q, cand_mask)

        s0 = F.cosine_similarity(q.unsqueeze(1), M, dim=-1)
        s0 = s0.masked_fill(~cand_mask, float("-inf"))
        assert torch.isinf(s0).any(), "test setup should actually include -inf entries"

        s_final = multi_hop_propagate(s0, L, hops=2, damping=0.5)
        assert torch.isfinite(s_final).all()

    def test_propagated_scores_finite_across_hop_counts(self, small_pool):
        link_module = small_pool["link_module"]
        M, q, cand_mask = small_pool["M"], small_pool["q"], small_pool["cand_mask"]
        L = link_module(M, q, cand_mask)
        s0 = F.cosine_similarity(q.unsqueeze(1), M, dim=-1)
        s0 = s0.masked_fill(~cand_mask, float("-inf"))
        for hops in (0, 1, 2, 4, 8):
            s_final = multi_hop_propagate(s0, L, hops=hops, damping=0.5)
            assert torch.isfinite(s_final).all(), f"failed at hops={hops}"

    def test_propagated_scores_finite_across_damping_values(self, small_pool):
        link_module = small_pool["link_module"]
        M, q, cand_mask = small_pool["M"], small_pool["q"], small_pool["cand_mask"]
        L = link_module(M, q, cand_mask)
        s0 = F.cosine_similarity(q.unsqueeze(1), M, dim=-1)
        s0 = s0.masked_fill(~cand_mask, float("-inf"))
        for damping in (0.0, 0.5, 1.0):
            s_final = multi_hop_propagate(s0, L, hops=2, damping=damping)
            assert torch.isfinite(s_final).all(), f"failed at damping={damping}"

    def test_damping_zero_recovers_base_scores(self, small_pool):
        """a=0 should recover the unmodified (no-KG) pipeline exactly, per
        the write-up's Step 3 claim."""
        link_module = small_pool["link_module"]
        M, q, cand_mask = small_pool["M"], small_pool["q"], small_pool["cand_mask"]
        L = link_module(M, q, cand_mask)
        s0 = F.cosine_similarity(q.unsqueeze(1), M, dim=-1)
        s0 = s0.masked_fill(~cand_mask, float("-inf"))
        s0_soft = F.softmax(s0.float(), dim=-1)
        s_final = multi_hop_propagate(s0, L, hops=3, damping=0.0)
        assert torch.allclose(s_final, s0_soft, atol=1e-5)

    def test_bf16_autocast_does_not_break_propagation(self, small_pool):
        """Regression test for the fp32-under-autocast bug documented in
        multi_hop_propagate's comments: torch.einsum silently downcasts under
        bf16 autocast, so casting must happen on every use inside the loop."""
        if not torch.cuda.is_available():
            pytest.skip("bf16 autocast behavior is only exercised meaningfully on CUDA")
        link_module = small_pool["link_module"]
        M, q, cand_mask = small_pool["M"], small_pool["q"], small_pool["cand_mask"]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            L = link_module(M, q, cand_mask)
            s0 = F.cosine_similarity(q.unsqueeze(1), M, dim=-1)
            s0 = s0.masked_fill(~cand_mask, float("-inf"))
            s_final = multi_hop_propagate(s0, L, hops=2, damping=0.5)
        assert torch.isfinite(s_final).all()


# Claim 4: compatibility of the propagated scores with the straight-through top-k

class TestTopKCompatibility:

    def test_topk_runs_on_propagated_scores(self, small_pool):
        link_module = small_pool["link_module"]
        M, q, cand_mask = small_pool["M"], small_pool["q"], small_pool["cand_mask"]
        L = link_module(M, q, cand_mask)
        s0 = F.cosine_similarity(q.unsqueeze(1), M, dim=-1)
        s0 = s0.masked_fill(~cand_mask, float("-inf"))
        s_final = multi_hop_propagate(s0, L, hops=2, damping=0.5)

        k = 20
        Z, topk_idx = differentiable_topk(torch.nan_to_num(s_final, nan=float("-inf")), k=k)
        assert Z.shape == (small_pool["B"], k, small_pool["D"])
        assert topk_idx.shape == (small_pool["B"], k)

    def test_topk_selection_produces_valid_embeddings(self, small_pool):
        link_module = small_pool["link_module"]
        M, q, cand_mask = small_pool["M"], small_pool["q"], small_pool["cand_mask"]
        L = link_module(M, q, cand_mask)
        s0 = F.cosine_similarity(q.unsqueeze(1), M, dim=-1)
        s0 = s0.masked_fill(~cand_mask, float("-inf"))
        s_final = multi_hop_propagate(s0, L, hops=2, damping=0.5)

        k = 20
        Z, _ = differentiable_topk(torch.nan_to_num(s_final, nan=float("-inf")), k=k)
        M_selected = torch.einsum("bkd,bde->bke", Z, M)
        assert M_selected.shape == (small_pool["B"], k, small_pool["E"])
        assert torch.isfinite(M_selected).all()

    def test_topk_never_selects_fully_padded_candidates(self, small_pool):
        """Padded candidates carry s_final ~ 0 after nan_to_num, so with a
        small enough k relative to the number of valid candidates, top-k
        should not select them for the padded batch element."""
        link_module = small_pool["link_module"]
        M, q, cand_mask = small_pool["M"], small_pool["q"], small_pool["cand_mask"]
        L = link_module(M, q, cand_mask)
        s0 = F.cosine_similarity(q.unsqueeze(1), M, dim=-1)
        s0 = s0.masked_fill(~cand_mask, float("-inf"))
        s_final = multi_hop_propagate(s0, L, hops=2, damping=0.5)

        k = 5  # well below the 80 valid candidates in batch element 0
        _, topk_idx = differentiable_topk(torch.nan_to_num(s_final, nan=float("-inf")), k=k)
        padded_indices = set(range(80, 100))
        selected_batch0 = set(topk_idx[0].tolist())
        assert selected_batch0.isdisjoint(padded_indices)

    def test_order_by_hop_reorders_within_selected_set(self, small_pool):
        """Step 5: order_by_hop should permute the k selected indices without
        changing which indices were selected."""
        link_module = small_pool["link_module"]
        M, q, cand_mask = small_pool["M"], small_pool["q"], small_pool["cand_mask"]
        L = link_module(M, q, cand_mask)
        s0 = F.cosine_similarity(q.unsqueeze(1), M, dim=-1)
        s0 = s0.masked_fill(~cand_mask, float("-inf"))
        s_final = multi_hop_propagate(s0, L, hops=2, damping=0.5)

        k = 20
        _, topk_idx = differentiable_topk(torch.nan_to_num(s_final, nan=float("-inf")), k=k)
        reordered_idx, perm = order_by_hop(topk_idx, L, s0, hops=2, damping=0.5)

        assert reordered_idx.shape == topk_idx.shape
        assert perm.shape == topk_idx.shape
        for b in range(small_pool["B"]):
            assert set(reordered_idx[b].tolist()) == set(topk_idx[b].tolist())


# Claim 5: gradients from a downstream loss reach M, q, and the link-module parameters

class TestEndToEndDifferentiability:

    def test_gradients_reach_M_q_and_link_module(self, small_pool):
        link_module = small_pool["link_module"]
        M, q, cand_mask = small_pool["M"], small_pool["q"], small_pool["cand_mask"]
        L = link_module(M, q, cand_mask)
        s0 = F.cosine_similarity(q.unsqueeze(1), M, dim=-1)
        s0 = s0.masked_fill(~cand_mask, float("-inf"))
        s_final = multi_hop_propagate(s0, L, hops=2, damping=0.5)

        k = 20
        Z, _ = differentiable_topk(torch.nan_to_num(s_final, nan=float("-inf")), k=k)
        M_selected = torch.einsum("bkd,bde->bke", Z, M)

        loss = M_selected.sum()
        loss.backward()

        assert M.grad is not None and torch.isfinite(M.grad).all()
        assert q.grad is not None and torch.isfinite(q.grad).all()
        assert link_module.rel.grad is not None and torch.isfinite(link_module.rel.grad).all()

    @pytest.mark.parametrize(
        "param_name", ["proj_src.weight", "proj_dst.weight", "query_gate.weight", "rel"]
    )
    def test_every_new_parameter_receives_gradient(self, small_pool, param_name):
        """Individually checks each of the four new parameters named in the
        write-up (W_src, W_dst, W_q, R) rather than only R, since 'the
        link-module parameters' plural implies all four."""
        link_module = small_pool["link_module"]
        M, q, cand_mask = small_pool["M"], small_pool["q"], small_pool["cand_mask"]
        L = link_module(M, q, cand_mask)
        s0 = F.cosine_similarity(q.unsqueeze(1), M, dim=-1)
        s0 = s0.masked_fill(~cand_mask, float("-inf"))
        s_final = multi_hop_propagate(s0, L, hops=2, damping=0.5)

        k = 20
        Z, _ = differentiable_topk(torch.nan_to_num(s_final, nan=float("-inf")), k=k)
        M_selected = torch.einsum("bkd,bde->bke", Z, M)
        loss = M_selected.sum()
        loss.backward()

        param = dict(link_module.named_parameters())[param_name]
        assert param.grad is not None, f"{param_name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{param_name} gradient contains non-finite values"
        assert param.grad.abs().sum() > 0, f"{param_name} gradient is exactly zero"

    def test_gradient_flows_through_reordering_unaffected(self, small_pool):
        """Step 5's order_by_hop runs under no_grad and must not itself break
        or silently detach the gradient path from Step 4's selection."""
        link_module = small_pool["link_module"]
        M, q, cand_mask = small_pool["M"], small_pool["q"], small_pool["cand_mask"]
        L = link_module(M, q, cand_mask)
        s0 = F.cosine_similarity(q.unsqueeze(1), M, dim=-1)
        s0 = s0.masked_fill(~cand_mask, float("-inf"))
        s_final = multi_hop_propagate(s0, L, hops=2, damping=0.5)

        k = 20
        Z, topk_idx = differentiable_topk(torch.nan_to_num(s_final, nan=float("-inf")), k=k)
        reordered_idx, perm = order_by_hop(topk_idx, L, s0, hops=2, damping=0.5)

        # Reorder Z's k-dimension to match reordered_idx, then continue the
        # differentiable path exactly as the training loop would.
        perm_expanded = perm.unsqueeze(-1).expand(-1, -1, Z.size(-1))
        Z_reordered = torch.gather(Z, 1, perm_expanded)
        M_selected = torch.einsum("bkd,bde->bke", Z_reordered, M)
        loss = M_selected.sum()
        loss.backward()

        assert M.grad is not None and torch.isfinite(M.grad).all()
        assert q.grad is not None and torch.isfinite(q.grad).all()

    def test_no_gradient_required_for_reordering_indices(self, small_pool):
        """order_by_hop's outputs are index/permutation tensors, which are
        integer-typed and must not carry (or require) gradients themselves."""
        link_module = small_pool["link_module"]
        M, q, cand_mask = small_pool["M"], small_pool["q"], small_pool["cand_mask"]
        L = link_module(M, q, cand_mask)
        s0 = F.cosine_similarity(q.unsqueeze(1), M, dim=-1)
        s0 = s0.masked_fill(~cand_mask, float("-inf"))

        k = 20
        s_final = multi_hop_propagate(s0, L, hops=2, damping=0.5)
        _, topk_idx = differentiable_topk(torch.nan_to_num(s_final, nan=float("-inf")), k=k)
        reordered_idx, perm = order_by_hop(topk_idx, L, s0, hops=2, damping=0.5)

        assert not reordered_idx.is_floating_point()
        assert not perm.is_floating_point()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))