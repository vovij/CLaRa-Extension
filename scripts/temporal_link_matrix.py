"""
Differentiable temporal-link multi-hop retrieval module for CLaRa.

Implements Eqs. (1)-(11) of the multi-hop extension write-up:
    Q -> Dict(cos-sim over D candidates) -> KG(this file) -> top-k (differentiable_topk) -> Ans

All ops are differentiable end-to-end via the same straight-through (ST) trick
already used in modeling_clara.py's differentiable_topk, so no new supervision
is required -- gradients from the NTP loss on the answer reach theta_qr, theta_c,
and the new link-matrix params {W_src, W_dst, W_q, R} through the same path as
the existing top-k selection.

This file is intentionally dependency-free w.r.t. modeling_clara.py (no import
of CLaRa internals) so it can be unit-tested in isolation -- see the __main__
block and tests/test_temporal_link_matrix.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLinkMatrix(nn.Module):
    """
    Learns a directed, query-conditioned adjacency over the D candidate
    document embeddings: L[b, i, j] ~ "how useful is it to hop from doc i to
    doc j, given q" (Eqs. 3-5 of the write-up).

    Parameters
    ----------
    hidden_dim : int
        E, the compressor embedding dimension (must match CLaRa's hidden_size,
        i.e. the last dim of the retrieved_doc_embeddings tensor).
    rank : int
        r, the low-rank projection dim. r << E keeps the D x D bilinear
        comparison cheap at D=100. Default 64, per the write-up's worked
        example (E=512, r=64).
    """

    def __init__(self, hidden_dim: int, rank: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        # W_src, W_dst: independent projections, no bias, no weight tying (see
        # "Role separation" in the design rationale -- tying them would force
        # L_ij == L_ji and undo the asymmetry R is introduced to provide).
        self.proj_src = nn.Linear(hidden_dim, rank, bias=False)
        self.proj_dst = nn.Linear(hidden_dim, rank, bias=False)
        # R: bilinear form, initialised near-identity/small so the *initial*
        # L is close to uniform rather than confidently (and randomly) peaked
        # before training has learned anything.
        self.rel = nn.Parameter(torch.eye(rank) * 0.01)
        # W_q: query-conditioning gate. Small init keeps early tanh activations
        # near the origin (gate ~= 0 by default), avoiding saturation.
        self.query_gate = nn.Linear(hidden_dim, rank, bias=False)

        nn.init.normal_(self.proj_src.weight, std=0.02)
        nn.init.normal_(self.proj_dst.weight, std=0.02)
        nn.init.normal_(self.query_gate.weight, std=0.02)

    def forward(self, M: torch.Tensor, q: torch.Tensor, cand_mask: torch.Tensor,
                temperature: float = 1.0) -> torch.Tensor:
        """
        Args:
            M: [B, D, E] candidate (flattened) memory-token embeddings
            q: [B, E]    query embedding, same space as M (from theta_qr)
            cand_mask: [B, D] bool, True for valid (non-padded) candidates
        Returns:
            L: [B, D, D] row-stochastic directed link matrix (Eq. 5).
               Rows for fully-masked queries are all-zero (nan_to_num'd),
               not uniform -- so a padded candidate contributes no outgoing
               relevance either.
        """
        B, D, E = M.shape
        assert E == self.hidden_dim, (
            f"TemporalLinkMatrix hidden_dim={self.hidden_dim} does not match "
            f"candidate embedding dim={E}. Check compr_rate / hidden_size wiring."
        )
        compute_dtype = self.proj_src.weight.dtype
        M = M.to(compute_dtype)
        q = q.to(compute_dtype)

        src = self.proj_src(M)                                # [B, D, r]
        dst = self.proj_dst(M)                                # [B, D, r]
        gate = torch.tanh(self.query_gate(q)).unsqueeze(1)     # [B, 1, r]

        src = src * (1.0 + gate)                               # Eq. 3, query-conditioned source
        logits = torch.einsum("bir,rs,bjs->bij", src, self.rel, dst) / (self.rank ** 0.5)
        # NOTE: the write-up divides by sqrt(E) (Vaswani-style scaling on the
        # embedding dim); we scale by sqrt(rank) instead since the bilinear
        # form actually operates in the r-dimensional projected space, which
        # keeps logits in a stable range across different `rank` choices.
        # If you need to match the paper equation literally, divide by E**0.5.

        pad = (~cand_mask).unsqueeze(1).expand(-1, D, -1)       # [B, D, D], j is masked
        eye = torch.eye(D, device=M.device, dtype=torch.bool).unsqueeze(0)
        logits = logits.masked_fill(pad | eye, float("-inf"))

        L = F.softmax(logits / max(temperature, 1e-6), dim=-1)  # row-stochastic
        L = torch.nan_to_num(L, nan=0.0)  # fully-masked rows -> all-zero row
        return L


def multi_hop_propagate(s0: torch.Tensor, L: torch.Tensor, hops: int = 2,
                         damping: float = 0.5) -> torch.Tensor:
    """
    Personalized-PageRank propagation of base retrieval scores through L
    (Eqs. 9-10). s0 is treated as the personalization/reset distribution.

    Args:
        s0: [B, D]    initial scores, e.g. cos(q, M_i), possibly containing
                       -inf for padded candidates (masked_fill'd upstream).
        L:  [B, D, D] output of TemporalLinkMatrix
        hops: T, number of propagation steps (2 for HotpotQA/2WikiMultiHopQA,
              up to 4 for MuSiQue per the write-up).
        damping: a in [0, 1]. a=0 recovers the current (no-KG) pipeline exactly.
    Returns:
        s_final: [B, D] graph-informed relevance scores, safe to feed straight
                  into differentiable_topk (no -inf/nan; padded slots -> ~0).
    """
    # Force fp32 explicitly at every step. Under bf16 autocast (e.g. DeepSpeed's
    # --bf16, which wraps the whole forward() call), torch.einsum is on the
    # "cast to lower precision" list -- it silently downcasts fp32 inputs to
    # bf16 for the op AND returns bf16, regardless of what dtype was passed
    # in. Relying on the caller's .float() alone is not sufficient: after one
    # loop iteration `s` becomes bf16 from the einsum's own output, and the
    # next iteration's implicit L(fp32)/s(bf16) mix raises
    # "expected scalar type Float but found BFloat16". Casting inside the
    # loop on every use makes this immune to ambient autocast state.
    s0_soft = F.softmax(s0.float(), dim=-1)  # softmax(-inf) -> 0 for padded slots
    s = s0_soft.clone()
    for _ in range(hops):
        propagated = torch.einsum("bij,bi->bj", L.float(), s.float())
        s = (1 - damping) * s0_soft + damping * propagated
    return s


def order_by_hop(topk_idx: torch.Tensor, L: torch.Tensor, s0: torch.Tensor,
                  hops: int, damping: float) -> torch.Tensor:
    """
    Step 5 (evidence ordering): order the k selected documents by the hop
    at which each first "crossed the selection threshold" -- approximated
    here as the hop at which each selected doc's running propagated score
    first exceeds its own s0 baseline by a growing margin, i.e. the earliest
    hop at which it is receiving meaningful incoming relevance rather than
    just its direct-similarity floor. Bridge documents (found at hop 0,
    since they score reasonably against q directly) sort before pure
    targets (which only light up after propagation).

    This is a best-effort, differentiable-pipeline-compatible ordering: it
    only permutes indices (no gradient needed through the ordering itself,
    only through which docs got selected and their embeddings), so it is
    applied with torch.no_grad().

    Args:
        topk_idx: [B, k] indices into the D candidates, from differentiable_topk
        L, s0: as above
        hops, damping: same values used in multi_hop_propagate
    Returns:
        reordered_topk_idx: [B, k], same index set, reordered along dim=1
        perm: [B, k] the permutation applied (useful for reordering embeddings)
    """
    with torch.no_grad():
        B, k = topk_idx.shape
        # Same explicit-fp32-under-autocast fix as multi_hop_propagate above.
        s0_soft = F.softmax(s0.float(), dim=-1)
        s = s0_soft.clone()
        first_hop = torch.full((B, s0.shape[1]), hops, device=s0.device, dtype=torch.long)
        gathered_hop = torch.zeros(B, k, device=s0.device, dtype=torch.long)
        for t in range(hops):
            propagated = torch.einsum("bij,bi->bj", L.float(), s.float())
            s = (1 - damping) * s0_soft + damping * propagated
            gained = torch.gather(s, 1, topk_idx) > torch.gather(s0_soft, 1, topk_idx) * 1.05
            newly_arrived = gained & (torch.gather(first_hop, 1, topk_idx) == hops)
            src_val = torch.full_like(gathered_hop, t)
            gathered_hop = torch.where(newly_arrived, src_val, gathered_hop)
            first_hop.scatter_(1, topk_idx, torch.where(newly_arrived, src_val, torch.gather(first_hop, 1, topk_idx)))
        gathered_hop = torch.where(gathered_hop == 0, torch.gather(first_hop, 1, topk_idx).clamp(max=0), gathered_hop)
        perm = gathered_hop.argsort(dim=1, stable=True)
        reordered_topk_idx = torch.gather(topk_idx, 1, perm)
        return reordered_topk_idx, perm


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, E = 2, 100, 512

    M = torch.randn(B, D, E, requires_grad=True)
    q = torch.randn(B, E, requires_grad=True)
    cand_mask = torch.ones(B, D, dtype=torch.bool)
    cand_mask[0, 80:] = False

    link_module = TemporalLinkMatrix(hidden_dim=E, rank=64)
    L = link_module(M, q, cand_mask)
    assert L.shape == (B, D, D)
    row_sums = L.sum(-1)
    assert torch.allclose(row_sums[cand_mask], torch.ones_like(row_sums[cand_mask]), atol=1e-4)
    print("[ok] L row-stochastic where valid, shape", L.shape)

    s0 = F.cosine_similarity(q.unsqueeze(1), M, dim=-1)
    s0 = s0.masked_fill(~cand_mask, float("-inf"))
    s_final = multi_hop_propagate(s0, L, hops=2, damping=0.5)
    assert torch.isfinite(s_final).all()
    print("[ok] s_final finite, shape", s_final.shape)

    from torch.nn.functional import softmax as _sm

    def differentiable_topk(logits, k, temperature=1.0):
        Bb, N = logits.shape
        perturbed = logits / max(temperature, 1e-6)
        topk_vals, topk_idx = perturbed.topk(k, dim=-1)
        K_hard = torch.zeros(Bb, k, N, device=logits.device, dtype=logits.dtype)
        K_hard.scatter_(2, topk_idx.unsqueeze(-1), 1.0)
        K_soft = torch.zeros_like(K_hard)
        taken = torch.zeros(Bb, N, device=logits.device, dtype=logits.dtype)
        for j in range(k):
            mask = (1.0 - taken.detach())
            masked = perturbed + (mask + 1e-8).log()
            pj = _sm(masked, dim=-1).float()
            K_soft[:, j, :] = pj
            taken = torch.clamp(taken + K_hard[:, j, :], max=1.0)
        W = K_hard + (K_soft - K_soft.detach())
        return W, topk_idx

    Z, topk_idx = differentiable_topk(torch.nan_to_num(s_final, nan=float("-inf")), k=20)
    M_selected = torch.einsum("bkd,bde->bke", Z, M)
    print("[ok] M_selected", M_selected.shape)

    reordered_idx, perm = order_by_hop(topk_idx, L, s0, hops=2, damping=0.5)
    print("[ok] reordered_idx", reordered_idx.shape)

    loss = M_selected.sum()
    loss.backward()
    assert M.grad is not None and q.grad is not None and link_module.rel.grad is not None
    print("[ok] gradients reach M, q, and link_module.rel")