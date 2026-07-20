"""
Sketch: differentiable temporal-link multi-hop retrieval module for CLaRa.

Slots into the existing pipeline as:

    Q -> Dict(cos-sim over 100 candidates) -> KG(this file) -> top-20 (existing differentiable_topk) -> Ans

All ops are differentiable end-to-end via the same straight-through (ST) trick
already used in modeling_clara.py's differentiable_topk, so no new loss labels
are required -- gradients from the NTP loss on the answer still reach theta_qr,
theta_c, and the new link-matrix params (theta_link).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLinkMatrix(nn.Module):
    """
    Learns a directed, query-conditioned adjacency over the D candidate
    document embeddings, analogous to the DNC's temporal link matrix:
    L[b, i, j] ~ "how useful is it to hop from doc i to doc j, given q".

    Kept deliberately close to modeling_clara.py's style (explicit B/D/E
    dims, per-row softmax with additive log-mask) so it's easy to review
    next to differentiable_topk / differentiable_topk_top_1.
    """

    def __init__(self, hidden_dim: int, rank: int = 128):
        super().__init__()
        # Low-rank bilinear form keeps this cheap at D=100: (D x r) @ (r x r) @ (r x D)
        self.proj_src = nn.Linear(hidden_dim, rank, bias=False)
        self.proj_dst = nn.Linear(hidden_dim, rank, bias=False)
        self.rel = nn.Parameter(torch.eye(rank) * 0.01)  # init near-identity, small
        # Query conditions the edge scores (a link is only useful for THIS question)
        self.query_gate = nn.Linear(hidden_dim, rank, bias=False)

    def forward(self, M: torch.Tensor, q: torch.Tensor, cand_mask: torch.Tensor,
                temperature: float = 1.0) -> torch.Tensor:
        """
        Args:
            M:  [B, D, E]  candidate (flattened) memory-token embeddings, D=100
            q:  [B, E]     query embedding (same space as M, from theta_qr)
            cand_mask: [B, D] bool, True for valid (non-padded) candidates
        Returns:
            L: [B, D, D] row-stochastic directed link matrix (soft "edges")
        """
        B, D, E = M.shape
        src = self.proj_src(M)                       # [B, D, r]
        dst = self.proj_dst(M)                        # [B, D, r]
        gate = torch.tanh(self.query_gate(q)).unsqueeze(1)  # [B, 1, r]

        src = src * (1.0 + gate)                       # query-condition the source side
        logits = torch.einsum("bir,rs,bjs->bij", src, self.rel, dst) / (E ** 0.5)

        # mask invalid candidates (padding) and self-loops
        pad = (~cand_mask).unsqueeze(1).expand(-1, D, -1)      # [B, D, D]
        eye = torch.eye(D, device=M.device, dtype=torch.bool).unsqueeze(0)
        logits = logits.masked_fill(pad | eye, float("-inf"))

        L = F.softmax(logits / max(temperature, 1e-6), dim=-1)  # row-stochastic
        L = torch.nan_to_num(L, nan=0.0)  # rows that are fully masked -> all-zero row
        return L


def multi_hop_propagate(s0: torch.Tensor, L: torch.Tensor, hops: int = 2,
                         damping: float = 0.5) -> torch.Tensor:
    """
    Personalized-PageRank-style propagation of the base retrieval scores
    through the learned link matrix, so a document's final score reflects
    both direct query similarity AND being reachable from other relevant docs.

        s_(t+1) = (1 - damping) * s0 + damping * L^T @ s_(t)

    Args:
        s0: [B, D]    initial scores, e.g. cos(q, M_i) from the existing Dict stage
        L:  [B, D, D] output of TemporalLinkMatrix
        hops: number of propagation steps (2-3 to match typical HotpotQA/MuSiQue hop counts)
    Returns:
        s_final: [B, D] graph-informed relevance scores, feed straight into
                  the existing differentiable_topk(logits=s_final, k=20)
    """
    s = F.softmax(s0, dim=-1)
    s0_soft = s.clone()
    for _ in range(hops):
        s = (1 - damping) * s0_soft + damping * torch.einsum("bij,bi->bj", L, s)
    return s


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, E = 2, 100, 512  # batch, candidate pool (was 20, now widened to 100), hidden dim

    M = torch.randn(B, D, E, requires_grad=True)
    q = torch.randn(B, E, requires_grad=True)
    cand_mask = torch.ones(B, D, dtype=torch.bool)
    cand_mask[0, 80:] = False  # simulate a shorter candidate list for sample 0

    link_module = TemporalLinkMatrix(hidden_dim=E, rank=64)
    L = link_module(M, q, cand_mask)
    print("L shape:", L.shape, "row sums (should be ~1 or 0 for fully masked rows):")
    print(L.sum(-1)[:, :5])

    s0 = F.cosine_similarity(q.unsqueeze(1), M, dim=-1)  # [B, D], the existing "Dict" score
    s0 = s0.masked_fill(~cand_mask, float("-inf"))
    s_final = multi_hop_propagate(s0, L, hops=2, damping=0.5)
    print("s_final shape:", s_final.shape, "sums:", s_final.sum(-1))

    # Reuse the EXISTING differentiable_topk from modeling_clara.py unchanged:
    def differentiable_topk(logits, k, temperature=1.0):
        B, N = logits.shape
        perturbed = logits / max(temperature, 1e-6)
        topk_vals, topk_idx = perturbed.topk(k, dim=-1)
        K_hard = torch.zeros(B, k, N, device=logits.device, dtype=logits.dtype)
        K_hard.scatter_(2, topk_idx.unsqueeze(-1), 1.0)
        K_soft = torch.zeros_like(K_hard)
        taken = torch.zeros(B, N, device=logits.device, dtype=logits.dtype)
        for j in range(k):
            mask = (1.0 - taken.detach())
            masked = perturbed + (mask + 1e-8).log()
            pj = F.softmax(masked, dim=-1).float()
            K_soft[:, j, :] = pj
            taken = torch.clamp(taken + K_hard[:, j, :], max=1.0)
        W = K_hard + (K_soft - K_soft.detach())
        return W, topk_idx

    s_final_filled = torch.nan_to_num(s_final, nan=float("-inf"))
    Z, topk_idx = differentiable_topk(s_final_filled, k=20)
    M_selected = torch.einsum("bkd,bde->bke", Z, M)  # [B, 20, E] -> feed to generator
    print("M_selected shape:", M_selected.shape)

    # Sanity check: gradient flows all the way back to M and q through
    # TemporalLinkMatrix, multi_hop_propagate, AND differentiable_topk.
    loss = M_selected.sum()
    loss.backward()
    print("grad reaches M:", M.grad is not None, "grad reaches q:", q.grad is not None)
    print("grad reaches link_module.rel:", link_module.rel.grad is not None)