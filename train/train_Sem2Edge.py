import os
import gc
import json
import csv
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

from dual_temporal_text2edge import DualTemporalText2Edge
from utils import (
    compute_auc_ap,
    compute_prf,
    compute_mpr,
    Logger,
)


def _add_pyg_safe_globals():
    try:
        from torch.serialization import add_safe_globals
        from torch_geometric.data import Data
        try:
            from torch_geometric.data.data import DataEdgeAttr
        except Exception:
            DataEdgeAttr = None
        add_safe_globals([Data] + ([DataEdgeAttr] if DataEdgeAttr else []))
    except Exception:
        pass


def _tload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _get_feats(data, embedding: str):
    if embedding == "semantic":
        return data.semantic if getattr(data, "semantic", None) is not None else data.x
    return data.x if getattr(data, "x", None) is not None else data.semantic


def _set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _align_to_ref(x_ref: torch.Tensor, x_other: torch.Tensor) -> torch.Tensor:
    n_ref, d_ref = x_ref.size()
    n_other, d_other = x_other.size()

    if d_ref != d_other:
        raise ValueError(f"Feature dimension mismatch: ref={d_ref}, other={d_other}")

    if n_other == n_ref:
        return x_other
    if n_other > n_ref:
        return x_other[:n_ref]

    pad = torch.zeros((n_ref - n_other, d_ref), dtype=x_other.dtype, device=x_other.device)
    return torch.cat([x_other, pad], dim=0)


def _build_sequence_inputs(graph_list, embedding: str, norm_for_bert: bool):
    """
    Build aligned feature tensors [x_{t-k}, ..., x_t]
    aligned to the node count of the latest graph in the window.
    """
    x_ref = _get_feats(graph_list[-1], embedding)

    if norm_for_bert:
        x_ref = F.normalize(x_ref, p=2, dim=-1)
        x_ref = F.layer_norm(x_ref, x_ref.shape[-1:])

    seq = []
    for g in graph_list:
        x = _get_feats(g, embedding)

        if norm_for_bert:
            x = F.normalize(x, p=2, dim=-1)
            x = F.layer_norm(x, x.shape[-1:])

        x = _align_to_ref(x_ref, x)
        seq.append(x)

    return seq


def save_metrics_csv_full(csv_path, epochs, aucs, aps, precisions, recalls, f1s, mprs, hits1, hits3, hits10, mrrs):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "auc", "ap", "precision", "recall", "f1", "mpr", "hits@1", "hits@3", "hits@10", "mrr"])
        for row in zip(epochs, aucs, aps, precisions, recalls, f1s, mprs, hits1, hits3, hits10, mrrs):
            w.writerow(list(row))


def _append_predictions_csv(csv_path: str, y_true: torch.Tensor, logits: torch.Tensor):
    y_true_np = y_true.detach().cpu().numpy().astype(np.int64)
    y_score_np = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    df = pd.DataFrame({"y_true": y_true_np, "y_score": y_score_np})
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)


def _save_checkpoint(path, epoch, model, optim, best, all_aucs, all_aps, all_precs, all_recs, all_f1s, all_mprs, all_hits1, all_hits3, all_hits10, all_mrrs):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "best": best,
        "all_aucs": all_aucs,
        "all_aps": all_aps,
        "all_precs": all_precs,
        "all_recs": all_recs,
        "all_f1s": all_f1s,
        "all_mprs": all_mprs,
        "all_hits1": all_hits1,
        "all_hits3": all_hits3,
        "all_hits10": all_hits10,
        "all_mrrs": all_mrrs,
    }
    torch.save(payload, path)


def main(
    data_dir: str,
    embedding: str = "x",
    epochs: int = 20,
    lr: float = 1e-3,
    result_dir: str = "results/enron/dual_temporal_text2edge",
    seed: int = 42,
    neg_mult: int = 2,
    threshold: float = 0.6,
    weight_decay: float = 1e-4,
    window_size: int = 4,
    hidden_channels: int = 128,
    out_channels: int = 64,
    gat_heads: int = 4,
    dropout: float = 0.1,
    resume: bool = False,
    rank_neg_mult: int = 100,
    rank_batch_size: int = 512,
):
    print("[INFO] train_dual_temporal_text2edge starting", flush=True)
    _add_pyg_safe_globals()
    _set_seed(seed)

    os.makedirs(result_dir, exist_ok=True)
    logger = Logger(os.path.join(result_dir, "train_log.txt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pt_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pt"))
    if len(pt_files) < window_size + 1:
        raise FileNotFoundError(
            f"Need at least {window_size + 1} .pt snapshots in {data_dir} "
            f"for a history window of {window_size}"
        )

    pt_paths = [os.path.join(data_dir, f) for f in pt_files]
    logger.log(f"[INFO] Loading {len(pt_paths)} snapshots once from: {data_dir}")
    logger.log(f"[INFO] Device: {device} | Embedding: {embedding}")
    logger.log(f"[INFO] Evaluation: use snapshots [t-{window_size-1} ... t] -> predict edges in t+1")

    # FAST MODE: load all snapshots once instead of reloading .pt files inside every epoch/window.
    logger.log("[LOAD] Loading all snapshots into RAM once...")
    snapshots = [_tload(p) for p in pt_paths]
    logger.log(f"[LOAD] Loaded {len(snapshots)} snapshots.")

    enc_from_path = "bert" if "bert" in data_dir.lower() else ("llama" if "llama" in data_dir.lower() else None)
    norm_for_bert = enc_from_path == "bert"
    logger.log(
        f"[CFG] ENCODER_HINT={enc_from_path or 'unknown'} | "
        f"INPUT_NORMALIZATION={'ON' if norm_for_bert else 'OFF'} | "
        f"WINDOW_SIZE={window_size}"
    )

    probe = snapshots[0]
    feats = _get_feats(probe, embedding)
    in_channels = int(feats.size(1))

    model = DualTemporalText2Edge(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        gat_heads=gat_heads,
        dropout=dropout,
    ).to(device)

    logger.log(
        f"[CFG] in_channels={in_channels} hidden_channels={hidden_channels} "
        f"out_channels={out_channels} gat_heads={gat_heads} dropout={dropout} "
        f"lr={lr} weight_decay={weight_decay} neg_mult={neg_mult} threshold={threshold} rank_neg_mult={rank_neg_mult} rank_batch_size={rank_batch_size} (ranking eval cost)"
    )

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    all_aucs, all_aps, all_precs, all_recs, all_f1s, all_mprs = [], [], [], [], [], []
    all_hits1, all_hits3, all_hits10, all_mrrs = [], [], [], []

    best = {"auc": -1.0, "ap": -1.0, "f1": -1.0, "mrr": -1.0, "hits@1": -1.0, "hits@3": -1.0, "hits@10": -1.0, "epoch": 0}
    final_predictions_path = os.path.join(result_dir, "predictions_final.csv")
    last_ckpt_path = os.path.join(result_dir, "last_checkpoint.pt")
    best_ckpt_path = os.path.join(result_dir, "best_model.pt")
    best_json_path = os.path.join(result_dir, "best.json")
    csv_path = os.path.join(result_dir, "metrics_full.csv")

    start_epoch = 1
    if resume and os.path.exists(last_ckpt_path):
        logger.log(f"[RESUME] Loading checkpoint: {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])

        best = ckpt.get("best", best)
        all_aucs = ckpt.get("all_aucs", [])
        all_aps = ckpt.get("all_aps", [])
        all_precs = ckpt.get("all_precs", [])
        all_recs = ckpt.get("all_recs", [])
        all_f1s = ckpt.get("all_f1s", [])
        all_mprs = ckpt.get("all_mprs", [])
        all_hits1 = ckpt.get("all_hits1", [])
        all_hits3 = ckpt.get("all_hits3", [])
        all_hits10 = ckpt.get("all_hits10", [])
        all_mrrs = ckpt.get("all_mrrs", [])
        start_epoch = int(ckpt["epoch"]) + 1
        logger.log(f"[RESUME] Continuing from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()

        tot_auc = tot_ap = tot_prec = tot_rec = tot_f1 = tot_mpr = 0.0
        tot_h1 = tot_h3 = tot_h10 = tot_mrr = 0.0
        valid = 0

        if epoch == epochs and os.path.exists(final_predictions_path):
            os.remove(final_predictions_path)

        for t in range(window_size - 1, len(pt_paths) - 1):
            graph_window = [snapshots[k] for k in range(t - window_size + 1, t + 1)]
            g_cur = graph_window[-1]
            g_next = snapshots[t + 1]

            x_seq = _build_sequence_inputs(graph_window, embedding, norm_for_bert)
            x_seq = [x.to(device) for x in x_seq]

            N = x_seq[-1].size(0)

            edge_index_seq = []
            for g_w in graph_window:
                ei = g_w.edge_index
                mask_ei = (ei[0] < N) & (ei[1] < N)
                ei = ei[:, mask_ei]
                if ei.numel() == 0:
                    # skip bad window
                    edge_index_seq = None
                    break
                edge_index_seq.append(ei.to(device))

            if edge_index_seq is None:
                del graph_window, g_cur, g_next, x_seq
                continue

            pos = g_next.edge_index
            mask_pos = (pos[0] < N) & (pos[1] < N)
            pos = pos[:, mask_pos]
            if pos.numel() == 0:
                del graph_window, g_cur, g_next, x_seq, edge_index_seq, pos
                continue
            pos = pos.to(device)

            optim.zero_grad()

            z = model(x_seq, edge_index_seq)

            neg = negative_sampling(
                edge_index=pos,
                num_nodes=N,
                num_neg_samples=pos.size(1) * int(neg_mult),
            ).to(device)

            edge_idx = torch.cat([pos, neg], dim=1)
            edge_lbl = torch.cat([
                torch.ones(pos.size(1), device=device),
                torch.zeros(neg.size(1), device=device),
            ])

            preds = model.decode(z, edge_idx).view(-1)

            pos_weight = torch.tensor(
                [neg.size(1) / max(pos.size(1), 1)],
                device=device,
                dtype=preds.dtype,
            )
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = criterion(preds, edge_lbl)

            auc, ap = compute_auc_ap(preds, edge_lbl)
            if np.isnan(auc) or np.isnan(ap):
                del graph_window, g_cur, g_next, x_seq, edge_index_seq, pos, neg, edge_idx, edge_lbl, z, preds, loss
                continue

            precision, recall, f1 = compute_prf(preds, edge_lbl, threshold=threshold)
            mpr = compute_mpr(preds, edge_lbl)

            # Correct multi-negative ranking metrics, computed in batches to avoid CPU/GPU OOM.
            # Each positive edge is ranked against rank_neg_mult sampled negatives.
            with torch.no_grad():
                pos_scores_rank_all = model.decode(z, pos).view(-1)
                rank_chunks = []

                bs = max(1, int(rank_batch_size))
                rn = max(1, int(rank_neg_mult))

                for start in range(0, pos.size(1), bs):
                    end = min(start + bs, pos.size(1))
                    cur_bs = end - start

                    pos_scores_b = pos_scores_rank_all[start:end]

                    rank_neg_b = negative_sampling(
                        edge_index=pos,
                        num_nodes=N,
                        num_neg_samples=cur_bs * rn,
                    ).to(device)

                    # If negative_sampling returns fewer samples, pad by resampling.
                    # This prevents reshape errors on dense/small graphs.
                    if rank_neg_b.size(1) < cur_bs * rn:
                        need = cur_bs * rn - rank_neg_b.size(1)
                        extra = negative_sampling(
                            edge_index=pos,
                            num_nodes=N,
                            num_neg_samples=need,
                        ).to(device)
                        rank_neg_b = torch.cat([rank_neg_b, extra], dim=1)

                    rank_neg_b = rank_neg_b[:, :cur_bs * rn]

                    neg_scores_b = model.decode(z, rank_neg_b).view(cur_bs, rn)
                    ranks_b = 1 + (neg_scores_b > pos_scores_b.view(-1, 1)).sum(dim=1)
                    rank_chunks.append(ranks_b.detach().cpu())

                    del rank_neg_b, neg_scores_b, ranks_b

                ranks = torch.cat(rank_chunks, dim=0)

            h1 = (ranks <= 1).float().mean().item()
            h3 = (ranks <= 3).float().mean().item()
            h10 = (ranks <= 10).float().mean().item()
            mrr = (1.0 / ranks.float()).mean().item()

            if epoch == epochs:
                _append_predictions_csv(final_predictions_path, edge_lbl, preds)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            tot_auc += float(auc)
            tot_ap += float(ap)
            tot_prec += float(precision)
            tot_rec += float(recall)
            tot_f1 += float(f1)
            tot_mpr += float(mpr)
            tot_h1 += float(h1)
            tot_h3 += float(h3)
            tot_h10 += float(h10)
            tot_mrr += float(mrr)
            valid += 1

            del graph_window, g_cur, g_next, x_seq, edge_index_seq, pos, neg, edge_idx, edge_lbl, z, preds, loss

        if valid == 0:
            raise RuntimeError("No valid snapshot windows were processed. Check snapshot contents and node alignment.")

        avg_auc = tot_auc / valid
        avg_ap = tot_ap / valid
        avg_prec = tot_prec / valid
        avg_rec = tot_rec / valid
        avg_f1 = tot_f1 / valid
        avg_mpr = tot_mpr / valid
        avg_h1 = tot_h1 / valid
        avg_h3 = tot_h3 / valid
        avg_h10 = tot_h10 / valid
        avg_mrr = tot_mrr / valid

        all_aucs.append(avg_auc)
        all_aps.append(avg_ap)
        all_precs.append(avg_prec)
        all_recs.append(avg_rec)
        all_f1s.append(avg_f1)
        all_mprs.append(avg_mpr)
        all_hits1.append(avg_h1)
        all_hits3.append(avg_h3)
        all_hits10.append(avg_h10)
        all_mrrs.append(avg_mrr)

        logger.log(
            f"[Epoch {epoch}] AUC={avg_auc:.4f} | AP={avg_ap:.4f} | "
            f"P={avg_prec:.4f} | R={avg_rec:.4f} | F1={avg_f1:.4f} | "
            f"MPR={avg_mpr:.4f} | Hits@1={avg_h1:.4f} | Hits@3={avg_h3:.4f} | Hits@10={avg_h10:.4f} | MRR={avg_mrr:.4f}"
        )

        # Select the best checkpoint by AP first, then AUC.
        # This is more appropriate for DLP evaluation than selecting by F1 threshold.
        if (avg_ap > best["ap"]) or (avg_ap == best["ap"] and avg_auc > best["auc"]):
            best.update({"auc": avg_auc, "ap": avg_ap, "f1": avg_f1, "mrr": avg_mrr, "hits@1": avg_h1, "hits@3": avg_h3, "hits@10": avg_h10, "epoch": epoch})
            _save_checkpoint(
                best_ckpt_path,
                epoch,
                model,
                optim,
                best,
                all_aucs,
                all_aps,
                all_precs,
                all_recs,
                all_f1s,
                all_mprs,
                all_hits1,
                all_hits3,
                all_hits10,
                all_mrrs,
            )

        save_metrics_csv_full(
            csv_path,
            list(range(1, len(all_aucs) + 1)),
            all_aucs,
            all_aps,
            all_precs,
            all_recs,
            all_f1s,
            all_mprs,
            all_hits1,
            all_hits3,
            all_hits10,
            all_mrrs,
        )

        with open(best_json_path, "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)

        _save_checkpoint(
            last_ckpt_path,
            epoch,
            model,
            optim,
            best,
            all_aucs,
            all_aps,
            all_precs,
            all_recs,
            all_f1s,
            all_mprs,
            all_hits1,
            all_hits3,
            all_hits10,
            all_mrrs,
        )

    logger.log(f"[DONE] Best epoch={best['epoch']} | AUC={best['auc']:.4f} | AP={best['ap']:.4f} | F1={best['f1']:.4f} | Hits@10={best['hits@10']:.4f} | MRR={best['mrr']:.4f}")
    logger.log(f"[DONE] Saved: {csv_path}")
    logger.log(f"[DONE] Saved best checkpoint: {best_ckpt_path}")
    logger.log(f"[DONE] Saved last checkpoint: {last_ckpt_path}")
    if os.path.exists(final_predictions_path):
        logger.log(f"[DONE] Saved raw scores: {final_predictions_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--embedding", default="x", choices=["semantic", "x"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neg_mult", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--window_size", type=int, default=4)
    p.add_argument("--hidden_channels", type=int, default=128)
    p.add_argument("--out_channels", type=int, default=64)
    p.add_argument("--gat_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--rank_neg_mult", type=int, default=100)
    p.add_argument("--rank_batch_size", type=int, default=512)
    args = p.parse_args()

    main(
        data_dir=args.data_dir,
        embedding=args.embedding,
        epochs=args.epochs,
        lr=args.lr,
        result_dir=args.result_dir,
        seed=args.seed,
        neg_mult=args.neg_mult,
        threshold=args.threshold,
        weight_decay=args.weight_decay,
        window_size=args.window_size,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
        resume=args.resume,
        rank_neg_mult=args.rank_neg_mult,
        rank_batch_size=args.rank_batch_size,
    )