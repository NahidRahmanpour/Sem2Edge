# train_graph_transformer.py
# UPDATED:
# - true temporal prediction: use snapshot t as input and snapshot t+1 as target
# - weighted BCE loss
# - adds Precision / Recall / F1 / MPR
# - keeps AUC / AP / Hits@1 / Hits@3 / Hits@10
# - saves metrics_full.csv + best.json
# - plots key metrics (no loss)

import os
import gc
import json
import csv
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

from graph_transformer import GraphTransformerDLP
from utils import (
    compute_auc_ap,
    compute_hits_at_k,
    compute_prf,
    compute_mpr,
    Logger,
)


# --------------------------- helpers ---------------------------

def _add_pyg_safe_globals():
    """Allow torch.load of older PyG Data types."""
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
    """Robust torch.load across PyTorch 2.6+ weights_only changes."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _get_feats(data, embedding: str):
    """Select semantic (text) vs x (structural) features."""
    if embedding == "semantic":
        return data.semantic if getattr(data, "semantic", None) is not None else data.x
    return data.x if getattr(data, "x", None) is not None else data.semantic


def _prefer_existing_with_count(base: Path, names):
    """Pick the first snapshot dir that exists with >=90 .pt files; else the one with most .pt."""
    best = None
    best_n = -1
    for name in names:
        p = base / name
        if not p.is_dir():
            continue
        n = sum(1 for _ in p.glob("*.pt"))
        if n >= 90:
            return p
        if n > best_n:
            best, best_n = p, n
    return best


def _resolve_data_dir(root: Path, dataset: str, encoder: str) -> Path:
    ds_map = {
        "enron": "enron",
        "reddit": "Reddit",
        "cit_hepth": "cit_hepth",
        "tgbl_wiki": "tgbl_wiki",
    }
    base = root / ds_map[dataset]

    if encoder == "bert":
        cand = [
            "snapshots_bert_concat768_fix",
            "snapshots_bert_dual_norm_fix",
            "snapshots_bert_hybrid_norm_fix",
            "snapshots_bert_bgeL_norm_fix",
            "snapshots_bert_bge_fix",
            "snapshots_bert_mpnet_norm_fix",
            "snapshots_bert_mpnet_fix",
            "snapshots_bert_mpnetQA_norm_fix",
            "snapshots_bert",
        ]
    else:
        cand = [
            "snapshots_llama_full_fix",
            "snapshots_llama_norm_fix",
            "snapshots_llama",
        ]

    pick = _prefer_existing_with_count(base, cand)
    if not pick or not pick.is_dir():
        raise FileNotFoundError(f"No snapshot dir found under {base} (tried {cand})")
    return pick


def _default_results_dir(repo_root: Path, dataset: str, encoder: str, epochs: int) -> Path:
    return repo_root / "results" / dataset / f"gt_{encoder}_{epochs}ep_temporal_eval"


def _set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def save_metrics_csv_full(
    csv_path,
    epochs,
    aucs,
    aps,
    precisions,
    recalls,
    f1s,
    mprs,
    hits1,
    hits3,
    hits10,
):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch",
            "auc",
            "ap",
            "precision",
            "recall",
            "f1",
            "mpr",
            "hits@1",
            "hits@3",
            "hits@10",
        ])
        for row in zip(
            epochs,
            aucs,
            aps,
            precisions,
            recalls,
            f1s,
            mprs,
            hits1,
            hits3,
            hits10,
        ):
            w.writerow(list(row))


def plot_selected_metrics(
    result_dir,
    aucs,
    aps,
    f1s,
    mprs,
    h1,
    h3,
    h10,
):
    import matplotlib.pyplot as plt

    os.makedirs(result_dir, exist_ok=True)
    x = list(range(1, len(aucs) + 1))

    def _plot_one(y, title, fname, ylabel):
        plt.figure()
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, fname), dpi=200)
        plt.close()

    _plot_one(aucs, "AUC over Epochs", "auc.png", "AUC")
    _plot_one(aps, "AP over Epochs", "ap.png", "AP")
    _plot_one(f1s, "F1 over Epochs", "f1.png", "F1")
    _plot_one(mprs, "MPR over Epochs", "mpr.png", "MPR")
    _plot_one(h1, "Hits@1 over Epochs", "hits1.png", "Hits@1")
    _plot_one(h3, "Hits@3 over Epochs", "hits3.png", "Hits@3")
    _plot_one(h10, "Hits@10 over Epochs", "hits10.png", "Hits@10")


# --------------------------- training ---------------------------

def main(
    data_dir: str,
    embedding: str = "semantic",
    epochs: int = 100,
    lr: float = 1e-3,
    result_dir: str = "results/enron/enron_graph_transformer",
    cast_float16: bool = False,
    cosine: bool = False,
    seed: int = 42,
    neg_mult: int = 0,      # 0 = auto (bert:2, llama:5)
    grad_clip: float = 0.0, # 0 = off
    threshold: float = 0.5,
):
    print("[INFO] train_graph_transformer starting", flush=True)
    _add_pyg_safe_globals()
    _set_seed(seed)

    os.makedirs(result_dir, exist_ok=True)
    logger = Logger(os.path.join(result_dir, "train_log.txt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(data_dir)

    pt_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pt"))
    if len(pt_files) < 2:
        raise FileNotFoundError(f"Need at least 2 .pt snapshots in {data_dir} for t->t+1 prediction")

    pt_paths = [os.path.join(data_dir, f) for f in pt_files]
    logger.log(f"[INFO] Streaming {len(pt_paths)} snapshots from: {data_dir}")
    logger.log(f"[INFO] Device: {device} | Embedding: {embedding}")
    logger.log(f"[INFO] Evaluation: temporal next-step prediction (t -> t+1)")

    enc_from_path = (
        "bert" if "bert" in data_dir.lower()
        else "llama" if "llama" in data_dir.lower()
        else None
    )
    norm_for_bert = (enc_from_path == "bert")
    logger.log(
        f"[CFG] ENCODER_HINT={enc_from_path or 'unknown'} | "
        f"INPUT_NORMALIZATION={'ON' if norm_for_bert else 'OFF'}"
    )

    probe = _tload(pt_paths[0])
    feats = _get_feats(probe, embedding)
    in_channels = int(feats.size(1))
    del probe
    gc.collect()
    logger.log(f"[INFO] In-channels = {in_channels}")

    def build_model(cfg):
        try:
            return GraphTransformerDLP(in_channels=in_channels, **cfg).to(device)
        except TypeError:
            cfg = {k: v for k, v in cfg.items() if k not in ("adapter", "encoder_hint", "decoder")}
            return GraphTransformerDLP(in_channels=in_channels, **cfg).to(device)

    if enc_from_path == "bert":
        cfg = dict(
            hidden_channels=192,
            out_channels=64,
            num_layers=4,
            heads=4,
            dropout=0.20,
            adapter=True,
            decoder="bilinear",
            encoder_hint="bert",
        )
        model = build_model(cfg)
        lr = max(lr, 8e-4)
        neg_mult = neg_mult or 2
        use_cosine = True
        grad_clip_val = grad_clip if grad_clip > 0 else 1.0
    else:
        cfg = dict(
            hidden_channels=96,
            out_channels=64,
            num_layers=2,
            heads=2,
            dropout=0.05,
            adapter=False,
            decoder="bilinear",
            encoder_hint="llama",
        )
        model = build_model(cfg)
        lr = min(lr, 2e-4)
        neg_mult = max(5, neg_mult or 5)
        use_cosine = bool(cosine)
        grad_clip_val = max(0.0, grad_clip)

    logger.log(
        f"[CFG] lr={lr} neg_mult={neg_mult} cosine={'ON' if use_cosine else 'OFF'} "
        f"grad_clip={grad_clip_val} threshold={threshold} | model_cfg={cfg}"
    )

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs) if use_cosine else None

    all_aucs, all_aps = [], []
    all_precs, all_recs, all_f1s, all_mprs = [], [], [], []
    all_hits1, all_hits3, all_hits10 = [], [], []

    best = {
        "auc": -1.0,
        "ap": -1.0,
        "precision": -1.0,
        "recall": -1.0,
        "f1": -1.0,
        "mpr": -1.0,
        "hits@1": -1.0,
        "hits@3": -1.0,
        "hits@10": -1.0,
        "epoch": 0,
    }

    final_predictions_path = os.path.join(result_dir, "predictions_final.csv")

    for epoch in range(1, epochs + 1):
        model.train()

        if epoch == epochs and os.path.exists(final_predictions_path):
            os.remove(final_predictions_path)

        tot_auc = 0.0
        tot_ap = 0.0
        tot_prec = 0.0
        tot_rec = 0.0
        tot_f1 = 0.0
        tot_mpr = 0.0
        tot_h1 = 0.0
        tot_h3 = 0.0
        tot_h10 = 0.0
        valid = 0

        # true temporal prediction: use snapshot t as input and snapshot t+1 as target
        for t in range(len(pt_paths) - 1):
            g = _tload(pt_paths[t])
            g_next = _tload(pt_paths[t + 1])

            x = _get_feats(g, embedding)

            if x is not None and norm_for_bert:
                x = F.normalize(x, p=2, dim=-1)
                x = F.layer_norm(x, x.shape[-1:])

            if x is not None and not norm_for_bert:
                x = F.dropout(x, p=0.15, training=True)
                x = x + 0.01 * torch.randn_like(x)

            if cast_float16:
                x = x.to(torch.float16)

            x = x.to(device)
            N = x.size(0)

            # current snapshot edges for message passing
            ei = g.edge_index
            mask_ei = (ei[0] < N) & (ei[1] < N)
            ei = ei[:, mask_ei]
            if ei.numel() == 0:
                del g, g_next, x
                gc.collect()
                continue
            ei = ei.to(device)

            # next snapshot edges are the prediction targets
            pos = g_next.edge_index
            mask_pos = (pos[0] < N) & (pos[1] < N)
            pos = pos[:, mask_pos]
            if pos.numel() == 0:
                del g, g_next, x, ei
                gc.collect()
                continue
            pos = pos.to(device)

            optim.zero_grad()
            z = model(x, ei, ei)

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

            if torch.unique(edge_lbl).numel() < 2:
                del g, g_next, x, ei, pos, neg, edge_idx, edge_lbl, z
                gc.collect()
                continue

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
                del g, g_next, x, ei, pos, neg, edge_idx, edge_lbl, z, preds, loss
                gc.collect()
                continue

            precision, recall, f1 = compute_prf(preds, edge_lbl, threshold=threshold)
            mpr = compute_mpr(preds, edge_lbl)
            h1 = compute_hits_at_k(preds, edge_lbl, k=1)
            h3 = compute_hits_at_k(preds, edge_lbl, k=3)
            h10 = compute_hits_at_k(preds, edge_lbl, k=10)

            # Save raw prediction scores for threshold tuning on the final epoch.
            # Append across all snapshot pairs so the final CSV reflects the full
            # temporal evaluation set, matching the Text2Edge evaluation flow.
            if epoch == epochs:
                import pandas as pd

                y_score = torch.sigmoid(preds).detach().cpu().numpy()
                y_true = edge_lbl.detach().cpu().numpy()

                pred_df = pd.DataFrame({
                    "y_true": y_true,
                    "y_score": y_score,
                })

                pred_path = final_predictions_path
                if not os.path.exists(pred_path):
                    pred_df.to_csv(pred_path, index=False)
                else:
                    pred_df.to_csv(pred_path, mode="a", header=False, index=False)

            loss.backward()
            if grad_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
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
            valid += 1

            del g, g_next, x, ei, pos, neg, edge_idx, edge_lbl, z, preds, loss
            gc.collect()

        if valid == 0:
            logger.log(f"[ERROR] Epoch {epoch}: no valid snapshot pairs")
            avg_auc = avg_ap = avg_prec = avg_rec = avg_f1 = avg_mpr = avg_h1 = avg_h3 = avg_h10 = 0.0
        else:
            avg_auc = tot_auc / valid
            avg_ap = tot_ap / valid
            avg_prec = tot_prec / valid
            avg_rec = tot_rec / valid
            avg_f1 = tot_f1 / valid
            avg_mpr = tot_mpr / valid
            avg_h1 = tot_h1 / valid
            avg_h3 = tot_h3 / valid
            avg_h10 = tot_h10 / valid

        all_aucs.append(avg_auc)
        all_aps.append(avg_ap)
        all_precs.append(avg_prec)
        all_recs.append(avg_rec)
        all_f1s.append(avg_f1)
        all_mprs.append(avg_mpr)
        all_hits1.append(avg_h1)
        all_hits3.append(avg_h3)
        all_hits10.append(avg_h10)

        msg = (
            f"[Epoch {epoch}] "
            f"AUC={avg_auc:.4f} | AP={avg_ap:.4f} | "
            f"P={avg_prec:.4f} | R={avg_rec:.4f} | F1={avg_f1:.4f} | MPR={avg_mpr:.4f} | "
            f"Hits@1={avg_h1:.4f} | Hits@3={avg_h3:.4f} | Hits@10={avg_h10:.4f}"
        )
        logger.log(msg)

        # best by F1 first, tie-break by AUC
        if (avg_f1 > best["f1"]) or (avg_f1 == best["f1"] and avg_auc > best["auc"]):
            best.update({
                "auc": avg_auc,
                "ap": avg_ap,
                "precision": avg_prec,
                "recall": avg_rec,
                "f1": avg_f1,
                "mpr": avg_mpr,
                "hits@1": avg_h1,
                "hits@3": avg_h3,
                "hits@10": avg_h10,
                "epoch": epoch,
            })

        if sched is not None:
            sched.step()

    epochs_list = list(range(1, len(all_aucs) + 1))
    csv_path = os.path.join(result_dir, "metrics_full.csv")
    save_metrics_csv_full(
        csv_path,
        epochs_list,
        all_aucs,
        all_aps,
        all_precs,
        all_recs,
        all_f1s,
        all_mprs,
        all_hits1,
        all_hits3,
        all_hits10,
    )

    with open(os.path.join(result_dir, "best.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    try:
        plot_selected_metrics(
            result_dir,
            all_aucs,
            all_aps,
            all_f1s,
            all_mprs,
            all_hits1,
            all_hits3,
            all_hits10,
        )
        logger.log("[INFO] Saved plots: auc.png, ap.png, f1.png, mpr.png, hits1.png, hits3.png, hits10.png")
    except Exception as e:
        logger.log(f"[WARN] Plotting failed: {e}")

    logger.log(f"[DONE] Training complete. Best epoch={best['epoch']}")
    logger.log(
        f"[DONE] Best AUC={best['auc']:.4f} | AP={best['ap']:.4f} | "
        f"P={best['precision']:.4f} | R={best['recall']:.4f} | F1={best['f1']:.4f} | "
        f"MPR={best['mpr']:.4f} | Hits@1={best['hits@1']:.4f} | "
        f"Hits@3={best['hits@3']:.4f} | Hits@10={best['hits@10']:.4f}"
    )
    logger.log(f"[DONE] Saved: {csv_path}")
    if os.path.exists(final_predictions_path):
        logger.log(f"[DONE] Saved raw scores: {final_predictions_path}")
    print(f"[INFO] Done. Results in: {result_dir}", flush=True)


# --------------------------- CLI ---------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parent


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["enron", "reddit", "cit_hepth", "tgbl_wiki"],
                   help="If set with --encoder, auto-pick data_dir and result_dir.")
    p.add_argument("--encoder", choices=["bert", "llama"],
                   help="Which snapshot family to train if --dataset is used.")
    p.add_argument("--root", default=str((_repo_root() / "data").resolve()),
                   help="Root data folder that contains dataset subfolders.")
    p.add_argument("--data_dir", help="Folder containing .pt snapshots")
    p.add_argument("--embedding", default="semantic", choices=["semantic", "x"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--result_dir", help="Where to write metrics/plots")
    p.add_argument("--cast_float16", action="store_true")
    p.add_argument("--cosine", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neg_mult", type=int, default=0, help="0=auto (bert:2, llama:5)")
    p.add_argument("--grad_clip", type=float, default=0.0, help="0=off")
    p.add_argument("--threshold", type=float, default=0.5, help="threshold for Precision/Recall/F1")
    a = p.parse_args()

    repo_root = _repo_root()
    if (a.data_dir is None or a.result_dir is None):
        if not (a.dataset and a.encoder):
            raise SystemExit("Either provide --data_dir and --result_dir, OR provide --dataset and --encoder.")
        data_dir = str(_resolve_data_dir(Path(a.root), a.dataset, a.encoder))
        result_dir = a.result_dir or str(_default_results_dir(repo_root, a.dataset, a.encoder, a.epochs))
    else:
        data_dir = a.data_dir
        result_dir = a.result_dir

    main(
        data_dir=data_dir,
        embedding=a.embedding,
        epochs=a.epochs,
        lr=a.lr,
        result_dir=result_dir,
        cast_float16=a.cast_float16,
        cosine=a.cosine,
        seed=a.seed,
        neg_mult=a.neg_mult,
        grad_clip=a.grad_clip,
        threshold=a.threshold,
    )