"""Training loop for ALIGNN-CSP."""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from alignn.inverse.data import (
    CrystalDataset,
    Normalizer,
    batch_to,
    collate,
    compute_normalizer,
)
from alignn.inverse.diffusion import (
    DiffusionSchedule,
    lattice_to_vec6,
    vec6_to_lattice,
)
from alignn.inverse.model import (
    ALIGNNCSP,
    available_modalities,
    build_cond_values,
)


class EMA:
    """Exponential moving average of parameters.

    Sampling quality for diffusion models depends on this far more than on the
    last few points of training loss, so it is on by default.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


def diffusion_loss(
    model: ALIGNNCSP,
    schedule: DiffusionSchedule,
    normalizer: Normalizer,
    batch: Dict,
    cond_dropout: Dict[str, float],
    lattice_weight: float,
    frac_weight: float,
) -> Dict[str, torch.Tensor]:
    device = batch["frac"].device
    natoms = batch["natoms"]
    n_graphs = int(natoms.shape[0])

    x0 = normalizer.norm_lattice(lattice_to_vec6(batch["lattice"], natoms))
    t = torch.randint(1, schedule.num_steps + 1, (n_graphs,), device=device)
    x_t, eps_lat = schedule.noise_lattice(x0, t)

    t_node = t[batch["node_graph_id"]]
    f_t, target_frac = schedule.noise_frac(batch["frac"], t_node)

    lattice_t = vec6_to_lattice(normalizer.denorm_lattice(x_t), natoms)

    # Classifier-free guidance: drop each modality independently, so one set
    # of weights serves every subset of conditioning at sampling time.
    cond_values = build_cond_values(batch, model.conditioner)
    cond_masks = model.conditioner.sample_masks(
        n_graphs,
        device,
        cond_dropout,
        available=available_modalities(batch, model.conditioner),
    )

    out = model(
        frac=f_t,
        lattice=lattice_t,
        lattice_vec6=x_t,
        atomic_numbers=batch["atomic_numbers"],
        natoms=natoms,
        node_graph_id=batch["node_graph_id"],
        t=t,
        cond_values=cond_values,
        cond_masks=cond_masks,
    )
    loss_lat = torch.nn.functional.mse_loss(out["eps_lattice"], eps_lat)
    loss_frac = torch.nn.functional.mse_loss(out["eps_frac"], target_frac)
    return {
        "loss": lattice_weight * loss_lat + frac_weight * loss_frac,
        "loss_lattice": loss_lat.detach(),
        "loss_frac": loss_frac.detach(),
    }


@torch.no_grad()
def sigma_bucket_report(
    model: ALIGNNCSP,
    schedule: DiffusionSchedule,
    normalizer: Normalizer,
    batch: Dict,
    buckets=((1, 300), (300, 600), (600, 800), (800, 1000)),
) -> str:
    """Coordinate loss against the predict-zero baseline, per noise level.

    The aggregate coordinate loss is a poor progress signal: most of it comes
    from small-sigma steps whose target is near-unit-variance noise, so a
    model that has learned a lot and one that has learned nothing can look
    almost identical. Comparing to the achievable baseline per bucket makes
    real progress visible.
    """
    device = batch["frac"].device
    n_graphs = int(batch["natoms"].shape[0])
    cond_values = build_cond_values(batch, model.conditioner)
    cond_masks = model.conditioner.full_masks(n_graphs, device)
    scale = schedule.num_steps / 1000.0
    parts = []
    for lo, hi in buckets:
        lo_s, hi_s = max(1, int(lo * scale)), max(2, int(hi * scale))
        torch.manual_seed(99)
        t = torch.randint(lo_s, hi_s, (n_graphs,), device=device)
        x0 = normalizer.norm_lattice(
            lattice_to_vec6(batch["lattice"], batch["natoms"])
        )
        x_t, _ = schedule.noise_lattice(x0, t)
        f_t, tgt = schedule.noise_frac(
            batch["frac"], t[batch["node_graph_id"]]
        )
        lat_t = vec6_to_lattice(
            normalizer.denorm_lattice(x_t), batch["natoms"]
        )
        out = model(
            frac=f_t,
            lattice=lat_t,
            lattice_vec6=x_t,
            atomic_numbers=batch["atomic_numbers"],
            natoms=batch["natoms"],
            node_graph_id=batch["node_graph_id"],
            t=t,
            cond_values=cond_values,
            cond_masks=cond_masks,
        )
        base = float((tgt**2).mean())
        got = float(((out["eps_frac"] - tgt) ** 2).mean())
        parts.append(f"[{lo}-{hi}) {base:.3f}->{got:.3f}")
    return "  ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden-features", type=int, default=256)
    ap.add_argument("--alignn-layers", type=int, default=3)
    ap.add_argument("--gcn-layers", type=int, default=3)
    ap.add_argument("--knn", type=int, default=12)
    ap.add_argument("--num-steps", type=int, default=1000)
    ap.add_argument("--sigma-min", type=float, default=0.005)
    ap.add_argument("--sigma-max", type=float, default=0.5)
    ap.add_argument("--prop-dropout", type=float, default=0.15)
    ap.add_argument("--composition-dropout", type=float, default=0.1)
    ap.add_argument("--lattice-weight", type=float, default=1.0)
    ap.add_argument("--frac-weight", type=float, default=10.0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument(
        "--augment",
        type=int,
        default=1,
        help="random signed-permutation relabelling of the lattice basis",
    )
    ap.add_argument(
        "--init-from",
        default=None,
        help="checkpoint to initialise weights from " "(pretrain -> finetune)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    train_ds = CrystalDataset(
        data_dir / "train.json", augment=bool(args.augment)
    )
    val_ds = CrystalDataset(data_dir / "val.json")
    normalizer = compute_normalizer(train_ds).to(device)
    print(f"train={len(train_ds)} val={len(val_ds)}")
    print("normalizer:", json.dumps(normalizer.to_dict(), indent=None))

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    # Conditioning: the target property plus the element-count vector. Both
    # are dropped independently during training, so the same checkpoint can
    # generate from composition alone, property alone, or both.
    conditioner_spec = {
        "prop": {
            "type": "scalar",
            "mean": normalizer.prop_mean,
            "std": normalizer.prop_std,
        },
        "composition": {"type": "composition"},
    }
    cond_dropout = {
        "prop": args.prop_dropout,
        "composition": args.composition_dropout,
    }

    model = ALIGNNCSP(
        denoiser_config={
            "hidden_features": args.hidden_features,
            "alignn_layers": args.alignn_layers,
            "gcn_layers": args.gcn_layers,
            "knn": args.knn,
            "num_steps": args.num_steps,
        },
        conditioner_spec=conditioner_spec,
    ).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_par / 1e6:.2f}M  modalities: {model.modalities}")

    if args.init_from:
        # Fine-tuning from a pretrained checkpoint. The property conditioner's
        # standardisation buffers must come from *this* dataset, not from the
        # pretraining run (where the property column was a placeholder), so
        # they are restored after loading.
        ckpt = torch.load(
            args.init_from, map_location=device, weights_only=False
        )
        state = ckpt.get("ema") or ckpt["model"]
        keep = {
            k: v.clone()
            for k, v in model.state_dict().items()
            if k.endswith(("mean", "std")) and "conditioner" in k
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        with torch.no_grad():
            for k, v in keep.items():
                model.state_dict()[k].copy_(v)
        print(
            f"initialised from {args.init_from} "
            f"(epoch {ckpt.get('epoch')}); "
            f"missing={len(missing)} unexpected={len(unexpected)}; "
            f"restored {len(keep)} conditioner stat buffers"
        )
        if missing:
            print(f"  missing keys (freshly initialised): {missing[:8]}")

    schedule = DiffusionSchedule(
        num_steps=args.num_steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    ).to(device)
    ema = EMA(model, decay=args.ema_decay)

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=args.lr,
        total_steps=args.epochs * max(1, len(train_dl)),
        pct_start=0.05,
    )

    cfg = vars(args) | {"n_parameters": n_par}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # One fixed batch reused for the per-noise-level report, so the numbers
    # are comparable across epochs.
    report_batch = batch_to(
        collate([val_ds[i] for i in range(len(val_ds))]), device
    )

    best_val = float("inf")
    history = []
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        agg = {"loss": 0.0, "loss_lattice": 0.0, "loss_frac": 0.0}
        nb = 0
        for batch in train_dl:
            batch = batch_to(batch, device)
            losses = diffusion_loss(
                model,
                schedule,
                normalizer,
                batch,
                cond_dropout,
                args.lattice_weight,
                args.frac_weight,
            )
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            ema.update(model)
            for k in agg:
                agg[k] += float(losses[k])
            nb += 1
        for k in agg:
            agg[k] /= max(nb, 1)

        # Validation uses the EMA weights, with the timestep draw fixed so the
        # curve is comparable epoch to epoch rather than dominated by noise.
        val_agg = {"loss": 0.0, "loss_lattice": 0.0, "loss_frac": 0.0}
        nv = 0
        gen_state = torch.random.get_rng_state()
        torch.manual_seed(1234)
        with torch.no_grad():
            for batch in val_dl:
                batch = batch_to(batch, device)
                losses = diffusion_loss(
                    ema.shadow,
                    schedule,
                    normalizer,
                    batch,
                    {k: 0.0 for k in cond_dropout},
                    args.lattice_weight,
                    args.frac_weight,
                )
                for k in val_agg:
                    val_agg[k] += float(losses[k])
                nv += 1
        torch.random.set_rng_state(gen_state)
        for k in val_agg:
            val_agg[k] /= max(nv, 1)

        history.append({"epoch": epoch, "train": agg, "val": val_agg})
        if val_agg["loss"] < best_val:
            best_val = val_agg["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.shadow.state_dict(),
                    "normalizer": normalizer.to_dict(),
                    "conditioner_spec": conditioner_spec,
                    "config": cfg,
                    "epoch": epoch,
                    "val_loss": best_val,
                },
                out_dir / "best_model.pt",
            )
        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"epoch {epoch:5d}  train {agg['loss']:.4f} "
                f"(lat {agg['loss_lattice']:.4f} frac {agg['loss_frac']:.4f})"
                f"  val {val_agg['loss']:.4f} "
                f"(lat {val_agg['loss_lattice']:.4f} "
                f"frac {val_agg['loss_frac']:.4f})"
                f"  best {best_val:.4f}  {time.time() - t_start:.0f}s",
                flush=True,
            )
            print(
                "    frac vs baseline: "
                + sigma_bucket_report(
                    ema.shadow, schedule, normalizer, report_batch
                ),
                flush=True,
            )
            (out_dir / "history.json").write_text(json.dumps(history))

    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.shadow.state_dict(),
            "normalizer": normalizer.to_dict(),
            "conditioner_spec": conditioner_spec,
            "config": cfg,
            "epoch": args.epochs,
        },
        out_dir / "last_model.pt",
    )
    (out_dir / "history.json").write_text(json.dumps(history))
    print(f"done in {time.time() - t_start:.0f}s  best val {best_val:.4f}")


if __name__ == "__main__":
    main()
