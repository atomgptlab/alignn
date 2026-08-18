"""Module for training script."""

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from functools import partial
from typing import Any, Dict, Union
import torch
import random
from sklearn.metrics import mean_absolute_error
import pickle as pk
import numpy as np
from torch import nn
from alignn.data import get_train_val_loaders
from alignn.config import TrainingConfig
from alignn.models.alignn_atomwise import ALIGNNAtomWise
from alignn.models.alignn_atomwise_pure import ALIGNNAtomWisePure
from alignn.models.alignn_atomwise_pure_smooth import (
    ALIGNNAtomWisePureSmooth,
)
from alignn.torch_graph_builder import unbatch as _graph_unbatch
from alignn.models.ealignn_atomwise import eALIGNNAtomWise
from alignn.models.alignn import ALIGNN
from jarvis.db.jsonutils import dumpjson
import json
import pprint
import os
import warnings
import time
from sklearn.metrics import roc_auc_score
from alignn.utils import (
    group_decay,
    setup_optimizer,
    print_train_val_loss,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _ddp_mean(value: float, use_ddp: bool) -> float:
    """All-reduce a Python scalar across ranks and return the mean."""
    if not use_ddp or not dist.is_available() or not dist.is_initialized():
        return float(value)
    t = torch.tensor(float(value), device=torch.cuda.current_device())
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


def _unwrap(net):
    """Return underlying module from a DDP-wrapped model, else net."""
    return net.module if isinstance(net, DDP) else net


def _ema_init(module):
    """Clone a module's state dict as the initial EMA shadow state."""
    return {
        k: v.detach().clone() for k, v in module.state_dict().items()
    }


def _ema_update(ema_state, module, decay):
    """Blend the module's current weights into the EMA shadow state.

    Floating-point tensors are lerped with ``decay``; integer buffers
    (e.g. counters) are copied through unchanged.
    """
    with torch.no_grad():
        msd = module.state_dict()
        for k, v in ema_state.items():
            cur = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(cur, alpha=1.0 - decay)
            else:
                v.copy_(cur)


def _graph_val_loss(module, val_loader, config, device, criterion):
    """Graph-level validation loss under the module's current weights.

    Used only for EMA checkpoint selection, so it evaluates the
    graph-level head alone. For pure property models (all other loss
    weights zero) this equals the full validation loss.
    """
    total = 0.0
    with torch.no_grad():
        for dats in val_loader:
            if (config.compute_line_graph) > 0:
                result = module(
                    [
                        dats[0].to(device),
                        dats[1].to(device),
                        dats[2].to(device),
                    ]
                )
            else:
                result = module([dats[0].to(device), dats[1].to(device)])
            loss = config.model.graphwise_weight * criterion(
                result["out"], dats[-1].to(device)
            )
            total += loss.item()
    return total / max(1, len(val_loader))


# torch.autograd.detect_anomaly()

figlet_alignn = """
    _    _     ___ ____ _   _ _   _
   / \  | |   |_ _/ ___| \ | | \ | |
  / _ \ | |    | | |  _|  \| |  \| |
 / ___ \| |___ | | |_| | |\  | |\  |
/_/   \_\_____|___\____|_| \_|_| \_|
"""


def train_dgl(
    config: Union[TrainingConfig, Dict[str, Any]],
    model: nn.Module = None,
    train_val_test_loaders=[],
    rank=0,
    world_size=0,
):
    """Training entry point for DGL networks.

    `config` should conform to alignn.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used
    """
    if rank == 0:
        if type(config) is dict:
            try:
                print("Trying to convert dictionary.")
                config = TrainingConfig(**config)
            except Exception as exp:
                print("Check", exp)
    print("config:", config.dict())

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    classification = False
    is_main = rank == 0
    tmp = config.dict()
    if is_main:
        f = open(os.path.join(config.output_dir, "config.json"), "w")
        f.write(json.dumps(tmp, indent=4))
        f.close()
    global tmp_output_dir
    tmp_output_dir = config.output_dir
    pprint.pprint(tmp)
    if config.classification_threshold is not None:
        classification = True
    TORCH_DTYPES = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat": torch.bfloat16,
    }
    torch.set_default_dtype(TORCH_DTYPES[config.dtype])
    line_graph = False
    if config.compute_line_graph > 0:
        line_graph = True
    if world_size > 1:
        use_ddp = True
    else:
        use_ddp = False
        device = "cpu"
        if torch.cuda.is_available():
            device = torch.device("cuda")
    if not train_val_test_loaders:
        (
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
        ) = get_train_val_loaders(
            dataset=config.dataset,
            target=config.target,
            n_train=config.n_train,
            n_val=config.n_val,
            n_test=config.n_test,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            batch_size=config.batch_size,
            atom_features=config.atom_features,
            neighbor_strategy=config.neighbor_strategy,
            standardize=config.atom_features != "cgcnn",
            line_graph=line_graph,
            id_tag=config.id_tag,
            pin_memory=config.pin_memory,
            workers=config.num_workers,
            save_dataloader=config.save_dataloader,
            use_canonize=config.use_canonize,
            filename=config.filename,
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors,
            three_body_cutoff=config.three_body_cutoff,
            read_existing=config.read_existing,
            output_features=config.model.output_features,
            classification_threshold=config.classification_threshold,
            target_multiplication_factor=config.target_multiplication_factor,
            standard_scalar_and_pca=config.standard_scalar_and_pca,
            keep_data_order=config.keep_data_order,
            output_dir=config.output_dir,
            use_lmdb=config.use_lmdb,
            dtype=config.dtype,
        )
    else:
        train_loader = train_val_test_loaders[0]
        val_loader = train_val_test_loaders[1]
        test_loader = train_val_test_loaders[2]
        prepare_batch = train_val_test_loaders[3]
    if use_ddp:
        # `rank` is GLOBAL: on a multi-node launch it exceeds the per-node
        # device count (and with --gpu-bind=closest each rank sees a single
        # GPU), so cuda:{rank} raises "invalid device ordinal". setup() has
        # already selected this process's device -- read it back.
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    prepare_batch = partial(prepare_batch, device=device)
    if classification:
        config.model.classification = True
    _model = {
        "alignn_atomwise": ALIGNNAtomWise,
        "alignn_atomwise_pure": ALIGNNAtomWisePure,
        "alignn_atomwise_pure_smooth": ALIGNNAtomWisePureSmooth,
        "ealignn_atomwise": eALIGNNAtomWise,
        "alignn": ALIGNN,
    }
    # torch_seed decouples training stochasticity (init, shuffling)
    # from random_seed, which also fixes the train/val/test split.
    # Seed replicas for ensembling vary torch_seed ONLY, so every
    # replica sees the identical leaderboard split.
    train_seed = config.random_seed
    if getattr(config, "torch_seed", None) is not None:
        train_seed = config.torch_seed
    if train_seed is not None:
        random.seed(train_seed)
        torch.manual_seed(train_seed)
        np.random.seed(train_seed)
        torch.cuda.manual_seed_all(train_seed)
        try:
            import torch_xla.core.xla_model as xm

            xm.set_rng_state(train_seed)
        except ImportError:
            pass
        os.environ["PYTHONHASHSEED"] = str(train_seed)
        if getattr(config, "deterministic", False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(":4096:8")
            torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.benchmark = True
    if model is None:
        net = _model.get(config.model.name)(config.model)
    else:
        net = model
    print(figlet_alignn)
    print("Model parameters", sum(p.numel() for p in net.parameters()))
    print("CUDA available", torch.cuda.is_available())
    print("CUDA device count", int(torch.cuda.device_count()))
    try:
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        from platform import system as platform_system

        platform_system = platform_system()
        statistics = (
            f"   GPU: {gpu_stats.name}. Max memory: {max_memory} GB"
            + f". Platform = {platform_system}.\n"
            f"   Pytorch: {torch.__version__}. CUDA = "
            + f"{gpu_stats.major}.{gpu_stats.minor}."
            + f" CUDA Toolkit = {torch.version.cuda}.\n"
        )
        print(statistics)
    except Exception:
        pass
    net.to(device)
    if use_ddp:
        net = DDP(
            net,
            # local device index, not the global rank (see `device` above)
            device_ids=[torch.cuda.current_device()],
            find_unused_parameters=bool(
                getattr(config, "ddp_find_unused_parameters", False)
            ),
        )
    # ------------------------------------------------------------------
    # Build optimizer ONCE and bind the scheduler to it.
    # (Previously this block created the optimizer twice — once here and
    #  once again inside the `if "alignn_" in config.model.name:` block —
    #  which left the scheduler bound to a stale optimizer that never
    #  stepped, triggering the
    #  "lr_scheduler.step() before optimizer.step()" warning and producing
    #  a broken LR schedule.)
    # ------------------------------------------------------------------
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)

    if config.scheduler == "none":
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )
    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            # Schedule spans the whole run; lr_total_epochs lets a resumed
            # chain keep one continuous cycle (falls back to epochs).
            epochs=(config.lr_total_epochs or config.epochs),
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
        )

    # OneCycleLR expects scheduler.step() per batch; other schedulers we
    # step per epoch.
    _step_scheduler_per_batch = config.scheduler == "onecycle"

    if "alignn_" in config.model.name:
        best_loss = np.inf
        best_ema_loss = np.inf
        ema_state = None
        if getattr(config, "use_ema", False):
            ema_state = _ema_init(_unwrap(net))
        # Honor config.criterion (historically this was hardcoded to L1Loss
        # regardless of the config value; every run before 2026-08-18 trained
        # with L1). "wmse" is target-intensity-weighted MSE: bins carrying
        # spectral weight are up-weighted to counter amplitude damping on
        # sparse multi-output targets.
        def _wmse(pred, target, _alpha=10.0):
            w = 1.0 + _alpha * target.detach().abs()
            return (w * (pred - target) ** 2).mean()

        criterion = {
            "l1": nn.L1Loss(),
            "mse": nn.MSELoss(),
            "wmse": _wmse,
        }.get(config.criterion, nn.L1Loss())
        if classification:
            criterion = nn.NLLLoss()
        # NOTE: optimizer / scheduler intentionally NOT recreated here.
        history_train = []
        history_val = []
        # Resume optimizer/scheduler/epoch (weights are restored separately
        # via --restart_model_path). current_state.pt is written each epoch
        # below, alongside the pure-weights current_model.pt.
        start_epoch = 0
        if config.resume_checkpoint:
            state_path = os.path.join(config.output_dir, "current_state.pt")
            if os.path.exists(state_path):
                ckpt = torch.load(state_path, map_location=device)
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                best_loss = ckpt.get("best_loss", best_loss)
                start_epoch = ckpt["epoch"]
                if rank == 0:
                    print(
                        "Resuming from epoch",
                        start_epoch,
                        "best_loss",
                        best_loss,
                    )
        for e in range(start_epoch, config.epochs):
            train_init_time = time.time()
            running_loss = 0
            running_loss1 = 0
            running_loss2 = 0
            running_loss3 = 0
            running_loss4 = 0
            running_loss5 = 0
            train_result = []
            for dats, jid in zip(train_loader, train_loader.dataset.ids):
                info = {}
                optimizer.zero_grad()
                _amp_ctx = torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=bool(getattr(config, "use_amp", False))
                    and torch.cuda.is_available(),
                )
                with _amp_ctx:
                    if (config.compute_line_graph) > 0:
                        result = net(
                            [
                                dats[0].to(device),
                                dats[1].to(device),
                                dats[2].to(device),
                            ]
                        )
                    else:
                        result = net([dats[0].to(device), dats[1].to(device)])
                info["target_out"] = []
                info["pred_out"] = []
                info["target_atomwise_pred"] = []
                info["pred_atomwise_pred"] = []
                info["target_grad"] = []
                info["pred_grad"] = []
                info["target_stress"] = []
                info["pred_stress"] = []
                info["target_additional"] = []
                info["pred_additional"] = []

                loss1 = 0  # Such as energy
                loss2 = 0  # Such as bader charges
                loss3 = 0  # Such as forces
                loss4 = 0  # Such as stresses
                loss5 = 0  # Such as dos
                if config.model.output_features is not None:
                    loss1 = config.model.graphwise_weight * criterion(
                        result["out"],
                        dats[-1].to(device),
                    )
                    info["target_out"] = dats[-1].cpu().numpy().tolist()
                    info["pred_out"] = (
                        result["out"].cpu().detach().numpy().tolist()
                    )
                    running_loss1 += loss1.item()
                if (
                    config.model.atomwise_output_features > 0
                    and config.model.atomwise_weight != 0
                ):
                    loss2 = config.model.atomwise_weight * criterion(
                        result["atomwise_pred"].to(device),
                        dats[0].ndata["atomwise_target"].to(device),
                    )
                    info["target_atomwise_pred"] = (
                        dats[0].ndata["atomwise_target"].cpu().numpy().tolist()
                    )
                    info["pred_atomwise_pred"] = (
                        result["atomwise_pred"].cpu().detach().numpy().tolist()
                    )
                    running_loss2 += loss2.item()

                if config.model.calculate_gradient:
                    loss3 = config.model.gradwise_weight * criterion(
                        result["grad"].to(device),
                        dats[0].ndata["atomwise_grad"].to(device),
                    )
                    info["target_grad"] = (
                        dats[0].ndata["atomwise_grad"].cpu().numpy().tolist()
                    )
                    info["pred_grad"] = (
                        result["grad"].cpu().detach().numpy().tolist()
                    )
                    running_loss3 += loss3.item()
                if config.model.stresswise_weight != 0:
                    targ_stress = torch.stack(
                        [
                            gg.ndata["stresses"][0]
                            for gg in _graph_unbatch(dats[0])
                        ]
                    ).to(device)
                    pred_stress = result["stresses"]
                    loss4 = config.model.stresswise_weight * criterion(
                        pred_stress.to(device),
                        targ_stress.to(device),
                    )
                    info["target_stress"] = targ_stress.cpu().numpy().tolist()
                    info["pred_stress"] = (
                        result["stresses"].cpu().detach().numpy().tolist()
                    )
                    running_loss4 += loss4.item()
                if config.model.additional_output_weight != 0:
                    additional_dat = [
                        gg.ndata["additional"][0]
                        for gg in _graph_unbatch(dats[0])
                    ]
                    targ = torch.stack(additional_dat).to(device)
                    loss5 = config.model.additional_output_weight * criterion(
                        (result["additional"]).to(device),
                        targ,
                    )
                    info["target_additional"] = targ.cpu().numpy().tolist()
                    info["pred_additional"] = (
                        result["additional"].cpu().detach().numpy().tolist()
                    )
                    running_loss5 += loss5.item()
                train_result.append(info)
                loss = loss1 + loss2 + loss3 + loss4 + loss5
                loss.backward()
                optimizer.step()
                if ema_state is not None:
                    _ema_update(
                        ema_state, _unwrap(net), config.ema_decay
                    )
                # Step OneCycleLR per batch (its design assumption); other
                # schedulers are stepped once per epoch below.
                if _step_scheduler_per_batch:
                    scheduler.step()
                running_loss += loss.item()
            # Normalize running losses by number of batches
            _n_tr = max(1, len(train_loader))
            running_loss /= _n_tr
            running_loss1 /= _n_tr
            running_loss2 /= _n_tr
            running_loss3 /= _n_tr
            running_loss4 /= _n_tr
            running_loss5 /= _n_tr
            running_loss = _ddp_mean(running_loss, use_ddp)
            running_loss1 = _ddp_mean(running_loss1, use_ddp)
            running_loss2 = _ddp_mean(running_loss2, use_ddp)
            running_loss3 = _ddp_mean(running_loss3, use_ddp)
            running_loss4 = _ddp_mean(running_loss4, use_ddp)
            running_loss5 = _ddp_mean(running_loss5, use_ddp)
            # Epoch-level scheduler step for non-OneCycle schedulers.
            if not _step_scheduler_per_batch:
                scheduler.step()
            train_final_time = time.time()
            train_ep_time = train_final_time - train_init_time
            history_train.append(
                [
                    running_loss,
                    running_loss1,
                    running_loss2,
                    running_loss3,
                    running_loss4,
                    running_loss5,
                ]
            )
            if is_main:
                dumpjson(
                    filename=os.path.join(
                        config.output_dir, "history_train.json"
                    ),
                    data=history_train,
                )
            val_loss = 0
            val_loss1 = 0
            val_loss2 = 0
            val_loss3 = 0
            val_loss4 = 0
            val_loss5 = 0
            val_result = []
            val_init_time = time.time()
            for dats, jid in zip(val_loader, val_loader.dataset.ids):
                info = {}
                info["id"] = jid
                optimizer.zero_grad()
                if (config.compute_line_graph) > 0:
                    result = net(
                        [
                            dats[0].to(device),
                            dats[1].to(device),
                            dats[2].to(device),
                        ]
                    )
                else:
                    result = net([dats[0].to(device), dats[1].to(device)])
                info["target_out"] = []
                info["pred_out"] = []
                info["target_atomwise_pred"] = []
                info["pred_atomwise_pred"] = []
                info["target_grad"] = []
                info["pred_grad"] = []
                info["target_stress"] = []
                info["pred_stress"] = []
                loss1 = 0
                loss2 = 0
                loss3 = 0
                loss4 = 0
                loss5 = 0
                if config.model.output_features is not None:
                    loss1 = config.model.graphwise_weight * criterion(
                        result["out"], dats[-1].to(device)
                    )
                    info["target_out"] = dats[-1].cpu().numpy().tolist()
                    info["pred_out"] = (
                        result["out"].cpu().detach().numpy().tolist()
                    )
                    val_loss1 += loss1.item()

                if (
                    config.model.atomwise_output_features > 0
                    and config.model.atomwise_weight != 0
                ):
                    loss2 = config.model.atomwise_weight * criterion(
                        result["atomwise_pred"].to(device),
                        dats[0].ndata["atomwise_target"].to(device),
                    )
                    info["target_atomwise_pred"] = (
                        dats[0].ndata["atomwise_target"].cpu().numpy().tolist()
                    )
                    info["pred_atomwise_pred"] = (
                        result["atomwise_pred"].cpu().detach().numpy().tolist()
                    )
                    val_loss2 += loss2.item()
                if config.model.calculate_gradient:
                    loss3 = config.model.gradwise_weight * criterion(
                        result["grad"].to(device),
                        dats[0].ndata["atomwise_grad"].to(device),
                    )
                    info["target_grad"] = (
                        dats[0].ndata["atomwise_grad"].cpu().numpy().tolist()
                    )
                    info["pred_grad"] = (
                        result["grad"].cpu().detach().numpy().tolist()
                    )
                    val_loss3 += loss3.item()
                if config.model.stresswise_weight != 0:
                    targ_stress = torch.stack(
                        [
                            gg.ndata["stresses"][0]
                            for gg in _graph_unbatch(dats[0])
                        ]
                    ).to(device)
                    pred_stress = result["stresses"]
                    loss4 = config.model.stresswise_weight * criterion(
                        pred_stress.to(device),
                        targ_stress.to(device),
                    )
                    info["target_stress"] = targ_stress.cpu().numpy().tolist()
                    info["pred_stress"] = (
                        result["stresses"].cpu().detach().numpy().tolist()
                    )

                    val_loss4 += loss4.item()
                if config.model.additional_output_weight != 0:
                    additional_dat = [
                        gg.ndata["additional"][0]
                        for gg in _graph_unbatch(dats[0])
                    ]
                    targ = torch.stack(additional_dat).to(device)
                    loss5 = config.model.additional_output_weight * criterion(
                        (result["additional"]).to(device),
                        targ,
                    )
                    info["target_additional"] = targ.cpu().numpy().tolist()
                    info["pred_additional"] = (
                        result["additional"].cpu().detach().numpy().tolist()
                    )

                    val_loss5 += loss5.item()
                loss = loss1 + loss2 + loss3 + loss4 + loss5
                val_result.append(info)
                val_loss += loss.item()
            _n_vl = max(1, len(val_loader))
            val_loss /= _n_vl
            val_loss1 /= _n_vl
            val_loss2 /= _n_vl
            val_loss3 /= _n_vl
            val_loss4 /= _n_vl
            val_loss5 /= _n_vl
            val_loss = _ddp_mean(val_loss, use_ddp)
            val_loss1 = _ddp_mean(val_loss1, use_ddp)
            val_loss2 = _ddp_mean(val_loss2, use_ddp)
            val_loss3 = _ddp_mean(val_loss3, use_ddp)
            val_loss4 = _ddp_mean(val_loss4, use_ddp)
            val_loss5 = _ddp_mean(val_loss5, use_ddp)
            val_fin_time = time.time()
            val_ep_time = val_fin_time - val_init_time
            current_model_name = "current_model.pt"
            if is_main:
                torch.save(
                    _unwrap(net).state_dict(),
                    os.path.join(config.output_dir, current_model_name),
                )
                # Resume state (optimizer/scheduler/next-epoch/best_loss),
                # kept separate so current_model.pt stays a pure state_dict.
                torch.save(
                    {
                        "epoch": e + 1,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_loss": best_loss,
                    },
                    os.path.join(config.output_dir, "current_state.pt"),
                )
            saving_msg = ""
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_name = "best_model.pt"
                if is_main:
                    torch.save(
                        _unwrap(net).state_dict(),
                        os.path.join(config.output_dir, best_model_name),
                    )
                    saving_msg = "Saving model"
                    dumpjson(
                        filename=os.path.join(
                            config.output_dir, "Train_results.json"
                        ),
                        data=train_result,
                    )
                    dumpjson(
                        filename=os.path.join(
                            config.output_dir, "Val_results.json"
                        ),
                        data=val_result,
                    )
                best_model = net
            if ema_state is not None and is_main:
                module = _unwrap(net)
                backup = {
                    k: v.detach().clone()
                    for k, v in module.state_dict().items()
                }
                module.load_state_dict(ema_state)
                ema_val = _graph_val_loss(
                    module, val_loader, config, device, criterion
                )
                module.load_state_dict(backup)
                if ema_val < best_ema_loss:
                    best_ema_loss = ema_val
                    torch.save(
                        ema_state,
                        os.path.join(
                            config.output_dir, "best_ema_model.pt"
                        ),
                    )
                    saving_msg += " Saving EMA (val %.6f)" % ema_val
            history_val.append(
                [
                    val_loss,
                    val_loss1,
                    val_loss2,
                    val_loss3,
                    val_loss4,
                    val_loss5,
                ]
            )
            if is_main:
                dumpjson(
                    filename=os.path.join(
                        config.output_dir, "history_val.json"
                    ),
                    data=history_val,
                )
            if rank == 0:
                print_train_val_loss(
                    e,
                    running_loss,
                    running_loss1,
                    running_loss2,
                    running_loss3,
                    running_loss4,
                    running_loss5,
                    val_loss,
                    val_loss1,
                    val_loss2,
                    val_loss3,
                    val_loss4,
                    val_loss5,
                    train_ep_time,
                    val_ep_time,
                    saving_msg=saving_msg,
                )

        if rank == 0 or world_size == 1:
            # This block runs on rank 0 only. Forward through the underlying
            # module, not the DDP wrapper: a DDP forward performs collectives
            # (buffer broadcast) that the other ranks never reach here, which
            # deadlocks rank 0 until the NCCL watchdog times out.
            if getattr(config, "eval_best_checkpoint", False):
                # Evaluate the best-validation weights rather than the
                # last epoch's. best_model = net below is an alias, so
                # loading here also covers the prediction-writing blocks.
                cand_path = os.path.join(
                    config.output_dir, "best_model.pt"
                )
                cand_loss = best_loss
                ema_path = os.path.join(
                    config.output_dir, "best_ema_model.pt"
                )
                if os.path.exists(ema_path) and best_ema_loss < cand_loss:
                    cand_path, cand_loss = ema_path, best_ema_loss
                if os.path.exists(cand_path):
                    _unwrap(net).load_state_dict(
                        torch.load(cand_path, map_location=device)
                    )
                    print(
                        "Loaded",
                        os.path.basename(cand_path),
                        "for final evaluation",
                        "(val loss %.6f)" % cand_loss,
                    )
            eval_net = _unwrap(net)
            test_loss = 0
            test_result = []
            for dats, jid in zip(test_loader, test_loader.dataset.ids):
                info = {}
                info["id"] = jid
                optimizer.zero_grad()
                if (config.compute_line_graph) > 0:
                    result = eval_net(
                        [
                            dats[0].to(device),
                            dats[1].to(device),
                            dats[2].to(device),
                        ]
                    )
                else:
                    result = eval_net(
                        [dats[0].to(device), dats[1].to(device)]
                    )
                loss1 = 0
                loss2 = 0
                loss3 = 0
                loss4 = 0
                if (
                    config.model.output_features is not None
                    and not classification
                ):
                    loss1 = config.model.graphwise_weight * criterion(
                        result["out"], dats[-1].to(device)
                    )
                    info["target_out"] = dats[-1].cpu().numpy().tolist()
                    info["pred_out"] = (
                        result["out"].cpu().detach().numpy().tolist()
                    )

                if config.model.atomwise_output_features > 0:
                    loss2 = config.model.atomwise_weight * criterion(
                        result["atomwise_pred"].to(device),
                        dats[0].ndata["atomwise_target"].to(device),
                    )
                    info["target_atomwise_pred"] = (
                        dats[0].ndata["atomwise_target"].cpu().numpy().tolist()
                    )
                    info["pred_atomwise_pred"] = (
                        result["atomwise_pred"].cpu().detach().numpy().tolist()
                    )

                if config.model.calculate_gradient:
                    loss3 = config.model.gradwise_weight * criterion(
                        result["grad"].to(device),
                        dats[0].ndata["atomwise_grad"].to(device),
                    )
                    info["target_grad"] = (
                        dats[0].ndata["atomwise_grad"].cpu().numpy().tolist()
                    )
                    info["pred_grad"] = (
                        result["grad"].cpu().detach().numpy().tolist()
                    )
                if config.model.stresswise_weight != 0:

                    targ_stress = torch.stack(
                        [
                            gg.ndata["stresses"][0]
                            for gg in _graph_unbatch(dats[0])
                        ]
                    ).to(device)
                    pred_stress = result["stresses"]
                    loss4 = config.model.stresswise_weight * criterion(
                        pred_stress.to(device),
                        targ_stress.to(device),
                    )
                    info["target_stress"] = targ_stress.cpu().numpy().tolist()
                    info["pred_stress"] = (
                        result["stresses"].cpu().detach().numpy().tolist()
                    )

                test_result.append(info)
                loss = loss1 + loss2 + loss3 + loss4
                if not classification:
                    test_loss += loss.item()
            if is_main:
                print("TestLoss", e, test_loss)
                dumpjson(
                    filename=os.path.join(
                        config.output_dir, "Test_results.json"
                    ),
                    data=test_result,
                )
                last_model_name = "last_model.pt"
                torch.save(
                    _unwrap(net).state_dict(),
                    os.path.join(config.output_dir, last_model_name),
                )
    if rank == 0 or world_size == 1:
        if config.write_predictions and classification:
            best_model.eval()
            f = open(
                os.path.join(
                    config.output_dir, "prediction_results_test_set.csv"
                ),
                "w",
            )
            f.write("id,target,prediction\n")
            targets = []
            predictions = []
            with torch.no_grad():
                ids = test_loader.dataset.ids
                for dat, id in zip(test_loader, ids):
                    g, lg, lat, target = dat
                    out_data = best_model(
                        [g.to(device), lg.to(device), lat.to(device)]
                    )["out"]
                    top_p, top_class = torch.topk(torch.exp(out_data), k=1)
                    target = int(target.cpu().numpy().flatten().tolist()[0])

                    f.write("%s, %d, %d\n" % (id, (target), (top_class)))
                    targets.append(target)
                    predictions.append(
                        top_class.cpu().numpy().flatten().tolist()[0]
                    )
            f.close()

            print("predictions", predictions)
            print("targets", targets)
            print(
                "Test ROCAUC:",
                roc_auc_score(np.array(targets), np.array(predictions)),
            )

        if (
            config.write_predictions
            and not classification
            and config.model.output_features > 1
        ):
            best_model.eval()
            mem = []
            with torch.no_grad():
                ids = test_loader.dataset.ids
                for dat, id in zip(test_loader, ids):
                    g, lg, lat, target = dat
                    out_data = best_model(
                        [g.to(device), lg.to(device), lat.to(device)]
                    )["out"]
                    out_data = (
                        out_data.detach().cpu().numpy().flatten().tolist()
                    )
                    if config.standard_scalar_and_pca:
                        sc = pk.load(open("sc.pkl", "rb"))
                        out_data = list(
                            sc.transform(np.array(out_data).reshape(1, -1))[0]
                        )
                    target = target.cpu().numpy().flatten().tolist()
                    info = {}
                    info["id"] = id
                    info["target"] = target
                    info["predictions"] = out_data
                    mem.append(info)
            dumpjson(
                filename=os.path.join(
                    config.output_dir, "multi_out_predictions.json"
                ),
                data=mem,
            )
        if (
            config.write_predictions
            and not classification
            and config.model.output_features == 1
            and config.model.gradwise_weight == 0
        ):
            best_model.eval()
            f = open(
                os.path.join(
                    config.output_dir, "prediction_results_test_set.csv"
                ),
                "w",
            )
            f.write("id,target,prediction\n")
            targets = []
            predictions = []
            with torch.no_grad():
                ids = test_loader.dataset.ids
                for dat, id in zip(test_loader, ids):
                    g, lg, lat, target = dat
                    out_data = best_model(
                        [g.to(device), lg.to(device), lat.to(device)]
                    )["out"]
                    out_data = out_data.cpu().numpy().tolist()
                    if config.standard_scalar_and_pca:
                        sc = pk.load(
                            open(os.path.join(tmp_output_dir, "sc.pkl"), "rb")
                        )
                        out_data = sc.transform(
                            np.array(out_data).reshape(-1, 1)
                        )[0][0]
                    target = target.cpu().numpy().flatten().tolist()
                    if len(target) == 1:
                        target = target[0]
                    # out_data is a (possibly nested) list after .tolist();
                    # unwrap single-output regression to a scalar, mirroring
                    # the target handling above. Multi-output falls back to
                    # a generic format so the write never raises.
                    if isinstance(out_data, list):
                        out_data = np.array(out_data).flatten().tolist()
                        if len(out_data) == 1:
                            out_data = out_data[0]
                    if isinstance(target, list) or isinstance(out_data, list):
                        f.write("%s, %s, %s\n" % (id, target, out_data))
                    else:
                        f.write("%s, %6f, %6f\n" % (id, target, out_data))
                    targets.append(target)
                    predictions.append(out_data)
            f.close()

            print(
                "Test MAE:",
                mean_absolute_error(np.array(targets), np.array(predictions)),
            )
            best_model.eval()
            f = open(
                os.path.join(
                    config.output_dir, "prediction_results_train_set.csv"
                ),
                "w",
            )
            f.write("target,prediction\n")
            targets = []
            predictions = []
            with torch.no_grad():
                ids = train_loader.dataset.ids
                for dat, id in zip(train_loader, ids):
                    g, lg, lat, target = dat
                    out_data = best_model(
                        [g.to(device), lg.to(device), lat.to(device)]
                    )["out"]
                    out_data = out_data.cpu().numpy().tolist()
                    if config.standard_scalar_and_pca:
                        sc = pk.load(
                            open(os.path.join(tmp_output_dir, "sc.pkl"), "rb")
                        )
                        out_data = sc.transform(
                            np.array(out_data).reshape(-1, 1)
                        )[0][0]
                    target = target.cpu().numpy().flatten().tolist()
                    for ii, jj in zip(target, out_data):
                        f.write("%6f, %6f\n" % (ii, jj))
                        targets.append(ii)
                        predictions.append(jj)
            f.close()
        if config.use_lmdb:
            print("Closing LMDB.")
            train_loader.dataset.close()
            val_loader.dataset.close()
            test_loader.dataset.close()


if __name__ == "__main__":
    config = TrainingConfig(
        random_seed=123, epochs=10, n_train=32, n_val=32, batch_size=16
    )
    history = train_dgl(config)
