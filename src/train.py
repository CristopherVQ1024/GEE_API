"""
train.py
========
Script principal de entrenamiento con Spatial K-Fold cross-validation.

Flujo esperado:
  1. Ya exportaste tus parches (imágenes 8-banda + máscaras) desde GEE y
     los descargaste a data/processed/images y data/processed/masks
     (ver src/gee_exporter.py y src/labels.py).
  2. Generaste un CSV de metadatos con columnas:
       patch_id, lon_center, lat_center
     (uno por cada parche exportado — el centro de cada tile lo puedes
     sacar de la lista `tiles` que devuelve build_export_grid()).
  3. Corriste src/spatial_split.py sobre ese CSV para añadir `fold` y
     guardaste el resultado, p.ej. en data/processed/patches_metadata.csv

Uso:
    python -m src.train --metadata data/processed/patches_metadata.csv

El fold 'external_holdout' NUNCA se usa aquí — se reserva para
evaluate.py (validación empírica final, sección "Resultados" del artículo).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import SelectiveLoggingDataset, build_paths_from_ids, compute_normalization_stats
from src.models.unet import build_unet
from src.utils.losses import BCEDiceLoss
from src.utils.metrics import RunningConfusionCounts

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flip_rot90(image, mask):
    """Aumento geométrico (flips + rotaciones de 90°): no altera los valores espectrales."""
    s = min(image.shape[0], image.shape[1])  # los parches miden 256x257: recorte cuadrado para rotar
    image, mask = image[:s, :s], mask[:s, :s]
    k = np.random.randint(4)
    image, mask = np.rot90(image, k, axes=(0, 1)), np.rot90(mask, k, axes=(0, 1))
    if np.random.rand() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    return {"image": np.ascontiguousarray(image), "mask": np.ascontiguousarray(mask)}


def train_one_fold(cfg: dict, metadata: pd.DataFrame, val_fold: int, device: torch.device) -> dict:
    """
    Entrena con todos los folds numéricos != val_fold (excluyendo siempre
    'external_holdout') y valida en val_fold. Devuelve las métricas de
    validación y la ruta del mejor checkpoint para este fold.
    """
    cv_pool = metadata[metadata["fold"] != "external_holdout"].copy()
    # El CSV mezcla folds numéricos y 'external_holdout' -> se lee como texto.
    fold_str = cv_pool["fold"].astype(str)
    train_df = cv_pool[fold_str != str(val_fold)]
    val_df = cv_pool[fold_str == str(val_fold)]

    images_dir = cfg["paths"]["patches_images"]
    masks_dir = cfg["paths"]["patches_masks"]

    def paths(df):
        # Conjuntos combinados (p. ej. Madre de Dios + DETER) traen rutas por parche en el CSV
        if "image_path" in df.columns and "mask_path" in df.columns:
            return df["image_path"].tolist(), df["mask_path"].tolist()
        return build_paths_from_ids(df["patch_id"].tolist(), images_dir, masks_dir)

    train_img_paths, train_mask_paths = paths(train_df)
    val_img_paths, val_mask_paths = paths(val_df)

    norm_stats = compute_normalization_stats(train_img_paths)

    train_ds = SelectiveLoggingDataset(train_img_paths, train_mask_paths, normalize_stats=norm_stats,
                                       transform=flip_rot90 if cfg["model"]["training"].get("augment") else None)
    val_ds = SelectiveLoggingDataset(val_img_paths, val_mask_paths, normalize_stats=norm_stats)

    train_cfg = cfg["model"]["training"]
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
                               num_workers=train_cfg["num_workers"], drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
                             num_workers=train_cfg["num_workers"])

    model_cfg = cfg["model"]
    model = build_unet(
        encoder_name=model_cfg["encoder"],
        encoder_weights=model_cfg["encoder_weights"],
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["classes"],
    ).to(device)

    pos_weight = model_cfg["loss"].get("pos_weight")
    loss_fn = BCEDiceLoss(
        bce_weight=model_cfg["loss"]["bce_weight"],
        dice_weight=model_cfg["loss"]["dice_weight"],
        pos_weight=torch.tensor([float(pos_weight)], device=device) if pos_weight else None,
    )

    opt_cfg = model_cfg["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])

    sched_cfg = model_cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=sched_cfg["factor"], patience=sched_cfg["patience"]
    )

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / f"unet_fold{val_fold}_best.pt"

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(train_cfg["epochs"]):
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_ds)

        model.eval()
        val_loss_total = 0.0
        val_counts = RunningConfusionCounts()
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = loss_fn(logits, masks)
                val_loss_total += loss.item() * imgs.size(0)
                val_counts.update(torch.sigmoid(logits), masks)
        val_loss = val_loss_total / max(len(val_ds), 1)
        val_metrics = val_counts.compute()

        scheduler.step(val_loss)

        print(
            f"[fold {val_fold}] epoch {epoch + 1}/{train_cfg['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"IoU={val_metrics['iou']:.4f} F1={val_metrics['f1']:.4f} "
            f"Prec={val_metrics['precision']:.4f} Rec={val_metrics['recall']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "norm_stats": norm_stats,
                "val_metrics": val_metrics,
                "epoch": epoch,
            }, best_ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
                print(f"[fold {val_fold}] Early stopping en epoch {epoch + 1}")
                break

    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    return {"fold": val_fold, "checkpoint": str(best_ckpt_path), **best_ckpt["val_metrics"]}


def load_to_device(img_paths: list[str], mask_paths: list[str], norm_stats: dict, device: torch.device):
    """Carga los parches ya normalizados a la GPU: imágenes en fp16 y máscaras en uint8 (0/1/255)."""
    ds = SelectiveLoggingDataset(img_paths, mask_paths, normalize_stats=norm_stats)
    x0, _ = ds[0]
    imgs = torch.empty((len(ds), *x0.shape), dtype=torch.float16, device=device)
    masks = torch.empty((len(ds), 1, *x0.shape[1:]), dtype=torch.uint8, device=device)
    for i in range(len(ds)):
        x, y = ds[i]
        imgs[i] = x.to(device, torch.float16)
        masks[i] = y.to(device, torch.uint8)
    return imgs, masks


def gpu_flip_rot90(x: torch.Tensor, y: torch.Tensor):
    """flip_rot90 por muestra, en la GPU."""
    xs, ys = [], []
    for xi, yi in zip(x, y):
        k = int(torch.randint(4, (1,)))
        xi, yi = torch.rot90(xi, k, (1, 2)), torch.rot90(yi, k, (1, 2))
        if torch.rand(1).item() < 0.5:
            xi, yi = xi.flip(2), yi.flip(2)
        xs.append(xi)
        ys.append(yi)
    return torch.stack(xs), torch.stack(ys)


def train_one_fold_balanced(cfg: dict, metadata: pd.DataFrame, val_fold: int, device: torch.device,
                            mdd_frac: float) -> dict:
    """
    Igual que train_one_fold, pero con los parches en la GPU y lotes que muestrean
    Madre de Dios (source == 'mdd') con probabilidad mdd_frac. La validación se mide
    por origen y el checkpoint se elige con la media de la pérdida MDD y DETER.
    """
    cv_pool = metadata[metadata["fold"] != "external_holdout"].copy()
    fold_str = cv_pool["fold"].astype(str)
    train_df = cv_pool[fold_str != str(val_fold)].reset_index(drop=True)
    val_df = cv_pool[fold_str == str(val_fold)].reset_index(drop=True)

    init = cfg["model"]["training"].get("init_ckpt")
    if init:
        # afinado desde un modelo previo del mismo fold, con su misma normalización
        init = torch.load(Path(init) / f"unet_fold{val_fold}_best.pt", map_location=device, weights_only=False)
        norm_stats = init["norm_stats"]
    else:
        norm_stats = compute_normalization_stats(train_df["image_path"].tolist())
    train_x, train_y = load_to_device(train_df["image_path"].tolist(), train_df["mask_path"].tolist(), norm_stats, device)
    is_mdd = torch.tensor((train_df["source"] == "mdd").to_numpy(), device=device)
    n_mdd = int(is_mdd.sum())
    weights = torch.where(is_mdd, mdd_frac / n_mdd, (1 - mdd_frac) / (len(train_df) - n_mdd)).double()

    val_sets = {}
    for name, part in (("mdd", val_df[val_df["source"] == "mdd"]), ("deter", val_df[val_df["source"] != "mdd"])):
        val_sets[name] = load_to_device(part["image_path"].tolist(), part["mask_path"].tolist(), norm_stats, device)
    print(f"[fold {val_fold}] train {len(train_df)} ({n_mdd} MDD) | val MDD {len(val_sets['mdd'][0])} "
          f"DETER {len(val_sets['deter'][0])} | VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GB")

    model_cfg = cfg["model"]
    train_cfg = model_cfg["training"]
    model = build_unet(
        encoder_name=model_cfg["encoder"],
        encoder_weights=model_cfg["encoder_weights"],
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["classes"],
    ).to(device)
    if init:
        model.load_state_dict(init["model_state_dict"])

    pos_weight = model_cfg["loss"].get("pos_weight")
    loss_fn = BCEDiceLoss(
        bce_weight=model_cfg["loss"]["bce_weight"],
        dice_weight=model_cfg["loss"]["dice_weight"],
        pos_weight=torch.tensor([float(pos_weight)], device=device) if pos_weight else None,
    )
    opt_cfg = model_cfg["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    sched_cfg = model_cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=sched_cfg["factor"], patience=sched_cfg["patience"]
    )

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / f"unet_fold{val_fold}_best.pt"

    bs = train_cfg["batch_size"]
    n_batches = len(train_df) // bs
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(train_cfg["epochs"]):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        for _ in range(n_batches):
            idx = torch.multinomial(weights, bs, replacement=True)
            imgs, masks = train_x[idx].float(), train_y[idx].float()
            if train_cfg.get("augment"):
                imgs, masks = gpu_flip_rot90(imgs, masks)
            optimizer.zero_grad()
            loss = loss_fn(model(imgs), masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / n_batches

        model.eval()
        per_source = {}
        all_counts = RunningConfusionCounts()
        with torch.no_grad():
            for name, (vx, vy) in val_sets.items():
                counts = RunningConfusionCounts()
                total = 0.0
                for i in range(0, len(vx), bs):
                    imgs, masks = vx[i:i + bs].float(), vy[i:i + bs].float()
                    logits = model(imgs)
                    total += loss_fn(logits, masks).item() * imgs.size(0)
                    probs = torch.sigmoid(logits)
                    counts.update(probs, masks)
                    all_counts.update(probs, masks)
                per_source[name] = {"loss": total / max(len(vx), 1), **counts.compute()}
        val_loss = (per_source["mdd"]["loss"] + per_source["deter"]["loss"]) / 2
        val_metrics = {**all_counts.compute(), "val_loss": val_loss,
                       **{f"{s}_{k}": v for s, m in per_source.items() for k, v in m.items()}}

        scheduler.step(val_loss)

        print(
            f"[fold {val_fold}] epoch {epoch + 1}/{train_cfg['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"MDD loss={per_source['mdd']['loss']:.4f} IoU={per_source['mdd']['iou']:.4f} | "
            f"DETER loss={per_source['deter']['loss']:.4f} IoU={per_source['deter']['iou']:.4f} "
            f"({time.time() - t0:.0f} s)",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "norm_stats": norm_stats,
                "val_metrics": val_metrics,
                "epoch": epoch,
            }, best_ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
                print(f"[fold {val_fold}] Early stopping en epoch {epoch + 1}")
                break

    print(f"[fold {val_fold}] VRAM pico {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    del train_x, train_y, val_sets
    torch.cuda.empty_cache()
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    return {"fold": val_fold, "checkpoint": str(best_ckpt_path), **best_ckpt["val_metrics"]}


def run_spatial_cross_validation(cfg: dict, metadata_path: str) -> None:
    metadata = pd.read_csv(metadata_path)
    if "noisy" in metadata.columns:  # parches con dNDFI ruidoso (neblina), ver src/refine_labels.py
        print(f"Excluyendo {int(metadata['noisy'].sum())} parches ruidosos")
        metadata = metadata[~metadata["noisy"].astype(bool)]
    device = get_device()
    print(f"Usando device: {device}")

    numeric_folds = sorted(
        [f for f in metadata["fold"].unique() if f != "external_holdout"],
        key=lambda x: int(x),
    )
    if cfg["model"]["training"].get("folds"):
        numeric_folds = [f for f in numeric_folds if str(f) in cfg["model"]["training"]["folds"].split(",")]

    metrics_dir = Path(cfg["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for fold in numeric_folds:
        # Reanudable: un fold terminado deja fold<N>_done.json y se salta al relanzar
        done = metrics_dir / f"fold{fold}_done.json"
        if done.exists():
            print(f"[fold {fold}] ya entrenado, se salta ({done})")
            results.append(json.loads(done.read_text()))
            continue
        mdd_frac = cfg["model"]["training"].get("mdd_frac")
        if mdd_frac:
            result = train_one_fold_balanced(cfg, metadata, val_fold=int(fold), device=device, mdd_frac=mdd_frac)
        else:
            result = train_one_fold(cfg, metadata, val_fold=int(fold), device=device)
        done.write_text(json.dumps(result, ensure_ascii=False, default=float))
        results.append(result)

    df_results = pd.DataFrame(results)
    df_results.to_csv(metrics_dir / "cv_results_per_fold.csv", index=False)

    summary = {
        metric: {"mean": float(np.mean(df_results[metric])), "std": float(np.std(df_results[metric]))}
        for metric in ["iou", "f1", "precision", "recall"]
    }
    with open(metrics_dir / "cv_results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Resumen validación cruzada espacial (n_folds) ===")
    for metric, stats in summary.items():
        print(f"{metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"\nResultados guardados en {metrics_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True,
                         help="CSV con patch_id, lon_center, lat_center, fold (ver src/spatial_split.py)")
    parser.add_argument("--pos_weight", type=float, default=None,
                         help="Peso de la clase positiva en BCE (ej. 10). Sobrescribe model.loss.pos_weight")
    parser.add_argument("--masks_dir", type=str, default=None,
                         help="Carpeta de máscaras alternativa (ej. data/processed/masks_v3)")
    parser.add_argument("--augment", action="store_true", help="Flips + rot90 en entrenamiento")
    parser.add_argument("--epochs", type=int, default=None, help="Sobrescribe model.training.epochs")
    parser.add_argument("--tag", type=str, default=None,
                         help="Guarda checkpoints/métricas en outputs/<tag>/ en vez de outputs/")
    parser.add_argument("--mdd_frac", type=float, default=None,
                         help="Fracción de Madre de Dios por lote (ej. 0.5); carga los parches en la GPU")
    parser.add_argument("--init_ckpt", type=str, default=None,
                         help="Carpeta con unet_fold<N>_best.pt para afinar desde ese modelo (requiere --mdd_frac)")
    parser.add_argument("--lr", type=float, default=None, help="Sobrescribe model.optimizer.lr")
    parser.add_argument("--folds", type=str, default=None, help="Solo estos folds, ej. 0 o 0,2")
    args = parser.parse_args()

    cfg = load_config()
    if args.pos_weight is not None:
        cfg["model"]["loss"]["pos_weight"] = args.pos_weight
    if args.masks_dir:
        cfg["paths"]["patches_masks"] = args.masks_dir
    cfg["model"]["training"]["augment"] = args.augment
    if args.epochs:
        cfg["model"]["training"]["epochs"] = args.epochs
    cfg["model"]["training"]["mdd_frac"] = args.mdd_frac
    cfg["model"]["training"]["folds"] = args.folds
    cfg["model"]["training"]["init_ckpt"] = args.init_ckpt
    if args.lr:
        cfg["model"]["optimizer"]["lr"] = args.lr
    if args.tag:
        cfg["paths"]["checkpoints"] = f"outputs/{args.tag}/checkpoints"
        cfg["paths"]["metrics"] = f"outputs/{args.tag}/metrics"
    run_spatial_cross_validation(cfg, args.metadata)
