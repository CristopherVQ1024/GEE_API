set -e
P=.venv-gpu/bin/python
run() {
  tag=$1; shift
  $P -m src.train --metadata data/processed/patches_metadata_v4.csv --augment --pos_weight 5 --init_ckpt outputs/v3/checkpoints --tag $tag "$@" > outputs/logs/train_$tag.log 2>&1
  $P -m src.evaluate_deter --ckpt outputs/$tag/checkpoints --tag $tag > outputs/logs/deter_eval_$tag.log 2>&1
  $P -m src.evaluate_patch_level --ckpt outputs/$tag/checkpoints --tag $tag --metadata data/processed/patches_metadata_v3.csv > outputs/logs/patch_eval_$tag.log 2>&1
}
run v7a --mdd_frac 0.7 --lr 2e-5 --epochs 30
run v7b --mdd_frac 0.5 --lr 5e-5 --epochs 30
run v7c --mdd_frac 0.7 --lr 1e-5 --epochs 40
