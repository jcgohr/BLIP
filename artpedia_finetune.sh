python -m torch.distributed.run --nproc_per_node=$0 train_retrieval.py \
--config $1 \
--output_dir $2
