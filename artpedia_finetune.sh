python -m torch.distributed.run --nproc_per_node=$1 train_retrieval.py \
--config $2 \
--output_dir $3
