python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py \
--config ./configs/artpedia_config.yaml \
--output_dir ../../blip
