## WebQSP
python runner.py --do_predict --add_dist_feature=1 --add_inv_edges_to_edge_index=1 --data_dir=public_subgraphs/webqsp --data_file_suffix=roberta-base_mean_pool_masked_cbr_subgraph_k=75 --dataset_name=webqsp --dist_metric=cosine --eval_batch_size=1 --eval_steps=100 --gradient_accumulation_steps=4 --learning_rate=0.0024129869604528702 --logging_steps=10 --loss_metric=txent --max_grad_norm=1 --num_gcn_layers=3 --num_neighbors_eval=5 --num_neighbors_train=10 --num_train_epochs=30 --output_dir=/scratch/rajarshidas_umass_edu/cbr-weak-supervision/expts/ --save_steps=80 --save_total_limit=1 --task=pt_match --temperature=0.06453104622317246 --train_batch_size=1 --transform_input=1 --use_fast_rgcn=1 --use_wandb=1 --model_ckpt_path=public_models/webqsp/pytorch_model.bin

## MetaQA-3
python runner.py --do_predict --add_dist_feature=1 --add_inv_edges_to_edge_index=1 --data_dir=public_subgraphs/MetaQA-3/ --dataset_name=metaqa --eval_batch_size=1 --eval_steps=1024 --gradient_accumulation_steps=8 --learning_rate=0.000999196499343006 --logging_steps=10 --loss_metric=txent --max_grad_norm=1 --num_gcn_layers=3 --num_neighbors_eval=10 --num_neighbors_train=7 --num_train_epochs=5 --output_dir=/gypsum/scratch1/rajarshi/cbr-weak-supervision/expts/metaqa_3hop/ --save_steps=80 --save_total_limit=1 --task=pt_match --temperature=0.054647601500658546 --train_batch_size=1 --transform_input=1 --use_fast_rgcn=1 --use_wandb=1 --model_ckpt_path=public_models/MetaQA-3/pytorch_model.bin

## MetaQA-2
python runner.py --do_predict --add_dist_feature=1 --add_inv_edges_to_edge_index=1 --data_dir=public_subgraphs/MetaQA-2/ --dataset_name=metaqa --eval_batch_size=1 --eval_steps=1024 --gradient_accumulation_steps=8 --learning_rate=0.0009636974197022258 --logging_steps=10 --loss_metric=txent --max_grad_norm=1 --num_gcn_layers=3 --num_neighbors_eval=10 --num_neighbors_train=5 --num_train_epochs=5 --output_dir=/gypsum/work1/mccallum/rajarshi/cbr-weak-supervision/expts/metaqa_2hop/ --save_steps=80 --save_total_limit=1 --task=pt_match --temperature=0.05659862804187725 --train_batch_size=1 --transform_input=1 --use_fast_rgcn=1 --use_wandb=1 --model_ckpt_path=public_models/MetaQA-2/pytorch_model.bin

## MetaQA-1
python runner.py --do_predict --add_dist_feature=1 --add_inv_edges_to_edge_index=1 --data_dir=public_subgraphs/MetaQA-1/ --dataset_name=metaqa --eval_batch_size=1 --eval_steps=1024 --gradient_accumulation_steps=2 --learning_rate=0.0009037897314516692 --logging_steps=10 --loss_metric=txent --max_grad_norm=1 --num_gcn_layers=3 --num_neighbors_eval=15 --num_neighbors_train=5 --num_train_epochs=5 --output_dir=/gypsum/work1/mccallum/rajarshi/cbr-weak-supervision/expts/metaqa_1hop/ --save_steps=80 --save_total_limit=1 --task=pt_match --temperature=0.0469471950581896 --train_batch_size=1 --transform_input=1 --use_fast_rgcn=1 --use_wandb=1 --model_ckpt_path=public_models/MetaQA-1/pytorch_model.bin

## Synthetic
python runner.py --do_predict --add_dist_feature=1 --add_inv_edges_to_edge_index=0 --data_dir=public_subgraphs/synthetic/outputs/dset2_1_bind_fix/ --dataset_name=synthetic --dist_metric=cosine --eval_steps=20 --gcn_dim=64 --learning_rate=0.0049182257710046005 --logging_steps=10 --loss_metric=margin --margin=0.19297202096191013 --num_gcn_layers=4 --num_train_epochs=30 --output_dir=/home/rajarshidas_umass_edu/scratch/cbr-weak-supervision/expts/synthetic_expts/ --save_steps=60 --save_total_limit=2 --task=pt_match --temperature=6.292113581852526 --transform_input=1 --use_fast_rgcn=0 --use_wandb=1 --weight_decay=0.0001732909211183048 --model_ckpt_path=public_models/synthetic/pytorch_model.bin --use_scoring_head=transe --mixture_coefficient_contrastive_loss=1

## FreebaseQA
python runner.py --do_predict  --add_dist_feature=1 --add_inv_edges_to_edge_index=1 --data_dir=public_subgraphs/FreebaseQA/ --data_file_suffix=roberta-base_mean_pool_masked_cbr_subgraph_k=25 --dataset_name=freebaseqa --dist_metric=cosine --eval_batch_size=1 --eval_steps=100 --gradient_accumulation_steps=4 --learning_rate=0.002571204072908148 --logging_steps=10 --loss_metric=txent --max_grad_norm=1 --num_gcn_layers=3 --num_neighbors_eval=7 --num_neighbors_train=5 --num_train_epochs=30 --output_dir=/mnt/nfs/work1/mccallum/rajarshi/cbr-weak-supervision/expts/freebaseqa/ --save_steps=80 --save_total_limit=1 --task=pt_match --temperature=0.09302024706116298 --train_batch_size=1 --transform_input=1 --use_fast_rgcn=1 --use_wandb=1 --model_ckpt_path=public_models/FreebaseQA/pytorch_model.bin
