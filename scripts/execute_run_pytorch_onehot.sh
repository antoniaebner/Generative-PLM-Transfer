# queue for running all pytorch models from embs

export CUDA_VISIBLE_DEVICES=0

### scl ###
python run_protran_pytorch.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="onehot" --if_encode_all=True --embed_batch_size=64 --flatten_emb="flatten" --loader_batch_size=256 --epochs=100 --all_plot_folder="results/pytorch_learning_curves-onehot" --all_result_folder="results/pytorch-onehot" --if_rerun_layer=True

### ss3 ###
python run_protran_pytorch.py --dataset_path="data/structure/secondary_structure/tape_ss3_processed.csv" --encoder_name="onehot" --if_encode_all=True --checkpoint=1 --embed_batch_size=64 --loader_batch_size=120 --epochs=100 --all_plot_folder="results/pytorch_learning_curves-onehot" --all_result_folder="results/pytorch-onehot" --if_rerun_layer=True

### rh ###
python run_protran_pytorch.py --dataset_path="data/evolution/remote_homology/tape_rh_processed.csv" --encoder_name="onehot" --if_encode_all=True --checkpoint=1 --embed_batch_size=64 --loader_batch_size=256  --flatten_emb="flatten" --learning_rate=1e-6 --epochs=500 --all_plot_folder="results/pytorch_learning_curves-onehot" --all_result_folder="results/pytorch-onehot" --if_rerun_layer=True
