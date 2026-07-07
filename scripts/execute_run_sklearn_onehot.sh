# queue for running all sklearn models from embs
export CUDA_VISIBLE_DEVICES=0

### GB1 ###
python run_protran_sklearn.py --dataset_path="data/proeng/gb1/one_vs_rest.csv" --encoder_name="onehot" --checkpoint=1 --embed_batch_size=64 --flatten_emb="flatten" --seq_end_idx=56 --all_embed_layers=True --all_result_folder="results/sklearn-onehot" --embed_torch_seed=42
python run_protran_sklearn.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="onehot" --checkpoint=1 --embed_batch_size=64 --flatten_emb="flatten" --seq_end_idx=56 --all_embed_layers=True --all_result_folder="results/sklearn-onehot" --embed_torch_seed=42
python run_protran_sklearn.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="onehot" --checkpoint=1 --embed_batch_size=64 --flatten_emb="flatten" --seq_end_idx=56 --all_embed_layers=True --all_result_folder="results/sklearn-onehot" --embed_torch_seed=42

# ### AAV ###
python run_protran_sklearn.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="onehot" --checkpoint=1 --embed_batch_size=64 --flatten_emb="flatten" --all_embed_layers=True --all_result_folder="results/sklearn-onehot" --embed_torch_seed=42
python run_protran_sklearn.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="onehot" --checkpoint=1 --embed_batch_size=64 --flatten_emb="flatten" --all_embed_layers=True --all_result_folder="results/sklearn-onehot" --embed_torch_seed=42
python run_protran_sklearn.py --dataset_path="data/proeng/aav/low_vs_high.csv" --encoder_name="onehot" --checkpoint=1 --embed_batch_size=64 --flatten_emb="flatten" --all_embed_layers=True --all_result_folder="results/sklearn-onehot" --embed_torch_seed=42


# ### thermo ###
python run_protran_sklearn.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="onehot" --checkpoint=1 --embed_batch_size=64 --flatten_emb="flatten" --all_embed_layers=True --all_result_folder="results/sklearn-onehot" --embed_torch_seed=42

