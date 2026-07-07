#!/bin/bash 

export CUDA_VISIBLE_DEVICES=0
eval "$(conda shell.bash hook)"

# Preprocessing of downloaded data (already done)
# conda activate mlm_prottl
# python run_data_preprocessing.py --data_folder="./data"
# conda deactivate

# Then, create MSAs for all datasets
conda activate msagen
python src/mmseqs/search.py data/proeng/gb1/wildtype.fasta msas/database msas/individual_msas/proeng/gb1_trunc/a3m_files --mmseqs $CONDA_PREFIX/bin/mmseqs
python src/mmseqs/search.py data/proeng/aav/wildtype.fasta msas/database msas/individual_msas/proeng/aav/a3m_files --mmseqs $CONDA_PREFIX/bin/mmseqs
python src/mmseqs/search.py data/proeng/thermo/mixed_split.fasta msas/database msas/individual_msas/proeng/thermo/a3m_files --mmseqs $CONDA_PREFIX/bin/mmseqs
python src/mmseqs/search.py data/annotation/scl/balanced.fasta msas/database msas/individual_msas/annotation/scl/a3m_files --mmseqs $CONDA_PREFIX/bin/mmseqs
python src/mmseqs/search.py data/structure/secondary_structure/tape_ss3_processed.fasta msas/database msas/individual_msas/structure/secondary_structure/tape_ss3_processed/a3m_files --mmseqs $CONDA_PREFIX/bin/mmseqs
python src/mmseqs/search.py data/evolution/remote_homology/tape_rh_processed.fasta msas/database msas/individual_msas/evolution/remote_homology/tape_rh_processed/a3m_files --mmseqs $CONDA_PREFIX/bin/mmseqs
conda deactivate

# Next, preprocess MSA data
conda activate gen_prottl
python run_msa_preprocessing.py msas/individual_msas/proeng/gb1_trunc/a3m_files msas/individual_msas/proeng/gb1_trunc/ --filename=gb1_trunc_memmap --use_wildtype_context=1
python run_msa_preprocessing.py msas/individual_msas/proeng/aav/a3m_files msas/individual_msas/proeng/aav/ --filename=aav_memmap --use_wildtype_context=1
python run_msa_preprocessing.py msas/individual_msas/proeng/thermo/a3m_files msas/individual_msas/proeng/thermo/ --filename=thermo_memmap
python run_msa_preprocessing.py msas/individual_msas/annotation/scl/a3m_files msas/individual_msas/annotation/scl/ --filename=annotation_memmap
python run_msa_preprocessing.py msas/individual_msas/structure/secondary_structure/tape_ss3_processed/a3m_files msas/individual_msas/structure/secondary_structure/tape_ss3_processed/ --filename=tape_ss3_processed_memmap
python run_msa_preprocessing.py msas/individual_msas/evolution/remote_homology/tape_rh_processed/a3m_files msas/individual_msas/evolution/remote_homology/tape_rh_processed/ --filename=tape_rh_processed_memmap
conda deactivate
