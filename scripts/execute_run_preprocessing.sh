# queue for generating Multiple Sequence Alignments for all datasets

export CUDA_VISIBLE_DEVICES=0

### GB1 ###
python run_preprocessing.py msas/individual_msas/proeng/gb1_trunc/a3m_files msas/individual_msas/proeng/gb1_trunc/ --filename=gb1_trunc_memmap --is_mutation=1
python run_preprocessing.py msas/individual_msas/proeng/gb1/a3m_files msas/individual_msas/proeng/gb1/ --filename=gb1_memmap

### AAV ###
python run_preprocessing.py msas/individual_msas/proeng/aav/a3m_files msas/individual_msas/proeng/aav/ --filename=aav_memmap --is_mutation=1

### Thermo ###
python run_preprocessing.py msas/individual_msas/proeng/thermo/a3m_files msas/individual_msas/proeng/thermo/ --filename=thermo_memmap # took 60h & run on big gpu

### Subcellular Localization ###
python run_preprocessing.py msas/individual_msas/annotation/scl/a3m_files msas/individual_msas/annotation/scl/ --filename=annotation_memmap # took 20h

### SS3 ###
python run_preprocessing.py msas/individual_msas/structure/secondary_structure/tape_ss3_processed/a3m_files msas/individual_msas/structure/secondary_structure/tape_ss3_processed/ --filename=tape_ss3_processed_memmap

### RH ###
python run_preprocessing.py msas/individual_msas/evolution/remote_homology/a3m_files msas/individual_msas/evolution/remote_homology --filename=tape_rh_processed_memmap


