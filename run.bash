# Training from scratch
python -m scripts.train_diffusion configs/training.yml

# Sampling for pockets in the testset
python -m scripts.sample_diffusion configs/sampling.yml --data_id {i} # Replace {i} with the index of the data. i should be between 0 and 99 for the testset.

# Evaluation from sampling results
python scripts/evaluate_diffusion.py {OUTPUT_DIR} --docking_mode vina_score --protein_root data/test_set

# Evaluation from meta files
python -m scripts.evaluate_from_meta sampling_results/targetdiff_vina_docked.pt --result_path eval_targetdiff