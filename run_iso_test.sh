# For some reason, after the first run, the MPI breaks interactive slurm job...
# Quick script to test things out.
# ititanrtx -m jagupard26
eval "$(conda shell.bash hook)"
conda activate lxuechen-summ-hf
pipenv run python iso_test.py --task "oai_test"
