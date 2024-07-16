#!/bin/bash
#SBATCH -p batch
#SBATCH --nodes 1
#SBATCH -c 20
#SBATCH --time=20:00:00
#SBATCH --mem=32GB
#SBATCH --array=1-108
#SBATCH --err="_logs/ml_results_openssl_rq2_%a.err"
#SBATCH --output="_logs/ml_results_openssl_rq2_%a.out"
#SBATCH --job-name="O_RQ2"

## Setup Python Environment
module load arch/haswell
module load Anaconda3/2020.07
module load CUDA/11.2.0
module load Java/1.8.0_191
module load Singularity
module load git/2.21.0-foss-2016b

source activate myenv
conda deactivate
source activate myenv

## Echo job id into results file
echo "array_job_index: $SLURM_ARRAY_TASK_ID"

## Read inputs
IFS=',' read -ra par <<< `sed -n ${SLURM_ARRAY_TASK_ID}p Code/evaluate_models_openssl_rq2.csv`
python3 -u Code/evaluate_models_rq2.py "${par[0]}" "${par[1]}" "${par[2]}" "${par[3]}" "${par[4]}"