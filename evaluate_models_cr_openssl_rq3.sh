#!/bin/bash
#SBATCH -p batch
#SBATCH --nodes 1
#SBATCH -c 20
#SBATCH --time=20:00:00
#SBATCH --mem=32GB
#SBATCH --array=1-72
#SBATCH --err="_logs/ml_results_cr_openssl_rq3_%a.err"
#SBATCH --output="_logs/ml_results_cr_openssl_rq3_%a.out"
#SBATCH --job-name="O_CR_RQ3"

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
IFS=',' read -ra par <<< `sed -n ${SLURM_ARRAY_TASK_ID}p Code/evaluate_models_cr_openssl_rq3.csv`
python3 -u Code/evaluate_models_cr_rq3.py "${par[0]}" "${par[1]}" "${par[2]}" "${par[3]}" "${par[4]}"