# Automatic Data Labeling for Software Vulnerability Prediction

This is the README file for the reproduction package of the paper: "Automatic Data Labeling for Software Vulnerability Prediction Models: How Far Are We?".

The package contains the following artefacts:
1. Data:
	+ Contains the code of the 1,582 human-labeled vulnerable files, 3,391 auto/D2A-labeled vulnerable files, and 73,042 non-vulnerable files, which can be used to train file-level vulnerability prediction models.
	+ Contains mappings between data and data splits used for 5-round evaluation, as described in section 4.4.1 in the paper.
2. D2A Validation: contains the Excel file (`D2A-labeled VFCs_Validation.xlsx`) of the manual validation of the auto/D2A-labeled vulnerable files
3. Code: contains the source code we used in our work to answer Research Questions (RQs) 2 and 3. It's noted that we setup the code to run on a computing cluster that runs on Slurm. Therefore, most of the code must be submitted using bash script (.sh) file, but our code can still be run locally by executing the python file directly.

The size of the `Data` folder is large, so they are not included in the GitHub repository. Please download them from this [link](https://figshare.com/s/d9c65dc6969e1a566b3e) instead.

Before running any code, please install all the required Python packages using the following command: `pip install -r requirements.txt`

1. For RQ2, train and evaluate models by running `evaluate_models_openssl_rq2.sh` and `evaluate_models_ffmpeg_rq2.sh` (slurm scripts)
2. For RQ3, train and evaluate models by running`evaluate_models_cl_openssl_rq3.sh`, `evaluate_models_cl_ffmpeg_rq3.sh` (Confident Learning), `evaluate_models_cr_openssl_rq3.sh`, `evaluate_models_cr_ffmpeg_rq3.sh` (Centroid-based Removal), `evaluate_models_dr_openssl_rq3.sh`, `evaluate_models_dr_ffmpeg_rq3.sh` (Domain-specific Removal)

Note that after these training/evaluation scripts finish, they will generate output folders, i.e., `ml_results_rq2/` and `ml_results_rq3/` containing the results (.csv files) for the RQ2 and RQ3 models, respectively.

These .csv result files can then be used for analysis and comparison as described in the paper.

The file `ML-wise_Results.xlsx` contains the results of RQ2 and RQ3 for each Machine Learning (ML) algorithm based on the main MCC metric. Each result of an ML algorithm is aggregated from all the feature extractors and all the five test folds.