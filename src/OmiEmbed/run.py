import subprocess
import sys


# seeds = list(range(1))


def run_script_sys(script_path, dataset, seed):
    try:
        subprocess.run([
            sys.executable,  # This gets the current Python interpreter path
            script_path,
            '--dataset', str(dataset),
            '--FoldID', str(seed)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")


def run():
    all_folds = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
             "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
             "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
             "TCGA-STAD", "TCGA-UCEC"]

    seeds = list(range(1, 11))
    for fold in all_folds:
        for seed in seeds:
            # Run the script using subprocess method
            run_script_sys("./train_predict.py", fold, seed)

