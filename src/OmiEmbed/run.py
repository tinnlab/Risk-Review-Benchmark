import subprocess
import sys


# seeds = list(range(1))


def run_script_sys(script_path, dataset, time, fold):
    try:
        subprocess.run([
            sys.executable,  # This gets the current Python interpreter path
            script_path,
            '--dataset', str(dataset),
            '--cvtime', str(time),
            '--FoldID', str(fold)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")


def run():
    alldatasets = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
             "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
             "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
             "TCGA-STAD", "TCGA-UCEC"]
    times = list(range(1, 6))
    folds = list(range(1, 11))

    for dataset in alldatasets:
        for time in times:
            for fold in folds:
                # Run the script using subprocess method
                run_script_sys("./train_predict.py", dataset, time, fold)


