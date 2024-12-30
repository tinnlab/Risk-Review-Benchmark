
This repository contains the code to reproduce the benchmark of risk prediction methods for the manuscript:

**A Comprehensive Review of Cancer Survival Prediction Using Multi-Omics Integration and Clinical Variables**

## Steps to Reproduce the Results

### 1. Install Anaconda or Miniconda
Follow the instructions to install either [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) on your system.

### 2. Create Conda Environments and Download The Data
To create the necessary Conda environments, navigate to the repository directory in your terminal and run the following commands:

```bash
chmod +x ./conda_envs_install.sh
./conda_envs_install.sh
```
This will set up all the required environments for running the notebooks and methods.

To download all the processed data for the benchmark, navigate to the repository directory in your terminal and run the following commands:

```bash
chmod +x ./data_download.sh
./data_download.sh
```
### 3. Run the Methods Using Jupyter Notebooks
In the terminal, navigate to the repository directory and run the following command to start Jupyter Lab:

```bash
jupyter lab
```
Once the command runs, it will provide a link to open Jupyter Lab in your browser.

Inside Jupyter Lab, navigate to the notebooks and run each method one by one. 

### 4. Generate Metrics
After running the notebooks, results will be stored in the run-results folder. To generate the performance metrics, open and run the ```metrics-generate.ipynb``` notebook.

### Notes
JupyterLab Installation: If JupyterLab is not installed by default in your Conda environment, you can install it with the following command:

```bash
conda install -c conda-forge jupyterlab
```

