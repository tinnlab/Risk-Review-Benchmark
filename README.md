This is the repo to run the benchmark of risk prediction methods for the manuscript: A comprehensive review of cancer survival prediction using multi-omics integration and clinical variables.
Please follow these steps to run reproduce the results:
1. Install anaconda or miniconda following [this instruction](https://docs.anaconda.com/miniconda/install/)
2. Create the conda environments
   To create all the required environments, open a terminal in the repo directory.
   ```
   chmod +x ./conda_envs_install.sh
   ./conda_envs_install.sh
   ```
3. Using the seven jupyter notebooks to run the methods
   1. Open a terminal in the repo directory, run the command:
   ```
   jupyter lab
   ```
   2. Open the link to the jupyter lab in a browser.
   3. Open the jupyter notebooks and run each method one by one
4. After all the results are generated in the ```run-results``` folder. Run the ```result-generation.ipynb``` notebook to produce the metrics. 
