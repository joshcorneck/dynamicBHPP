# Dynamic BHPP: online changepoint detection

This repository contains the supporting Python code for the paper "*Online Bayesian changepoint detection for network Poisson processes with community structure*" by Joshua Corneck, Ed Cohen, James Martin, and Francesco Sanna Passino. This repository contains the following directories:

- `analyses` contains code for reproducing the simulations studies in the paper and for plotting the resulting output.
- `data` contains scripts to download the Santander Cycles data used in the paper, code to process the data, and scripts for inference on the network and for plotting the resulting output.
- `src` contains the code for simulating a dynamic BHPP and for the online inference procedure.

The file `requirements.txt` contains all relevant packages that need to be installed, and these can be installed by running

```sh
pip install -r requirements.txt
```

## General use of the code

The file `example.ipynb` contains example code for using the classes contained within the files in `src`.

## Reproducing the simulation studies in the paper

In the directory `analyses/simulation_studies/run_files` can be found `.sh` scripts for reproducing the simulation studies from the paper. For example, to reproduce the results from simulation study 1, you can run the following commands:

```sh
cd analyses/simulation_studies/run_files
chmod +x run_simulation_study.sh
./run_simulation_study_1.sh
```

The resulting output is saved into `analyses/simulation_studies/simulation_output/simulation_study_1`, and the plots in the paper can then be reproduced using `analyses/simulation_studies/simulation_output/simulation_study_1/output_analysis_1.ipynb`. 

## Santander Cycles data: downloading, preprocessing and inference

To download the required data, the following commands need to be run:

```sh
cd data/santander_bikes/get_and_process_data
bash get_data.sh
```

The downloaded data can then be processed by running 

```sh
python3 data_processor.py &
```

Once these commands have been run, the inference procedures can be performed by running the commands:

```sh
cd ../inference
chmod +x run_santander_inference.sh
./run_santander_inference.sh
```

The output is saved in ``analyses/santander_bikes/output`` and the plots of the paper can be reproduced using ``output_analysis.ipynb``.

