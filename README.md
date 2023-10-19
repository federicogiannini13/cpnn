# cPNN
This repository contains the code used for the experimentation shown in the paper.

Link to the paper: https://link.springer.com/chapter/10.1007/978-3-031-33383-5_26

## 1) Installation
execute:

`conda create -n cpnn python=3.8`

`conda activate cpnn`

`pip install -r requirements.txt`

## 2) Project structure
The project is composed by the following directories.
#### datasets
It contains the generated data streams.
* **sine_rw_10_13**: S1+, S1-.
* **sine_rw_10_31**: S1-, S1+.
* **sine_rw_10_24**: S2+, S1-.
* **sine_rw_10_42**: S2-, S2+.
* **sine_rw_10_1234**: S1+, S2+, S1-, S2-.
* **sine_rw_10_1432**: S1+, S2-, S1-, S2+.
* **sine_rw_10_2143**: S2+, S1+, S2-, S1-.
* **sine_rw_10_2341**: S2+, S1-, S2-, S1+.
#### lab
It contains Jupyter Notebooks.
#### models
It contains the python modules implementing cPNN and cLSTM.
#### run_test
It contains the script to execute to replicate results and plot metrics.
#### data
It contains the python modules implementing the data stream generator.

## 3) Run test
#### run_test/test_then_train.py
It runs the prequential evaluation. Run `python -m run_test.test_then_train --help` to see the arguments.
Change the variable `dataset` in the code to set the data stream.
It stores pickle files containing the results in the `performance/<data_stream_name>/<model_name>_<hidden_size>hs` folder. 

## 4) Show results
All the following scripts/notebooks require the prequential evaluation to be executed on all the three architectures (cPNN, cLSTM, cLSTMs)
#### run_test/plot_performance.py
It plots the metrics trend of different concepts.
You should change the variable `dataset` in the code to set the data stream.
It stores the plots of accuracy, Kappa Temporal and Cohen Kappa in files: `performance/<data_stream_name>/_plot_clstm_<hidden_size>hs/test_then_train___clstm_<metric>.png`. 
#### lab/normality.py:
It performs the Shapiro-Wilk test to check for the normality of the samples.
It stores the results in `performance/<data_stream_name>/_prob_test/normality_<data_stream_name>` in .csv and .xlsx format.
#### lab/test.py:
It performs all the mentioned tests in Section 5.
It stores the results in `performance/<data_stream_name>/_prob_test/test_complete_<data_stream_name>` in .csv and .xlsx format.
#### lab/generate_sine_rw:
It generates a new data stream containing concepts S1+, S2+, S1-, S2- and stores it in the dataset folder.

## Credits
https://github.com/AndreaCossu/ContinualLearning-SequentialProcessing

https://github.com/alvarolemos/pyism
