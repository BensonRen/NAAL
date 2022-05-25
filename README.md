
# Towards Robust Deep Active Learning for Regression

This repository is the implementation of "Towards Robust Deep Active Learning for Regression" .

## Environment
The detailed conda environment is packaged in [.yml file](./environment.yml). (Note that to satisfy the anonymity requirement we've deleted the 'prefix' and 'name' row of yml file)

### Folder/sciprt names and usages
| Folder/script name | Usage |
|:---------------------------------------------:|:------------------------------------------------------------------:|
| meta/ |  The folder holding the datasets and the forward model scripts| 
| results/ |  (To be created by running) The result folder| 
| parameters.py| configuration input| 
| flag_reader.py |  configuration parsing| 
| class_wrapper.py |  The main functioning part of the code| 
| forward_models.py | The API to call the oracle functions | 
| model_maker.py |  The model architecture definition | 
| ADMpredict.py |  ADM dataset API| 

##  benchmark datasets and its data source
| Dataset | Download / Direcotry |
|:---------------------------------------------:|:------------------------------------------------------------------:|
| 1D sine wave |  [script](./forward_model.py) | 
| 2D robotic arm/ |  [script](./forward_model.py) | 
| Metamaterial: Stack | [script](./meta/Stack/generate_chen.py)/[original](https://github.com/closest-git/MetaLab)| 
| Metamaterial: ADM  | [script](./ADMpredict.py)/[original](https://github.com/ydeng-MLM/ML_MM_Benchmark)| 
| NASA Airfoil | [script](./meta/airfoil/airfoil.ipynb)/[original](https://archive.ics.uci.edu/ml/datasets/airfoil+self-noise)| 
| Hydro & Yacht | [script](./meta/hydro/hydro.ipynb)/[original](https://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics)| 

## Runing the active learning

To train the models in the paper, simply run the train.py (with adjusted parameters within, commented)

```train
python train.py 
```

> The main loop is the hyper_sweep_AL where all the experiments are run. The major for loop component would be the 'al_mode' where is specifies which AL algorithm to use and 'al_x_pool_factor' where the pool ratio would be determined.
> 
> The dataset specific information is located in the parameters.py where each dataset chunks of code and parameters are clearly marked using comments. To run the Active learning experiement for all of the datasets, one need to run this train.py for a dataset, change and comment out in the parameters.py to switch to another dataset, then run the train.py in a alterating fashion.
> 
> To counter the stochastic nature of the neural network training process, the training is done 5 times. After the training, the results would be placed under the results/ folder and one can use the provided script in the below section to recreate the plots that we plotted in the paper
> 
> Note that for QBC diversity and density, they use the same chunk of code and one need to adjust the lambda and beta parameter (density and diversity weighting) in the parameters.py function before running the corresponding code


## Plotting

```plot
result_compare_AL.ipynb
Comparison_plot_NA_pool.ipynb
```

> These two script contains all the code you need for reproducing the plots we presented in the paper. It plots both the traditional test loss vs number of training data, the data burden plot etc. 
> First plot and collect the data using result_compare_AL.ipynb for each of the individual experiments. Simply change the directory of your resulting file (as commented instructions). Then use the Comparison_plot_NA_pool.ipynb script to compile them into our paper format plot.


## Contributing
If you would like to contribute either method or dataset to this github repo, feel free to raise a issue or make a merge request.
