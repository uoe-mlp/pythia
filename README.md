![Build Python](https://github.com/GrowlingM1ke/MLP-CW2-2021/workflows/Build%20Python/badge.svg)

# Pythia
Times series LSTM based deep learning model. To know more about the Oracle of Delphi, follow the [white rabbit](https://en.wikipedia.org/wiki/Pythia).

## Main Classes
- ```experiment``` is the configurations of everything. In theory, this should be reproducible and deterministic (maybe I need to add a seed in the json). You can find an example in test/experiment/data/simple_experiment.json
- ```analysis``` is a property of ```experiment``` that identifies the broader temporal framework used. Values can be for instance walk-forward backtest, cross-validation, standard (=train-val-test), robustness analysis, transfer learning and so on
- ```market``` is a property of ```experiment``` that identifies the environment used
- ```agent``` is a property of ```experiment``` that identifies the agent. In the first case, ```agent``` is ```supervised``` and has two parameters: ```prediction``` and ```trading```.