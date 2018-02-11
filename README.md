# Restaurant Predictions

This is my unique XGB Random Forest / Chainer LSTM ensemble model that I developed
for the Kaggle competition [Recruit Restaurant Visitor Forecasting](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting).

The basic idea is...
1. Run `code/Make_Tabular_Data.py` to create the input files for XGB and
`code/Make_Chainer_Data.ipynb` to create the input files for Chainer
2. Train the Chainer model first using `chainer/train.py` and `chainer/predict.py` (preferrably on GPU!)
3. Finally fit the XGB model using `code/XGB.ipynb`, which will take the Chainer predictions into account as part of the ensemble.

### Contents 

* The main programs are in `code`. 
* The various subdirectories in `code` contain the intermediate output that will be referrenced by `XGB.ipynb`.
* The directory `input` contains the competition input files, plus some predictions from public
models that we use as ensemble inputs.

### Notes

The Chainer predictions can be run with some dropout in order to create an ensemble prediction
and then you can take the averages, to avoid getting stuck in local maxima.

The input datasets must be obtained from Kaggle and placed in the `input` folder.
