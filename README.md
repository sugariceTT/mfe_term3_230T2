# 230T-2 Final Report - Forecasting volatility by hybrid models of LSTM and GARCH-type models

This is the 230T-2 final project by group 6. We explored the idea of  combining inputs from 3 GARCH-type models (GARCH, EGARCH, EWMA) with LSTM to predict volatility. 

## Contributor

Yang Xu, Xuanyi Ji, Ruobing Tang, Haoran Liu, Wenqiang Wang

## Packages

numpy, pandas, matplotlib, seaborn, arch, statsmodels, sklearn, pytorch

## File Description

INPUT: Contains dataset we used and GARCH-type models parameters we generate.

garch_model.py: Experiments of 3 GARCH-type models and generate variables for further combination with LSTM.

KR_experiments.py: Experiments of hybrid models in the South Korea market.

US_experiments.py: Experiments of hybrid models in the US market.

## Usages

Put python file and data in the same folder for the initialization part. 

Run garch_model.py first to generate parameters, and then run the other two python file to train hybrid models.
