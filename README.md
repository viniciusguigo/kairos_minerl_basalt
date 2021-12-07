# KAIROS MineRL BASALT

Codebase for the solution that won first place and was awarded the most human-like agent in the 2021 NeurIPS Competition MineRL BASALT Challenge.

Original README of the competition: https://github.com/minerllabs/basalt_competition_baseline_submissions

## Installation Guide

Follow the steps below to install Anaconda, create Anaconda environment (```conda-env create -f environment.yml conda activate basalt```), then activate the conda environment (```conda activate basalt```), load environment variables: ```source ./utility/environ.sh```, and try to run the KAIROS agent locally (```./utility/evaluation_locally.sh --verbose```) to make sure everything is working properly.

If new dependencies are added to the environment.yml file, activate the environemnt and run ```conda env update -f environment.yml --prune```.

## Usage

0. Activate your Conda environment: ```conda activate basalt```

1. Load environment variables: ```source ./utility/environ.sh```

2. (Optional) If you modified the main KAIROS package, update it: ```pip install -e ./kairos_minerl/```

3. Run KAIROS agent: ```./utility/evaluation_locally.sh --verbose```
 

## Label Data and Retrain Agent

Instruction to label data for the state-classifier is available at https://docs.google.com/document/d/11RxGh40WVZY1RX0734E0bWiHmpVX0lJMs7-paU1dqY8/edit#heading=h.fdoz4yxhczbg.