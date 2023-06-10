# Sample code

## Setting up environment

```
conda create -n env_name python swig
conda activate env_name
conda install -c conda-forge box2d-py
pip install numpy scipy pandas pygame gym joblib matplotlib seaborn tqdm
```


## Getting code

Download two files: `windy_frozen_lake_experiments.py` and `classic_control_experiments.py`

## Run all experiments

You can run these commands to go through all experiments sequentially 
```
python windy_frozen_lake_experiments.py
python windy_frozen_lake_model_free.py
python classic_control_experiments.py
```

Please note that the classic_control experiments may be time consuming. Expensive computations will get saved to disk for saving time in the next run. The initial selected parameters take shorter time, but the solutions are not as good. You can find the set of parameters used in the paper within the script. To reproduce these results rerun the script with the flag `--full`:
```
python classic_control_experiments.py --full
```


