# Sample code

## Setting up environment

```
conda create -n env_name python swig
conda activate env_name
conda install -c conda-forge numpy scipy pandas box2d-py gym=0.21 joblib matplotlib seaborn
```


## Getting code

Download two files: `windy_frozen_lake_experiments.py` and `classic_control_experiments.py`

## Run all experiments

You can run this commands to go through all experiments sequentially 
```
python windy_frozen_lake_experiments.py
python classic_control_experiments.py
```

Please note that the classic_control experiments may be time consuming. Expensive computations will get saved to disk for saving time in the next run. The initial selected parameters take shorter time, but the solutions are not as good. You can find the set of parameters used in the paper within the script. Just set `EXPERIMENTS = FULL_EXPERIMENTS` and rerun the script.


