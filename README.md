# Pushing the Limits of Pre-training for Time Series Forecasting in the CloudOps Domain

Official code repository for the paper "Pushing the Limits of Pre-training for Time Series Forecasting in the CloudOps Domain". 
Check out our [paper](https://arxiv.org/abs/2310.05063) for more details. Accompanying datasets can be found [here](https://huggingface.co/datasets/Salesforce/cloudops_tsf).

# Usage
We use [Hydra](https://hydra.cc/) for config management.

Run the pre-training script:
```bash
python -m pretraining.pretrain_exp backbone=BACKBONE size=SIZE ++data.dataset_name=DATASET
```
* where the options for ```BACKBONE```, ```SIZE``` options can be found in ```conf/backbone``` and ```conf/size``` respectively.
* ```DATASET``` is one of ```azure_vm_traces_2017```, ```borg_cluster_data_2011```, or ```alibaba_cluster_trace_2018```.
* see ```confg/pretrain.yaml``` for more details on the options.
* training logs and checkpoints will be saved in ```outputs/```

Run the forecast script:
```bash
python -m pretraining.forecast_exp backbone=BACKBONE forecast=FORECAST size=SIZE ++data.dataset_name=DATASET
```
* where the options for ```BACKBONE```, ```FORECAST```, ```SIZE``` options can be found in ```conf/backbone```, ```conf/forecast```, and ```conf/size``` respectively.
* ```DATASET``` is one of ```azure_vm_traces_2017```, ```borg_cluster_data_2011```, or ```alibaba_cluster_trace_2018```.
* see ```confg/forecast.yaml``` for more details on the options.
* training logs and checkpoints will be saved in ```outputs/```

# Citation
If you find the paper or the source code useful to your projects, please cite the following bibtex:
<pre>
@article{woo2023pushing,
  title={Pushing the Limits of Pre-training for Time Series Forecasting in the CloudOps Domain},
  author={Woo, Gerald and Liu, Chenghao and Kumar, Akshat and Sahoo, Doyen},
  journal={arXiv preprint arXiv:2310.05063},
  year={2023}
}
</pre>
