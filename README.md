# README

## conda

create a new conda environment
```bash
conda create --name phy_gnn python=3.8.18
```

basic conda environment command
```bash
conda info --envs  # check conda environment
conda activate phy_gnn # activate phy_gnn conda environment
conda deactivate # deactivate phy_gnn conda environment
```

if it is your first time setup the python dependency, please run the following command to install the dependency
```bash
pip install -r requirements.txt
```

if you have import or update the python dependency, please run the following command to update the requirement file.
```bash
conda list --export --no-pip | awk -F= '{print $1"=="$2}' > requirements.txt
```

## tensorboard
you need to install both 'tensorboard' and 'tensorboardX' package. And please use the following command to check
your model
```bash
tensorboard --logdir=tmp/passive_lv_gnn_emul/1/logs/ 
```