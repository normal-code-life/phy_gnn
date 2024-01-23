# README

## conda

basic conda environment command
```bash
conda info --envs  # check conda environment
conda activate xxx # activate xxx conda environment
conda deactivate # deactivate xxx conda environment
```

create a new conda environment
```bash
conda create --name gnn python=3.7
```

if you have import or update the python dependency, please run the following command to update the requirement file.
```bash
conda list --export --no-pip | awk -F= '{print $1"="$2}' > requirements.txt
```