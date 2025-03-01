# Passive LV Model Training

This folder contains code for training graph models on passive left ventricle (LV) data.

## Directory Structure
```
passive_lv/
├── README.md
├── start_model_train.sh          # Training script
├── single_case_evaluation.py     # Evaluation script for individual cases
├── fe_heart_sage_v1/            # Model version 1
│   ├── train/
│   │   ├── model.py             # Model architecture
│   │   └── config.py            # Training configuration
├── fe_heart_sage_v3/            # Model version 3
│   ├── train/
│   │   ├── model.py
│   │   └── config.py
└── passive_lv_gnn_emul/         # Basic GNN model
    ├── train/
    │   ├── model.py
    │   └── config.py
```

## Model Architectures
Each model variant has specific architectural features:

### passive_lv_gnn_emul
- Basic GNN architecture
- Direct message passing between nodes
- Simple aggregation functions

### fe_heart_sage_v1
- GraphSAGE-based architecture
- Neighborhood sampling
- Enhanced feature propagation

### fe_heart_sage_v3
- Advanced GraphSAGE model
- Improved message passing
- Enhanced aggregation mechanisms
- Better handling of geometric features

## Training Configuration
Common training parameters:
- Learning rate: Specified in respective config files
- Batch size: Model dependent
- Optimizer: Adam
- Loss function: MSE for displacement and stress

## Usage

### Step 1: Data Preparation
First, run the data preparation script to process and format the raw data:
```bash
sh phy_gnn/task/passive_lv/start_data_preparation.sh \
    --task_name <task_name> \
    --model_name <model_name> \
    --config_name <config_name>
```

Available options:
- model_name: passive_lv_gnn_emul, fe_heart_sage_v1, fe_heart_sage_v3 (note: in this case, all three models share the same data)
- config_name: Usually 'train_config' or as specified in the model's config folder

This step will:
- Process the raw FE simulation data
- Generate graph structures
- Create normalized features
- Save prepared data in the specified format

### Step 2: Model Training
After data preparation, start the model training using:
```bash
sh phy_gnn/task/passive_lv/start_model_train.sh \
    -model_name <model_name> \
    -config_name <config_name> > <model_name>.log 2>&1 &
```

Example usage for different models:


#### 1. passive_lv_gnn_emul
Basic GNN model for LV emulation.
```cmd
sh phy_gnn/task/passive_lv/start_model_train.sh -model_name passive_lv_gnn_emul -config_name train_config_lv_data > passive_lv_gnn_emul.log 2>&1 &
```

#### 2. fe_heart_sage_v1
GraphSAGE-based model with FE-specific adaptations.
```cmd
sh phy_gnn/task/passive_lv/start_model_train.sh -model_name fe_heart_sage_v1 -config_name train_config > fe_heart_sage_v1.log 2>&1 &
```

#### 3. fe_heart_sage_v3
Enhanced GraphSAGE model with improved architecture.
```cmd
sh phy_gnn/task/passive_lv/start_model_train.sh -model_name fe_heart_sage_v3 -config_name train_config > fe_heart_sage_v3.log 2>&1 &
```

## Requirements
- PyTorch
- NumPy
- Pandas
- Other dependencies as specified in the project's main requirements.txt