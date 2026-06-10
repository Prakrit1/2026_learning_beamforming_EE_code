

# Work in progress

.
├── models                  | trained models
├── outputs
│   ├── metrics             | metrics from evaluation
├── README.md               | this file
├── requirements.txt        | project dependencies
├── src                     | python source related to..
│   ├── analysis            |   evaluation
│   ├── config              |   configuration
│   ├── data                |   data generation, i.e., satellite model
│   ├── models              |   learning models
│   ├── plotting            |   plotting
│   ├── tests               |   code tests
└── └── utils               |   shared helper functions
```


## Energy Efficiency Extension (Parajuli)

This branch extends the base code with a Soft Actor-Critic agent trained to maximize energy efficiency.

## Environment Setup

**GPU Training (Recommended)**
```bash
conda env create -f environment.yml
conda activate gpu_beamforming
```

**CPU / pip only**
```bash
pip install -r requirements.txt
```


> **Note:** Tested on `fuchu` node (`gpu:rtx8000`). Nodes `madrid`, `miami`, `mumbai` excluded due to driver incompatibility with TF 2.15.
