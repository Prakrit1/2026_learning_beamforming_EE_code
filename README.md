

# Work in progress
oThis code was used in the following unpublished work [(preprint not available yet)](url).

[1]  Alea Schröder, Steffen Gracla, Dirk Wübben, Armin Dekorsy,
"Robust Rate-Splitting Multiple Access using Reinforcement Learning in Low Earth Orbit Satellite Communications", under preparation.

Email: {**schroeder**, gracla, wuebben, dekorsy}@ant.uni-bremen.de

The code version associated with this paper along with the used learned models and evaluation results is found in the releases.
The project structure is as follows
```
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
