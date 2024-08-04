test pull/push check

# Project tree

MLOPS_Project/
├── .github/
│   └── workflows/
│       └── main.yml
├── data/
│   └── diabetes.csv
├── models/
│   └── best_model.pkl
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── tune_hyperparameters.py
│   ├── app.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   └── test_train.py
├── requirements.txt
├── Dockerfile
├── dvc.yaml
├── .dvc/
│   └── config
├── diabetes.csv.dvc
├── .gitignore
└── README.md


# Install Requirements
pip3 install requirements.txt
