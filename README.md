# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Requirements

To install the dependencies, you need first to create an environment with python 3.8, which can be done using conda and the following command:
```bash
conda create --name clean_code_3.8 python=3.8
```

Next, run the command below to install the dependencies:
```bash
pip install -r requirements/requirements_py38.txt
```
## Project Description

This project is the application of the coding best practices to refactor the notebook churn_notebook.ipynb, yielding the module called churn_library_solution.py.

## Files and data description
### Directory structure
```bash
predict-customer-churn-clean-code/
├── code
├── data
├── images 
│   ├── eda
│   └── results
├── logs
├── models
├── notebooks 
├── README.md
└── requirements
``` 
### Description of the folders
**code**: This folder contains the module of the churn library solution and the scripts to test it.\
**data**: This folder contains the data used in the churn library solution.\
**images**: This folder contains the images of the eda made in the data folder and results obtained with the random forest and logistic regression models training.\
**logs**: This folder contains the log file of the tests.\
**models**: This folder contains the pickle files of both random forest and logistic regression models.\
**notebooks**: This folder cointains the guide notebook and the notebook that was refactored.\
**requirements**: This folder cointains the requirement files.

## Running Files

To test the module churn_library_solution.py, please run the command below:
```bash
python3 code/churn_script_logging_and_tests.py 
```



