'''
Module for the function of the churn library solution.

Author: Khaled
Date: May 16 2023

'''


# import libraries
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            dataframe: pandas dataframe
    '''
    dataframe = pd.read_csv(pth)
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return dataframe


def save_figure(fig, figname):
    '''
    save figures to ./images folder
    input:
            fig: figure to save
            figname: figure name

    output:
            None
    '''
    fig.savefig(figname)
    plt.close(fig)


def perform_eda(dataframe):
    '''
    perform eda on dataframe and save figures to ./images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    '''
    plot_list = [
        "Churn",
        "Customer_Age",
        "Marital_Status",
        'Total_Trans_Ct',
        'corr']

    for plot in plot_list:
        fig = plt.figure(figsize=(20, 10))
        if plot == "Marital_Status":
            dataframe[plot].value_counts('normalize').plot(kind='bar')
        elif plot == 'Total_Trans_Ct':
            sns.histplot(dataframe[plot], stat='density', kde=True)
        elif plot == "corr":
            sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        else:
            dataframe[plot].hist()
        save_figure(fig, "./images/eda/" + plot.lower() + ".png")


def encoder_helper(dataframe, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for
            naming variables or index out_data column]

    output:
            dataframe: pandas dataframe with new columns for
    '''
    for category in category_lst:
        lst = []
        groups = dataframe.groupby(category).mean()[response]

        for val in dataframe[category]:
            lst.append(groups.loc[val])
        key = category + "_" + response
        dataframe[key] = lst

    return dataframe


def perform_keeping_columns(dataframe):
    '''
    input:
              dataframe: pandas dataframe

    output:
              in_data: dataframe with kept columns
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    in_data = pd.DataFrame()
    in_data[keep_cols] = dataframe[keep_cols]

    return in_data


def perform_feature_engineering(dataframe, response):
    '''
    input:
              dataframe: pandas dataframe
              response: string of response name [optional argument that could be used
              for naming variables or index out_data column]

    output:
              in_data_train: in_data training data
              in_data_test: in_data testing data
              out_data_train: out_data training data
              out_data_test: out_data testing data
    '''

    out_data = dataframe[response]
    in_data = perform_keeping_columns(dataframe)
    in_data_train, in_data_test, out_data_train, out_data_test = train_test_split(
        in_data, out_data, test_size=0.3, random_state=42)

    return in_data_train, in_data_test, out_data_train, out_data_test


def classification_report_image(out_data_train,
                                out_data_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in ./images folder
    input:
            out_data_train: training response values
            out_data_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    fig = plt.figure(figsize=(10, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                out_data_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                out_data_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    save_figure(fig, "./images/results/class_report_lr.png")

    fig = plt.figure(figsize=(10, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                out_data_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                out_data_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    save_figure(fig, "./images/results/class_report_rf.png")


def feature_importance_plot(model, in_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            in_data: pandas dataframe of input data values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [in_data.columns[i] for i in indices]

    # Create plot
    fig = plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(in_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(in_data.shape[1]), names, rotation=90)

    save_figure(fig, output_pth)


def roc_plot(rf_model, lr_model, in_data_test, out_data_test):
    '''
    plot and save the roc curve of both random forest and logistic regression models
    input:
              rf_model: model from random forest
              lr_model: model from logistic regression
              in_data_test: input data testing data
              out_data_test: output data testing data
    output:
              None
    '''
    lrc_plot = plot_roc_curve(lr_model, in_data_test, out_data_test)
    fig = plt.figure(figsize=(15, 8))
    axis = plt.gca()
    plot_roc_curve(rf_model, in_data_test, out_data_test, ax=axis, alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    save_figure(fig, "./images/results/roc.png")


def train_models(in_data_train, in_data_test, out_data_train, out_data_test):
    '''
    train, store model results: ./images + scores, and store models
    input:
              in_data_train: input data training data
              in_data_test: input data testing data
              out_data_train: output data training data
              out_data_test: output data testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']}
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(in_data_train, out_data_train)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(in_data_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(in_data_test)
    feature_importance_plot(
        cv_rfc.best_estimator_,
        in_data_train,
        "./images/results/rfc.png")
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')

    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(in_data_train, out_data_train)
    y_train_preds_lr = lrc.predict(in_data_train)
    y_test_preds_lr = lrc.predict(in_data_test)
    joblib.dump(lrc, './models/logistic_model.pkl')

    classification_report_image(out_data_train,
                                out_data_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    roc_plot(cv_rfc.best_estimator_, lrc, in_data_test, out_data_test)
