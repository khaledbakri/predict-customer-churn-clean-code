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
    returns df for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


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


def perform_eda(df):
    '''
    perform eda on df and save figures to ./images folder
    input:
            df: pandas dataframe

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
            df[plot].value_counts('normalize').plot(kind='bar')
        elif plot == 'Total_Trans_Ct':
            sns.histplot(df[plot], stat='density', kde=True)
        elif plot == "corr":
            sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        else:
            df[plot].hist()
        save_figure(fig, "./images/eda/" + plot.lower() + ".png")


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for
            naming variables or index out_data column]

    output:
            df: pandas dataframe with new columns for
    '''
    for category in category_lst:
        lst = []
        groups = df.groupby(category).mean()[response]

        for val in df[category]:
            lst.append(groups.loc[val])
        key = category + "_" + response
        df[key] = lst

    return df


def perform_keeping_columns(df):
    '''
    input:
              df: pandas dataframe

    output:
              X: df with kept columns
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
    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols]

    return X


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used
              for naming variables or index out_data column]

    output:
              X_train: input training data
              X_test: input testing data
              y_train: output training data
              y_test: output testing data
    '''

    out_data = df[response]
    X = perform_keeping_columns(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, out_data, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in ./images folder
    input:
            y_train: training response values
            y_test:  test response values
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
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    save_figure(fig, "./images/results/class_report_lr.png")

    fig = plt.figure(figsize=(10, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    save_figure(fig, "./images/results/class_report_rf.png")


def feature_importance_plot(model, X, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X: pandas dataframe of input data values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]

    # Create plot
    fig = plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X.shape[1]), names, rotation=90)

    save_figure(fig, output_pth)


def roc_plot(rf_model, lr_model, X_test, y_test):
    '''
    plot and save the roc curve of both random forest and logistic regression models
    input:
              rf_model: model from random forest
              lr_model: model from logistic regression
              X_test: input testing data
              y_test: output testing data
    output:
              None
    '''
    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    fig = plt.figure(figsize=(15, 8))
    axis = plt.gca()
    plot_roc_curve(rf_model, X_test, y_test, ax=axis, alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    save_figure(fig, "./images/results/roc.png")


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: ./images + scores, and store models
    input:
              X_train: input training data
              X_test: input testing data
              y_train: output training data
              y_test: output testing data
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
    cv_rfc.fit(X_train, y_train)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train,
        "./images/results/rfc.png")
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')

    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(X_train, y_train)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    joblib.dump(lrc, './models/logistic_model.pkl')

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    roc_plot(cv_rfc.best_estimator_, lrc, X_test, y_test)

if __name__ == "__main__":
    dataframe = import_data("./data/bank_data.csv")
    category_list = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category"]
    dataframe_encoded = encoder_helper(dataframe, category_list, "Churn")
    X_training, y_training, X_testing, y_testing = perform_feature_engineering(
        dataframe_encoded, "Churn")

    train_models(
        X_training,
        y_training,
        X_testing,
        y_testing)
