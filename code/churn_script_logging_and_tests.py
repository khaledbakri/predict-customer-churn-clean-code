'''
Module for logging and testing the churn library solution.

Author: Khaled
Date: May 16 2023

'''
import logging
from os.path import exists
import churn_library_solution as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info(
            "Testing import_data: The shapes of df are greater than zero")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''

    df = cls.import_data("./data/bank_data.csv")
    perform_eda(df)
    try:
        perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda: The file wasn't found")
        raise err

    try:
        plot_list = [
            "Churn",
            "Customer_Age",
            "Marital_Status",
            'Total_Trans_Ct',
            'corr']
        for plot in plot_list:
            assert exists("images/eda/" + plot.lower() + ".png")
        logging.info(
            "Testing perform_eda: The plots were stored in eda folder correctly")
    except AssertionError as err:
        logging.error("Testing perform_eda: One or more plots are missing")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    df = cls.import_data("./data/bank_data.csv")
    category_list = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category"]

    try:
        encoder_helper(df, category_list, "Churn")
        logging.info("Testing encoder_helper: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing encoder_helper: The file wasn't found")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    df = cls.import_data("./data/bank_data.csv")
    category_list = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category"]
    df_encoded = cls.encoder_helper(df, category_list, "Churn")

    try:
        perform_feature_engineering(df_encoded, "Churn")
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing perform_feature_engineering: The file wasn't found")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    df = cls.import_data("./data/bank_data.csv")
    category_list = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category"]
    df_encoded = cls.encoder_helper(df, category_list, "Churn")
    X_train, y_train, X_test, y_test = cls.perform_feature_engineering(
        df_encoded, "Churn")

    try:
        train_models(
            X_train,
            y_train,
            X_test,
            y_test)
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: The file wasn't found")
        raise err

    try:
        plot_list = [
            "class_report_lr.png",
            "class_report_rf.png",
            "rfc.png",
            "roc.png"]
        for plot in plot_list:
            assert exists("images/results/" + plot)
        logging.info(
            "Testing train_models: The plots were stored in eda folder correctly")
    except AssertionError as err:
        logging.error("Testing train_models: One or more plots are missing")
        raise err

    try:
        assert exists("models/logistic_model.pkl")
        assert exists("models/rfc_model.pkl")
        logging.info(
            "Testing train_models: The models were stored in eda folder correctly")
    except AssertionError as err:
        logging.error("Testing train_models: One or more models are missing")
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
