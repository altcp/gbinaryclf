""" Automated Tests and (or) Usage Examples """
import pandas as pd

from gbinaryclf.ai import mlmodels, select


def test_selection():

    train = "./gbinaryclf/data/Train Data.xlsx"
    test = "./gbinaryclf/data/Test Data.xlsx"

    df_train = pd.read_excel(train).fillna(0)
    df_test = pd.read_excel(test).fillna(0)

    clf, outcome = mlmodels(df_train, "Y", True).classifiers()
    selected_x, selection = select(df_test, "Y", clf).feature()

    if selection == True:
        df_test_select = df_test[[selected_x, "Y"]]
        clf, outcome = mlmodels(df_test_select, "Y", False).classifiers()
    else:
        clf, outcome = mlmodels(df_test, "Y", False).classifiers()

    assert not outcome.empty
