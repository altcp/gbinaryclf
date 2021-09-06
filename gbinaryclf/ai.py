# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import gc

import pandas as pd
import scipy.stats as sps
from sklearn.ensemble import StackingClassifier
from sklearn.feature_selection import RFECV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
)
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import LinearSVC
from statsmodels.stats import diagnostic
from xgboost import XGBClassifier


# %%
class distribution:
    def __init__(self, y):
        self.y = y

    def aware(self):
        data = []
        data = self.y
        shapiro_test = sps.shapiro(data)
        ksstat, pvalue = diagnostic.lilliefors(data)

        if shapiro_test.pvalue > 0.05:
            if pvalue < 0.05:
                distribution = "alt"
            else:
                distribution = "norm"
        else:
            distribution = "alt"

        return distribution


# %%
class mlmodels:
    def __init__(self, df, target="Y", selection=False):
        self.df = df
        self.target = target
        self.select = selection

    def classifiers(self):

        y = self.df[self.target]
        x = self.df.loc[:, self.df.columns.difference([self.target])]

        # PPC
        pipe_ppc = Pipeline(
            steps=[
                ("N", MinMaxScaler()),
                ("M", Perceptron()),
            ]
        )

        param_grid_ppc = {
            "M__eta0": [0.0001, 0.001, 0.01, 0.1, 1.0],
            "M__early_stopping": [True],
        }

        # PAC
        pipe_pac = Pipeline(
            steps=[
                ("N", MinMaxScaler()),
                ("M", PassiveAggressiveClassifier()),
            ]
        )

        param_grid_pac = {
            "M__C": [0.2, 0.4, 0.6, 0.8],
            "M__early_stopping": [True],
            "M__class_weight": ["balanced"],
        }

        # XGB
        pipe_xgb = Pipeline(
            steps=[
                ("N", MinMaxScaler()),
                ("clf", XGBClassifier(eval_metric="logloss", use_label_encoder=False)),
            ]
        )

        param_grid_xgb = {
            "clf__gamma": [1, 2, 3],
            "clf__max_depth": [1, 3, 5],
            "clf__eta": [0.4, 0.6, 0.8, 1.0],
            "clf__reg_alpha": [0.1, 0.3, 0.5, 0.7],
            "clf__reg_lambda": [0.1, 0.3, 0.5, 0.7],
        }

        # GPC
        dist = distribution(y).aware()
        if dist == "norm":

            pipe_gpc = Pipeline(
                steps=[
                    ("T", PowerTransformer(method="yeo-johnson")),
                    ("M", GaussianProcessClassifier()),
                ]
            )

        else:

            pipe_gpc = Pipeline(
                steps=[
                    ("S", RobustScaler()),
                    ("T", PowerTransformer(method="yeo-johnson")),
                    ("M", GaussianProcessClassifier()),
                ]
            )

        ker_rbf = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
            1.0, length_scale_bounds="fixed"
        )
        ker_rq = ConstantKernel(1.0, constant_value_bounds="fixed") * RationalQuadratic(
            alpha=0.1, length_scale=1
        )
        ker_ess = ConstantKernel(1.0, constant_value_bounds="fixed") * ExpSineSquared(
            1.0, 5.0, periodicity_bounds=(1e-2, 1e1)
        )
        ker_wk = DotProduct() + WhiteKernel()
        kernel_list = [ker_rbf, ker_rq, ker_ess, ker_wk]

        param_grid_gpc = {
            "M__kernel": kernel_list,
            "M__n_restarts_optimizer": [0, 2, 4, 8],
        }

        # SKS
        pipe_sks = [
            (
                "XGB",
                GridSearchCV(
                    pipe_xgb, param_grid_xgb, cv=5, scoring="accuracy", n_jobs=-2
                ),
            ),
            ("SVC", make_pipeline(StandardScaler(), LinearSVC())),
        ]

        # SKR
        pipe_skr = [
            (
                "XGB",
                GridSearchCV(
                    pipe_xgb, param_grid_xgb, cv=5, scoring="accuracy", n_jobs=-2
                ),
            ),
            ("SVC", make_pipeline(RobustScaler(), LinearSVC())),
        ]

        # SKP
        pipe_skp = [
            (
                "PAC",
                GridSearchCV(
                    pipe_pac, param_grid_pac, cv=5, scoring="accuracy", n_jobs=-2
                ),
            ),
            ("SVC", LinearSVC()),
        ]

        # Get Performance
        gc.collect()
        if self.select == False:

            models = []
            models.append(
                (
                    "GPC",
                    GridSearchCV(
                        pipe_gpc, param_grid_gpc, cv=5, scoring="accuracy", n_jobs=-2
                    ),
                )
            )
            models.append(
                (
                    "SKP",
                    StackingClassifier(
                        estimators=pipe_skp,
                        final_estimator=LogisticRegression(),
                        n_jobs=-2,
                    ),
                )
            )
            models.append(
                (
                    "SKS",
                    StackingClassifier(
                        estimators=pipe_sks,
                        final_estimator=LogisticRegression(),
                        n_jobs=-2,
                    ),
                )
            )
            models.append(
                (
                    "SKR",
                    StackingClassifier(
                        estimators=pipe_skr,
                        final_estimator=LogisticRegression(),
                        n_jobs=-2,
                    ),
                )
            )
            models.append(
                (
                    "PPC",
                    GridSearchCV(
                        pipe_ppc, param_grid_ppc, cv=5, scoring="accuracy", n_jobs=-2
                    ),
                )
            )

        else:

            models = []
            models.append(("PAC", PassiveAggressiveClassifier()))
            models.append(
                ("XGB", XGBClassifier(eval_metric="logloss", use_label_encoder=False))
            )
            models.append(("LRC", LogisticRegression()))
            models.append(("PPC", Perceptron()))
            models.append(("SVC", LinearSVC()))

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, stratify=y, random_state=232
        )

        names = []
        result = []
        best_score = 0

        for name, model in models:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            score = f1_score(y_test, y_pred)
            result.append(score)
            names.append(name)

            if score > best_score:
                best_score = score
                best_clf = name
            else:
                continue

        outcome = pd.DataFrame({"Name": names, "Score": result})
        outcome = outcome.sort_values(by="Score", ascending=True)
        outcome.reset_index(drop=True, inplace=True)

        for name, model in models:
            if name == best_clf:
                clf = model
                break
            else:
                continue

        return clf, outcome


# %%
class select:
    def __init__(self, df, target, clf):
        self.df = df
        self.target = target
        self.clf = clf

    def feature(self):

        y = self.df[self.target]
        x = self.df.loc[:, self.df.columns.difference([self.target])]
        features = x.columns

        val_score_before = cross_val_score(
            self.clf, x, y, cv=5, scoring="accuracy"
        ).mean()

        rfecv = RFECV(estimator=self.clf, step=1, cv=7, scoring="accuracy")
        rfecv.fit(x, y)

        features_importance = list(zip(features, rfecv.support_))
        selected_x = []
        for key, value in enumerate(features_importance):
            if value[1] == True:
                selected_x.append(value[0])

        x1 = x[selected_x]

        val_score_after = cross_val_score(
            self.clf, x1, y, cv=5, scoring="accuracy"
        ).mean()

        if val_score_after > val_score_before:
            return selected_x, True
        else:
            return list(x.columns.values), False


# %%


# %%


# %%
