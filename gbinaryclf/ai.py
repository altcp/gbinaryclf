# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import gc

import pandas as pd
import scipy.stats as sps
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC, LinearSVC
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
                ("M", XGBClassifier(eval_metric="logloss", use_label_encoder=False)),
            ]
        )

        param_grid_xgb = {
            "M__gamma": [1, 2, 3],
            "M__max_depth": [1, 3, 5],
            "M__eta": [0.4, 0.6, 0.8, 1.0],
            "M__reg_alpha": [0.1, 0.3, 0.5, 0.7],
            "M__reg_lambda": [0.1, 0.3, 0.5, 0.7],
        }

        # GBC
        pipe_gbc = Pipeline(
            steps=[
                ("N", MinMaxScaler()),
                ("M", GradientBoostingClassifier()),
            ]
        )

        param_grid_gbc = {
            "M__loss": ["deviance"],
            "M__learning_rate": [0.01, 0.1, 0.2],
            "M__max_depth": [3, 5, 8],
            "M__max_features": ["log2", "sqrt"],
            "M__criterion": ["friedman_mse"],
            "M__subsample": [0.5, 0.75, 1.0],
            "M__n_estimators": [10],
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
            "M__alpha": [1e-10, 1e7, 1e-5, 1e-3],
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
            models.append(
                (
                    "GBC",
                    GridSearchCV(
                        pipe_gbc, param_grid_gbc, cv=5, scoring="accuracy", n_jobs=-2
                    ),
                )
            )
            models.append(
                ("KNN", make_pipeline(MinMaxScaler(), KNeighborsClassifier()))
            )
            models.append(
                (
                    "XGB",
                    GridSearchCV(
                        pipe_xgb, param_grid_xgb, cv=5, scoring="accuracy", n_jobs=-2
                    ),
                )
            )
            models.append(("LRC", make_pipeline(MinMaxScaler(), LogisticRegression())))
            models.append(("SVC", make_pipeline(MinMaxScaler(), SVC())))

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

        # Cross Validation (BEFORE)
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
            selection = True
            return selected_x, selection
        else:
            selection = False
            return x.columns, selection


# %%


# %%


# %%
