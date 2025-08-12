import mlflow
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score


def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")


def objective(trial, X_train, y_train, experiment_id):
    """
    Optimize hyperparameters for a classifier using Optuna.

    Parameters:
    -----------
    trial : optuna.trial.Trial
        A trial is a process of evaluating an objective function.
    X_train : pandas.DataFrame
        Input features for training.
    y_train : pandas.Series
        Target variable for training.
    experiment_id : int
        ID of the MLflow experiment where results will be logged.

    Returns:
    --------
    float
        Mean F1 score of the classifier after cross-validation.
    """

    # Comienza el run de MLflow. Este run debería ser el hijo del run padre.
    with mlflow.start_run(experiment_id=experiment_id, 
                          run_name=f"Trial: {trial.number}", nested=True):

        # Parámetros a logguear
        params = {}

        # Sugiere valores para los hiperparámetros utilizando el objeto trial de optuna.
        classifier_name = trial.suggest_categorical('classifier', [
            'SVC_linear', 
            'SVC_poly', 
            'SVC_rbf',
            'DecisionTreeClassifier', 
            'RandomForest',
            'LogisticRegression',
            'XGBoost'
        ])

        if 'SVC' in classifier_name:
            # Support Vector Classifier (SVC)
            params["model"] = "SVC"
            svc_c = trial.suggest_float('svc_c', 0.01, 100, log=True)  # Regularización
            kernel = 'linear'
            degree = 3

            if classifier_name == 'SVC_poly':
                degree = trial.suggest_int('svc_poly_degree', 2, 6)  # Grado del polinomio
                kernel = 'poly'
                params["degree"] = degree
            elif classifier_name == 'SVC_rbf':
                kernel = 'rbf'

            params["kernel"] = kernel
            params["C"] = svc_c

            classifier_obj = SVC(C=svc_c, kernel=kernel, gamma='scale', degree=degree)

        elif classifier_name == 'DecisionTreeClassifier':
            # Decision Tree Classifier
            tree_max_depth = trial.suggest_int("tree_max_depth", 2, 32, log=True)
            classifier_obj = DecisionTreeClassifier(max_depth=tree_max_depth)
            params["model"] = "DecisionTreeClassifier"
            params["max_depth"] = tree_max_depth

        elif classifier_name == 'RandomForest':
            # Random Forest Classifier
            rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
            rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 200, log=True)
            classifier_obj = RandomForestClassifier(max_depth=rf_max_depth, 
                                                    n_estimators=rf_n_estimators)
            params["model"] = "RandomForestClassifier"
            params["max_depth"] = rf_max_depth
            params["n_estimators"] = rf_n_estimators

        elif classifier_name == 'LogisticRegression':
            # Logistic Regression
            logreg_c = trial.suggest_float("logreg_c", 0.01, 100, log=True)
            classifier_obj = LogisticRegression(C=logreg_c, solver='liblinear')
            params["model"] = "LogisticRegression"
            params["C"] = logreg_c

        elif classifier_name == 'XGBoost':
            # XGBoost Classifier
            xgb_max_depth = trial.suggest_int("xgb_max_depth", 3, 10)
            xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 50, 300)
            xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.3)
            classifier_obj = XGBClassifier(max_depth=xgb_max_depth, 
                                           n_estimators=xgb_n_estimators, 
                                           learning_rate=xgb_learning_rate)
            params["model"] = "XGBoost"
            params["max_depth"] = xgb_max_depth
            params["n_estimators"] = xgb_n_estimators
            params["learning_rate"] = xgb_learning_rate

        # Realizamos validación cruzada y calculamos el score F1
        score = cross_val_score(classifier_obj, X_train, y_train.to_numpy().ravel(), 
                                n_jobs=-1, cv=5, scoring='f1')

        # Logueo de los hiperparámetros y el F1 en MLflow
        mlflow.log_params(params)
        mlflow.log_metric("f1", score.mean())

    return score.mean()
