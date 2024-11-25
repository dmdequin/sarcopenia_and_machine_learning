import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score, precision_score, recall_score, f1_score


def baseline_test(datasets, y, dict_classifiers):
    """
    Baseline classification test function.

    Function that trains different classification models using a given set of
    datasets, and outputs the results in a dataframe.

    Parameters
    -----------
    datasets: dictionary of datasets
        key: name of dataset
        value: dataframe containing the dataset
    y: true class values
        pandas series
    dict_classifiers: dictionary of classifiers to be trained on
        key: name of classifier
        value: instantiation of model

    Returns
    -----------
    results_df: Dataframe containing best training results per
        model/dataset combination using gridsearch.
    best_model: List of best performing model(s).
    best_score: Float of best model score.
    """
    # Define custom scorers for multiclass support
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='macro', zero_division=0),
        'recall': make_scorer(recall_score, average='macro'),
        # ROC AUC scorer only for models that support predict_proba or decision_function
        'roc_auc': make_scorer(
            roc_auc_score, needs_proba=True, multi_class='ovr', average='macro'
        ),
        'f1': make_scorer(f1_score, average='macro') 
    }

    metrics = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']

    # Columns for dataframe - names of each classifier tested
    cols = list(set(dict_classifiers.keys()))

    # MultiIndex DataFrame for results
    index = pd.MultiIndex.from_product([datasets.keys(), metrics], names=['dataset', 'metric'])
    results_df = pd.DataFrame(columns=cols, index=index, dtype='float64')

    best_model = []
    best_score = 0

    # Select a dataset
    for dataset_name, X in tqdm(datasets.items()):

        # model = name of model
        for model_name, model_instance in dict_classifiers.items():
            try:
                # Create k-fold object
                k_folds = KFold(n_splits=5)

                # Use cross_validate to compute multiple metrics
                scores = cross_validate(
                    model_instance, X, y, cv=k_folds, scoring=scoring, return_train_score=False
                )

                # Store mean metric scores in the results DataFrame
                results_df.loc[(dataset_name, 'accuracy'), model_name] = round(scores['test_accuracy'].mean(), 5)
                results_df.loc[(dataset_name, 'precision'), model_name] = round(scores['test_precision'].mean(), 5)
                results_df.loc[(dataset_name, 'recall'), model_name] = round(scores['test_recall'].mean(), 5)
                results_df.loc[(dataset_name, 'f1'), model_name] = round(scores['test_f1'].mean(), 5)
                # For ROC AUC, handle cases where scoring might fail
                if 'test_roc_auc' in scores:
                    results_df.loc[(dataset_name, 'roc_auc'), model_name] = round(scores['test_roc_auc'].mean(), 5)
                else:
                    results_df.loc[(dataset_name, 'roc_auc'), model_name] = None

                score = results_df.loc[(dataset_name, 'accuracy'), model_name]

                # Keep Track of best accuracy score and model
                if score > best_score:
                    best_score = score
                    best_model = [f"{model_name}:{dataset_name}"]
                elif score == best_score:
                    best_model.append(f"{model_name}:{dataset_name}")

            except ValueError as e:
                print(f"Error with model {model_name} on dataset {dataset_name}: {e}")
                # Set ROC AUC as NaN if it fails
                results_df.loc[(dataset_name, 'roc_auc'), model_name] = None

    print("------ TRAINING COMPLETE ------")

    return results_df, best_model, best_score


def get_models_and_scores(dataset, models, scores, feat_count=None):
    """
    Function that prints best models and corresponding scores
    from classification tests.

    Parameters:
    -------------
    dataset: string - Name of the dataset used in test.
    models: list of model/dataset combination best performers.
    scores: list of accuracy scores from best performing models.
    feat_count: True if test is comparing features from aggregated
                select features.
    """

    print(f"Best model(s) for datos {dataset}:")
    for model in models:
        print(model)
    if feat_count is not None:
        print(f"Highest accuracy: {scores}")
        print(f"{set(feat_count)}\n")
    else:
        print(f"Highest accuracy: {scores}\n")

    return None


def gridsearch(datasets, Y, parameters, model):
    """
    Function to run gridsearch and find the best parameters.

    Parameters:
    -------------
    datasets: Dictionary of k,v as dataset name, dataframe.
    Y: pandas series of MQI/sarcopenia true values, used for training.
    parameters: Dictionary of parameters to use in gridsearch.
        k: parameter name
        v: list of parameter values to search through
    model: instantiated model used for test.

    Returns:
    -------------
    results_df: Dataframe containing best training results per
        model/dataset combination using gridsearch.
    best_params: List of best parameters associated with each dataset.
    best_model: List of best performing model(s).
    best_score: Float of best model score.
    """

    # Make dataframe
    cols = ['dataset','accuracy', 'precision', 'recall', 'roc_auc']
    results_df = pd.DataFrame(columns=cols)
    results_df.set_index('dataset', inplace=True)

    # Define custom scorers for multiclass support
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='macro', zero_division=0),  # Adjust average as needed
        'recall': make_scorer(recall_score, average='macro'),        # Adjust average as needed
        'roc_auc': make_scorer(
            roc_auc_score, needs_proba=True, multi_class='ovr', average='macro' # needs_threshold=True,
        ),
        'f1': make_scorer(f1_score, average='macro') 
    }

    best_params = {}
    best_model = []
    best_score = 0

    # loop through datasets - dataset=dataset name,  X=dataset
    for dataset, X in tqdm(datasets.items()):

        # Add row to dataframe with dataset as index
        results_df.loc[dataset] = pd.Series(dtype='float64')

        # Instantiate gridsearch with params. Automatically uses 5-fold CV
        cv = GridSearchCV(model, parameters, n_jobs=-1, scoring=scoring, refit='roc_auc') # refit='accuracy' #####

        cv.fit(X, Y)

        # Add best params to dictionary
        best_params[dataset] = cv.best_params_
        print(cv.best_params_)

        # mean cv score of the best performing model
        score = round(cv.best_score_, 5)

        # Add metric to results dataframe
        results_df.loc[dataset, 'accuracy'] = score
        results_df.loc[dataset, 'precision'] = round(cv.cv_results_['mean_test_precision'][cv.best_index_], 5)
        results_df.loc[dataset, 'recall'] = round(cv.cv_results_['mean_test_recall'][cv.best_index_], 5)
        results_df.loc[dataset, 'roc_auc'] = round(cv.cv_results_['mean_test_roc_auc'][cv.best_index_], 5)
        results_df.loc[dataset, 'f1'] = round(cv.cv_results_['mean_test_f1'][cv.best_index_], 5)

        # Keep Track of best score and model
        if score > best_score:
            best_score = score
            best_model = [f"{model}:{dataset}"]
        elif score == best_score:
            best_model.append(f"{model}:{dataset}")

    print("------ TRAINING COMPLETE ------")

    return results_df, best_params, best_model, best_score, cv.cv_results_


def compare_models_with_best_params(datasets, y, dict_classifiers):
    """
    Function to run classification with various
    models and datasets using the best parameters.

    Parameters:
    -------------
    datasets: Dictionary of k, v as dataset name, dataframe.
    Y: pandas series of MQI/sarcopenia true values, used for training.
    dict_classifiers: dictionary of models that have been given
        best parameters that correspond to each dataset.
        Includes a pipeline with scaling as needed.
        k, v is dataset name, instantiated model/pipeline.

    Returns:
    -------------
    results_df: Dataframe containing best training results per
        model/dataset combination.
    best_model: List of best performing model(s).
    best_score: Float of best model score.
    """

    # columns for dataframe
    cols = list(set(dict_classifiers.keys()))
    cols.append('dataset')

    # make dataframe
    results_df = pd.DataFrame(columns=cols)
    results_df.set_index('dataset', inplace=True)

    results = []
    best_model = []
    best_score = 0

    # Select a dataset
    for key, X in tqdm(datasets.items()):  # key = name of dataset

        # Add row with name of dataset to dataframe with empty rows
        results_df.loc[key] = pd.Series(dtype='float64')

        # model = name of model
        for model in dict_classifiers.keys():

            # Create k-fold object
            k_folds = KFold(n_splits=5)

            # Get cross validation scores
            scores = cross_val_score(dict_classifiers[model][key],
                                     X, y, cv=k_folds)
            score = round(scores.mean(), 5)

            results.append(score)

            # Add to dataframe
            results_df.loc[key][model] = score

            if score > best_score:
                best_score = score
                best_model = [f"{model}:{key}"]
            elif score == best_score:
                best_model.append(f"{model}:{key}")

    print("------ TRAINING COMPLETE ------")

    return results_df, best_model, best_score
