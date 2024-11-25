import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.decomposition import PCA


def standardize_yo(data, location, columns):
    """Function to standardize and normalize data."""

    # Instantiate Scaler and scale data (bring values between 0 and 1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # normalize data (change shape to fit normal distribution)
    standardized_data = normalize(scaled_data)

    # Put back into dataframe
    standardized_data = pd.DataFrame(standardized_data, columns=columns)

    # Save to csv
    standardized_data.to_csv('../data/'+location)

    return standardized_data


def get_cor_variables(cor_data, target, threshold):
    """
    Function that finds features in a dataset correlated to a target.

    Parameters:
    -------------
    cor_data : correlation data - result of data.corr()
    target : target variable to be removed from correlation data
    threshold : threshold for selecting correlated variables

    Returns:
    -------------
    ranked_corr : dataframe with ranked correlated features
    correlated_features : list of features correlated to target

    """
    # Set aside target
    cor_target = cor_data[target]

    # Select features most highly correlated to target
    relevant_features = abs(cor_target[(
        cor_target > threshold) | (cor_target < -threshold)])

    # Create df to track ranking
    ranked_corr = pd.DataFrame(relevant_features).drop(target, axis=0)

    # List of correlated features
    correlated_features = list(relevant_features.index)

    # Remove target from list of correlated features
    correlated_features.remove(target)

    return ranked_corr, correlated_features


def create_rank_df(ranked_corr_features, attributes):
    """
    Function to create feature ranking dataframe
    starting with ranked features from pearson correlation.

    Parameters:
    -----------
    ranked_corr_features : ranked features selected using pearson correlation
    attributes : list of features to be ranked

    Returns:
    -----------
    feature_ranking : dataframe
    """
    # Make dataframe with ranking = index+1
    feature_ranking = pd.DataFrame(index=attributes)

    feat_count = len(attributes)

    # Add empty column filled with 21 (last place)
    feature_ranking['corr'] = np.full(feat_count, feat_count, dtype=int)

    # loop through selected, ranked features
    for i, row in ranked_corr_features.iterrows():
        rank = i + 1  # calculate rank based on index+1
        feature = row['feat']
        # Add rank
        feature_ranking.loc[feature]['corr'] = rank

    return feature_ranking


def ols_feature_selection(data, Y):
    """
    Function for selecting ranked features using OLS.

    Parameters:
    -----------
    data : dataframe containing features for training
    Y : pandas series with target feature

    Returns:
    ranked_ols : dataframe containing selected and ranked features
    sel_feat_ols : list of selected features
    """
    # Backward Elimination
    sel_feat_ols = list(data.columns)
    pmax = 1
    ranked_ols = pd.DataFrame()

    while (len(sel_feat_ols) > 0):
        p = []
        x_1 = data[sel_feat_ols]

        # Add constant column of ones, mandatory for sm.OLS model
        x_1 = sm.add_constant(x_1)

        if 'const' in x_1.columns:
            x_1 = x_1.drop('const', axis=1)

        model = sm.OLS(Y, x_1).fit()  # Fit sm.OLS model
        p = pd.Series(model.pvalues.values, index=sel_feat_ols)  # pvalues

        # select highest p-value
        pmax = max(p)  # highest p value
        feature_with_p_max = p.idxmax()  # feature with max p value

        # remove feature with highest p-value if above 0.05
        if (pmax > 0.05):
            sel_feat_ols.remove(feature_with_p_max)
        else:
            ranked_ols = pd.DataFrame(p).reset_index()\
                    .sort_values(0, ascending=True)\
                    .rename(columns={'index': 'feat', 0: 'ols'})\
                    .drop('ols', axis=1)\
                    .reset_index().drop('index', axis=1)

            return ranked_ols, sel_feat_ols


def add_to_feat_rank_df(feat_rank_df, ranked_new_feat, column_name):
    """
    Function to add feature ranking from other selection methods
    to feature ranking dataframe.

    Parameters:
    -----------
    feat_rank_df : dataframe containing feature ranking data
    ranked_new_feat : dataframe of ranked features from new method
    column : name of feature selection method to name column in df

    Returns:
    -----------
    feat_rank_df : original feature ranking df with new column added
    """

    feat_count = feat_rank_df.shape[0]

    # Add empty column filled with 22 (last place)
    feat_rank_df[column_name] = np.full(feat_count, feat_count, dtype=int)

    # loop through selected, ranked features
    for i, row in ranked_new_feat.iterrows():
        rank = i + 1  # calculate rank based on index+1
        feature = row['feat']

        # Add rank
        feat_rank_df.at[feature, column_name] = rank

    feat_rank_df[column_name] = feat_rank_df[column_name]

    return feat_rank_df


def random_for_feat_sel(data, Y, path):
    """
    Select features using random forest and save dataset.

    Parameters:
    -----------
    data : dataframe of features used for training
    Y : pandas series of dependent variable
    path : full path to save data subset

    Returns:
    -----------
    X_forest : dataframe subset including only selected features
    ranked_for : dataframe of selected and ranked features
    """

    # Create train and test sets
    x_train, x_test, y_train, y_test = train_test_split(data, Y, test_size=0.3)

    # Instantiate SelectFromModel transformer to select features
    sel = SelectFromModel(RandomForestClassifier(n_estimators=100))

    # Fit training data
    sel.fit(x_train, y_train)

    # Selected features based on feature importance
    selected_feat = x_train.columns[(sel.get_support())]

    # create subset of data based on random forest feature selection
    x_forest = data[selected_feat]

    # number of features selected
    feat_count = len(list(x_forest.columns))

    # Save to csv
    x_forest.to_csv('../data/'+path)

    # Instantiate model
    forest = RandomForestClassifier(n_estimators=100)

    # Fit to training data
    forest.fit(x_train, y_train)

    # Get feature importances to use for ranking
    importances = forest.feature_importances_

    # Use rankings to put features into a dataframe
    forest_importances = pd.DataFrame(importances, index=data.columns)\
        .reset_index()\
        .sort_values(0, ascending=False)\
        .rename(columns={'index': 'feat', 0: 'importance'})\
        .drop('importance', axis=1)\
        .reset_index().drop('index', axis=1)

    # Select same number of features selected as from SelectFromModel
    ranked_for = forest_importances.iloc[0:feat_count]

    return x_forest, ranked_for


def svm_feat_sel(data_norm, data, Y, path, threshold):
    """
    Function for selecting features using SVM.

    Parameters:
    -----------
    data_norm : dataframe of normalized features used for training
    data: dataframe of features used to create subset
    Y : pandas series of dependent variable
    path : full path to save data subset
    threshold : float used for selecting features with coeficient to remove

    Returns:
    -----------
    X_svm : dataframe subset including only selected features
    ranked_svm : dataframe of selected and ranked features
    """

    # Backward Elimination
    selected_features_svm = data_norm.columns.to_list()

    # Current number of features
    feat_count = len(selected_features_svm)

    # Original number of features
    original_feat_count = len(selected_features_svm)

    while (feat_count > 0):
        coefs = []

        # Make subset based on remaining features
        x_1 = data_norm[selected_features_svm]

        model = svm.SVC(kernel='linear', random_state=42)
        model.fit(x_1, Y)

        # Calculate average coef
        coef_svm = pd.DataFrame(model.coef_).transpose()
        coef_svm['sum_abs'] = coef_svm.abs().sum(axis=1)
        coefs = pd.Series(coef_svm['sum_abs'])

        # get the index of the lowest coef
        feat_index_with_coef_min = coefs.idxmin()

        # Select lowest coef
        feature_with_coef_min = selected_features_svm[feat_index_with_coef_min]

        if (feat_count/original_feat_count > threshold):
            selected_features_svm.remove(feature_with_coef_min)
            feat_count -= 1
        else:
            # DF with remainging features and corresponding weight averages
            select_feat = pd.DataFrame(columns=['feat', 'weight'])
            select_feat['feat'] = selected_features_svm
            select_feat['weight'] = coefs
            select_feat.sort_values('weight', inplace=True, ascending=False)

            break

    # save subset of data based on selected features
    x_svm = data[selected_features_svm]

    ranked_svm = select_feat.reset_index()\
        .drop('index', axis=1)\

    # Save to csv
    x_svm.to_csv('../data/'+path)

    return x_svm, ranked_svm


def feature_counts(pear, ols, forest, svm):
    """
    Function to compute the count that each feature
    is selected using all four methods.

    Parameters:
    -----------
    pear: Pearson selected dataset.
    ols: OLS selected dataset.
    forest: Random Forest selected dataset.
    svm: SVM selected dataset.

    Returns:
    -----------
    feature_counts: DataFrame of all selected features,
                    1 if selected per method,
                    0 if not selected
    feat: Transpose of feature_counts
    """

    pear_feat = list(pear.columns)
    ols_feat = list(ols.columns)
    forest_feat = list(forest.columns)
    svm_feat = list(svm.columns)

    columns = list(pear.columns)
    columns.extend(ols_feat)
    columns.extend(forest_feat)
    columns.extend(svm_feat)
    columns = list(set(columns))

    # Make dataframe to track selected features
    feature_counts = pd.DataFrame(columns=columns)
    feature_counts.loc['pearson'] = np.zeros(len(columns))
    feature_counts.loc['ols'] = np.zeros(len(columns))
    feature_counts.loc['random_forest'] = np.zeros(len(columns))
    feature_counts.loc['svm'] = np.zeros(len(columns))

    for i in range(len(pear_feat)):
        feature_counts[pear_feat[i]].loc['pearson'] = 1

    for i in range(len(ols_feat)):
        feature_counts[ols_feat[i]].loc['ols'] = 1

    for i in range(len(forest_feat)):
        feature_counts[forest_feat[i]].loc['random_forest'] = 1

    for i in range(len(svm_feat)):
        feature_counts[svm_feat[i]].loc['svm'] = 1

    # Transpose dataframes and sum to to sort features by frequency selected
    feat = feature_counts.transpose()
    feat['count'] = feat.sum(axis=1)
    feat = feat.sort_values('count', ascending=False)

    return feature_counts, feat


def compute_ranking(rank_data, path):
    """
    Function to create final ranking of combined selected features.

    Prameters:
    ----------
    rank_data : dataframe containing rank of all selected features
    path :

    Returns:
    ----------
    rank data : dataframe with rank column added
    """

    never_chosen = rank_data.shape[0] * 4

    rank_data['rank_sum'] = rank_data.sum(axis=1)  # Column for sum of rankings
    rank_data['rank'] = 1/rank_data['rank_sum']  # Compute rank

    # Sort by rank_sum
    rank_data = rank_data.sort_values('rank_sum', ascending=True)

    # Drop if never selected
    rank_data = rank_data.drop(
        rank_data.loc[rank_data['rank_sum'] == never_chosen].index
        )
    rank_data.to_csv('../data/'+path)

    return rank_data


def get_components(data, path):
    """
    Function for finding the principal components
    that explain 95% of the data.
    Prints the % that each component explains.

    Parameters:
    ----------
    data:
    path:

    Returns:
    ----------
    X_pca_95: DataFrame of the PCA reduced data.

    """
    # .95 : the ratio of variance to preserve
    pca = PCA(n_components=.95)
    pca.fit(data)

    # Transformed data
    x_pca_95 = pd.DataFrame(pca.transform(data))

    # Explained variance per component
    pca_exp = pca.explained_variance_ratio_

    cum_explained_var = 0
    components = 1
    for i in pca_exp:
        cum_explained_var += i
        var = round(100*cum_explained_var, 2)
        print(f'{components} components explain {var}% of the data')
        components += 1

    print("")

    # Save to csv
    x_pca_95.to_csv(path)

    return x_pca_95
