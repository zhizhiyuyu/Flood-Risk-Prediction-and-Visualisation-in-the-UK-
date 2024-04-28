import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

pd.options.mode.chained_assignment = None

def train_medianPrice_model(data, threshold=1000000):
    """
    Gets trained model (as three sub-models) for classified regression of median house price.

    Parameters
    ----------

    data : pandas.DataFrame
         Labelled geographic location data for postcodes
    threshold : int (optional)
        Value specifying split between 'low' and 'high' house price

    Returns
    -------

    tuple
        Three models:
        - KNNClassifier to predict the high/low price tag
        - KNNRegressor to predict the median price for 'high' price houses
        - KNNRegressor to predict the median price for 'low' price houses
    """

    data_train = pd.DataFrame(data = data[['easting', 'northing', 'medianPrice']], columns=['easting', 'northing', 'medianPrice'])
    data_train['high_low']=0
    data_train['high_low'][data_train['medianPrice']>=threshold] = 1

    data_high_train = data_train[data_train['high_low']==1]
    data_low_train = data_train[data_train['high_low']==0]

    X_hl_train = data_train.drop(columns=['medianPrice', 'high_low'])
    y_hl_train = data_train.high_low

    X_hprice_train = data_high_train.drop(columns=['medianPrice', 'high_low'])
    y_hprice_train = data_high_train.medianPrice
    X_lprice_train = data_low_train.drop(columns=['medianPrice', 'high_low'])
    y_lprice_train = data_low_train.medianPrice

    knn_class = KNeighborsClassifier(n_neighbors = 11, weights ='distance')
    knn_class.fit(X_hl_train, y_hl_train)

    knn_high = KNeighborsRegressor(n_neighbors=17, weights='distance')
    knn_high = knn_high.fit(X_hprice_train, y_hprice_train)

    knn_low = KNeighborsRegressor(n_neighbors=15, weights='distance')
    knn_low = knn_low.fit(X_lprice_train, y_lprice_train)

    return knn_class, knn_high, knn_low