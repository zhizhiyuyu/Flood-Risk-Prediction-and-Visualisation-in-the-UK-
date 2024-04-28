"""Test Module."""

import pandas as pd

pd.options.mode.chained_assignment = None

import numpy as np
import glob
import sys
import os

sys.path.append('..')

from flood_tool import *

tool = Tool()

def test_get_easting_northing():
    """Check """

    data = tool.get_easting_northing(['BN1 5PF'])

    assert np.isclose(data.iloc[0].easting, 530401.0).all()
    assert np.isclose(data.iloc[0].northing, 105619.0).all()

def test_get_lat_long():
    """Check """

    data = tool.get_lat_long(['BN1 5PF'])

    assert np.isclose(data.iloc[0].latitude, 50.8354, 1.0e-3).all()
    assert np.isclose(data.iloc[0].longitude, -0.1495, 1.0e-3).all()

def test_get_flood_class_from_postcodes_knc():
    """Check """

    postcodes = ['BN1 5PF', 'BN7 2HP']
    classes = list(range(1, 11))
    data = tool.get_flood_class_from_postcodes(postcodes)

    model = tool.model

    assert type(model).__name__ == 'KNeighborsClassifier'
    assert data.values.all() in classes
    assert type(data) == pd.Series

def test_get_flood_class_from_postcodes_gbr():
    """Check """

    postcodes = ['BN1 5PF', 'BN7 2HP']
    classes = list(range(1, 11))
    data = tool.get_flood_class_from_postcodes(postcodes, method=1)

    model = tool.model

    assert type(model).__name__=='GradientBoostingRegressor'
    assert data.values.all() in classes
    assert type(data) == pd.Series

def test_get_flood_class_from_postcodes_kncc():
    """Check """

    postcodes = ['BN1 5PF', 'BN7 2HP']
    classes = list(range(1, 11))
    data = tool.get_flood_class_from_postcodes(postcodes, method=2)

    model = tool.model

    assert type(model).__name__=='KNeighborsClassifier'
    assert data.values.all() in classes
    assert type(data) == pd.Series

def test_get_flood_class_from_postcodes_fake():
    """Check """

    postcodes = ['BN1 5PF', 'BN7 2HP']
    classes = list(range(1, 11))
    data = tool.get_flood_class_from_postcodes(postcodes, method=0.1)

    model = tool.model

    assert type(model).__name__=='KNeighborsClassifier'
    assert data.values.all() in classes
    assert type(data) == pd.Series

def tests_get_flood_class_from_OSGB36_locations_knc():
    """Check """

    eastings = [530401, 541934]
    northings = [105619, 110957]
    classes = list(range(1, 11))
    data = tool.get_flood_class_from_OSGB36_locations(eastings, northings)

    model = tool.model

    assert type(model).__name__=='KNeighborsClassifier'
    assert data.values.all() in classes
    assert type(data) == pd.Series

def tests_get_flood_class_from_OSGB36_locations_gbr():
    """Check """

    eastings = [530401, 541934]
    northings = [105619, 110957]
    classes = list(range(1, 11))
    data = tool.get_flood_class_from_OSGB36_locations(eastings, northings, method = 1)

    model = tool.model

    assert type(model).__name__=='GradientBoostingRegressor'
    assert data.values.all() in classes
    assert type(data) == pd.Series

def tests_get_flood_class_from_OSGB36_locations_kncc():
    """Check """

    eastings = [530401, 541934]
    northings = [105619, 110957]
    classes = list(range(1, 11))
    data = tool.get_flood_class_from_OSGB36_locations(eastings, northings, method = 2)

    model = tool.model

    assert type(model).__name__=='KNeighborsClassifier'
    assert data.values.all() in classes
    assert type(data) == pd.Series

def tests_get_flood_class_from_OSGB36_locations_fake():
    """Check """

    eastings = [530401, 541934]
    northings = [105619, 110957]
    classes = list(range(1, 11))
    data = tool.get_flood_class_from_OSGB36_locations(eastings, northings, method = 0.1)

    model = tool.model

    assert type(model).__name__=='KNeighborsClassifier'
    assert data.values.all() in classes
    assert type(data) == pd.Series

def tests_get_flood_class_from_WGS84_locations_knc():
    """Check """

    postcodes = ['BN1 5PF', 'BN7 2HP']
    data = tool.get_lat_long(postcodes)
    latitudes = data.latitude
    longitudes = data.longitude

    classes = list(range(1, 11))
    data = tool.get_flood_class_from_OSGB36_locations(latitudes,longitudes)

    model = tool.model

    assert type(model).__name__=='KNeighborsClassifier'
    assert data.values.all() in classes
    assert type(data) == pd.Series

def tests_get_flood_class_from_WGS84_locations_gbr():
    """Check """

    postcodes = ['BN1 5PF', 'BN7 2HP']
    data = tool.get_lat_long(postcodes)
    latitudes = data.latitude
    longitudes = data.longitude

    classes = list(range(1, 11))
    data = tool.get_flood_class_from_OSGB36_locations(latitudes,longitudes, method = 1)

    model = tool.model

    assert type(model).__name__=='GradientBoostingRegressor'
    assert data.values.all() in classes
    assert type(data) == pd.Series

def tests_get_flood_class_from_WGS84_locations_kncc():
    """Check """

    postcodes = ['BN1 5PF', 'BN7 2HP']
    data = tool.get_lat_long(postcodes)
    latitudes = data.latitude
    longitudes = data.longitude

    classes = list(range(1, 11))
    data = tool.get_flood_class_from_OSGB36_locations(latitudes,longitudes, method=2)

    model = tool.model

    assert type(model).__name__=='KNeighborsClassifier'
    assert data.values.all() in classes
    assert type(data) == pd.Series

def tests_get_flood_class_from_WGS84_locations_fake():
    """Check """

    postcodes = ['BN1 5PF', 'BN7 2HP']
    data = tool.get_lat_long(postcodes)
    latitudes = data.latitude
    longitudes = data.longitude

    classes = list(range(1, 11))
    data = tool.get_flood_class_from_OSGB36_locations(latitudes,longitudes, method=0.1)

    model = tool.model

    assert type(model).__name__=='KNeighborsClassifier'
    assert data.values.all() in classes
    assert type(data) == pd.Series

def test_get_median_house_price_estimate_rfr():
    """Check """

    postcodes = ['BN1 5PF', 'BN7 2HP']
    data = tool.get_median_house_price_estimate(postcodes)

    model = tool.model

    assert type(data) == pd.Series
    assert data.values.all() > 0
    assert type(model).__name__ == 'RandomForestRegressor'

def test_get_median_house_price_estimate_knr():
    """Check """

    postcodes = ['BN1 5PF', 'BN7 2HP']
    data = tool.get_median_house_price_estimate(postcodes, method=1)

    model = tool.model

    assert type(data) == pd.Series
    assert data.values.all() > 0
    assert type(model).__name__ == 'KNeighborsRegressor'

def test_get_median_house_price_estimate_kncr():
    """Check """

    postcodes = ['BN1 5PF', 'BN7 2HP']
    data = tool.get_median_house_price_estimate(postcodes, method=2)

    assert type(data) == pd.Series
    assert data.values.all() > 0

def test_get_median_house_price_estimate_fake():
    """Check """

    postcodes = ['BN1 5PF', 'BN7 2HP']
    data = tool.get_median_house_price_estimate(postcodes, method=0.1)

    model = tool.model

    assert type(data) == pd.Series
    assert data.values.all() > 0
    assert type(model).__name__ == 'RandomForestRegressor'

def test_get_local_authority_estimate():
    """Check """

    eastings = [530401, 541934]
    northings = [105619, 110957]
    authorities = set(pd.read_csv(os.sep.join((os.path.dirname(__file__), '../resources', 'postcodes_sampled.csv')))['localAuthority'].values)
    data = tool.get_local_authority_estimate(eastings, northings)

    assert data.values.all() in authorities
    assert type(data) == pd.Series

def test_get_local_authority_estimate_fake():
    """Check """

    eastings = [530401, 541934]
    northings = [105619, 110957]
    authorities = set(pd.read_csv(os.sep.join((os.path.dirname(__file__), '../resources', 'postcodes_sampled.csv')))['localAuthority'].values)
    data = tool.get_local_authority_estimate(eastings, northings, method = 0.1)

    assert data.values.all() in authorities
    assert type(data) == pd.Series

def test_get_total_value():
    """Check """

    postcodes = ['BN1 5PF', 'BN7 2HP']
    data = tool.get_total_value(postcodes)

    assert type(data) == pd.Series

def test_get_annual_flood_risk():
    """Check """

    postcodes = ['BN1 5PF', 'BN7 2HP']
    data = tool.get_annual_flood_risk(postcodes)

    assert type(data) == pd.Series

def test_get_augmented_training_data():
    """Check """

    data = tool.get_augmented_training_data()

    assert len(data) == 40000
    assert len(data[0]) == 5
    assert np.isnan(data).all() == False

def test_augment_prediction_data():
    """Check """

    eastings = [530401, 541934]
    northings = [105619, 110957]
    X_pred = pd.DataFrame({'easting': eastings, 'northing': northings})

    data = tool.augment_prediction_data(X_pred)

    assert len(data) == 2
    assert len(data[0]) == 5
    assert np.isnan(data).all() == False


files = glob.glob('*.csv')
for f in files:
    os.remove(f)

#NEED TO TEST CONTROL FLOW!!!!!
