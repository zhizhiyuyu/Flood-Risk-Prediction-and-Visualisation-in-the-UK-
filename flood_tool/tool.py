"""Flood risk prediction tool """

import os

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from .models import knn_c_r

from .geo import *

__all__ = ['Tool']

class Tool(object):
    """Class to interact with a postcode database file."""

    def __init__(self, full_postcode_file='', sample_labels='', household_file=''):
        """
        Parameters
        ----------

        full_postcode_file : str, optional
            Filename of a .csv file containing geographic location data for postcodes.

        sample_labels: str, optional
            Filename of a .csv file containing labelled geographic location data for postcodes.

        household_file : str, optional
            Filename of a .csv file containing information on households by postcode.
        """

        if full_postcode_file == '':
            full_postcode_file = os.sep.join((os.path.dirname(__file__),'resources','postcodes_unlabelled.csv'))

        if household_file == '':
            household_file = os.sep.join((os.path.dirname(__file__),'resources', 'households_per_sector.csv'))

        if sample_labels == '':
            sample_labels = os.sep.join((os.path.dirname(__file__), 'resources', 'postcodes_sampled.csv'))

        postcode_df = pd.read_csv(full_postcode_file)
        self.postcode_df = postcode_df.set_index('postcode')
        self.household_df = pd.read_csv(household_file)
        self.training_df = pd.read_csv(sample_labels)

        stations_df = pd.read_csv(os.sep.join((os.path.dirname(__file__), 'resources', 'stations.csv')))
        stations_df = stations_df.dropna()
        closest_station_model = KNeighborsClassifier(n_neighbors=1)
        X = stations_df[['latitude', 'longitude']]
        y = stations_df.stationReference
        self.closest_station_model = closest_station_model.fit(X, y)

        self.typical_day_df = pd.read_csv(os.sep.join((os.path.dirname(__file__), 'resources', 'station_typicalday.csv')))
        self.typical_day_df = self.typical_day_df[['stationReference', 'extreme_river_data', 'mean_river_data','typicalRangeHigh', 'typicalRangeLow']]

        self.imputer = SimpleImputer()
        self.scaler = StandardScaler()

    def train(self, labelled_samples='', flag=0):
        """
        Helper function for training each model using a labelled set of samples.

        Parameters
        ----------
        labelled_samples : str, optional
            Filename of a .csv file containing a labelled set of samples.

        flag : int
            Specifies data for model to be trained on.
        """

        if labelled_samples == '':
            labelled_samples = self.training_df

        if flag == 0:
            X = labelled_samples[["easting", "northing"]]
            y = labelled_samples['riskLabel']
        elif flag == 1:
            X = labelled_samples[["easting", "northing"]]
            y = labelled_samples['medianPrice']
        elif flag == 2:
            X = labelled_samples[["easting", "northing"]]
            y = labelled_samples['localAuthority']
        elif flag == 3:
            X = self.get_augmented_training_data()
            y = labelled_samples['riskLabel']

        self.model.fit(X,y)

    def get_augmented_training_data(self):
        """
        Return labelled geographic location data for postcodes augmented with
        their nearest station's rainfall data, for training the risk prediction model.

        Returns
        ----------

        pandas.DataFrame
            DataFrame containing OSGB36 easting and northing, mean river data, and typical low and high ranges
            for each postcode in the labelled training data, with missing values imputed.

        """

        training_df = self.training_df.drop_duplicates()

        lat_longs = get_gps_lat_long_from_easting_northing(training_df['easting'], training_df['northing'])
        training_df['latitude'] = lat_longs[0]
        training_df['longitude'] = lat_longs[1]

        training_df['stationReference'] = self.closest_station_model.predict(training_df[['latitude', 'longitude']])

        augmented_df = pd.merge(self.typical_day_df, training_df, on='stationReference', how='outer')

        augmented_df = augmented_df[augmented_df['postcode'].notna()]
        X_augmented = augmented_df[['easting','northing','mean_river_data','typicalRangeHigh','typicalRangeLow']]
        return self.scaler.fit_transform(self.imputer.fit_transform(X_augmented))

    def augment_prediction_data(self, X_pred):
        """
        Return unlabelled geographic location data for postcodes augmented with their
        nearest station's rainfall data, for training the risk prediction model.

        Returns
        ----------

        pandas.DataFrame
            DataFrame containing OSGB36 easting and northing, mean river data, and typical low and high ranges
            for each postcode in the unlabelled prediction data, with missing values imputed.

        """

        lat_longs = get_gps_lat_long_from_easting_northing(X_pred['easting'], X_pred['northing'])
        X_pred['latitude'] = lat_longs[0]
        X_pred['longitude'] = lat_longs[1]

        X_pred['stationReference'] = self.closest_station_model.predict(X_pred[['latitude', 'longitude']])

        augmented_df = pd.merge(self.typical_day_df, X_pred, on='stationReference', how='outer')

        augmented_df = augmented_df[augmented_df['easting'].notna()]
        X_augmented = augmented_df[['easting','northing','mean_river_data','typicalRangeHigh','typicalRangeLow']]
        return self.scaler.fit_transform(self.imputer.fit_transform(X_augmented))

    def get_easting_northing(self, postcodes):
        """
        Get a frame of OS eastings and northings from a collection of input postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only OSGB36 easthing and northing indexed
            by the input postcodes. Invalid postcodes (i.e. not in the
            input unlabelled postcodes file) return as NaN.
         """

        postcodes = [pc for pcs in [postcodes] for pc in pcs]
        df = pd.DataFrame(columns=(['easting', 'northing']))

        valid_postcodes = self.postcode_df.index.tolist()

        for postcode in postcodes:
            if postcode in valid_postcodes:
                df.loc[postcode] = self.postcode_df.loc[postcode]
            else:
                df.loc[postcode] = np.nan

        return df

    def get_lat_long(self, postcodes):
        """
        Get a frame containing GPS latitude and longitude information for a collection of postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only WGS84 latitude and longitude pairs for
            the input postcodes. Invalid postcodes (i.e. not in the
            input unlabelled postcodes file) return as NAN.
        """

        postcodes = [pc for pcs in [postcodes] for pc in pcs]
        df = pd.DataFrame(columns=(['latitude', 'longitude']))

        valid_postcodes = self.postcode_df.index.tolist()

        for postcode in postcodes:
            if postcode in valid_postcodes:
                df.loc[postcode] = np.concatenate(get_gps_lat_long_from_easting_northing([self.postcode_df.loc[postcode].easting], [self.postcode_df.loc[postcode].northing]), dtype=object)
            else:
                df.loc[postcode] = np.nan

        return df

    @staticmethod
    def get_flood_class_from_postcodes_methods():
        """
        Get a dictionary of available flood classification methods for postcodes.

        Returns
        -------

        dict
            Dictionary mapping classification method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_flood_class_from_postcode method.
        """

        return {'knn': 0, 'gbr': 1, 'knn_a': 2}

    def get_flood_class_from_postcodes(self, postcodes, method=0):
        """
        Generate series predicting flood probability classification
        for a collection of postcodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : int (optional)
            optionally specify (via a value in
            get_flood_class_from_postcodes_methods) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by postcodes.
        """

        print('Generating flood risk labels...')

        X_pred = self.get_easting_northing(postcodes)

        if method not in list(self.get_flood_class_from_postcodes_methods().values()):
            method = 0

        if method == self.get_flood_class_from_locations_methods()["knn"] or  method == self.get_flood_class_from_locations_methods()["knn_a"]:
            self.model = KNeighborsClassifier(algorithm = 'auto', n_neighbors = 4, weights = 'distance')
        elif method == self.get_flood_class_from_locations_methods()["gbr"]:
            self.model = GradientBoostingRegressor(subsample = 0.4, n_estimators = 100, min_samples_split = 2, max_features = 'sqrt',max_depth = 45, loss = 'quantile',learning_rate = 0.1, alpha = 0.7)


        if method == self.get_flood_class_from_locations_methods()["knn"]:
            self.train(flag=0)
            y_pred = self.model.predict(X_pred)
        else:
            self.train(flag=3)
            y_pred = np.asarray(np.round(self.model.predict(self.augment_prediction_data(X_pred))), dtype=int)
            y_pred[y_pred < 1] = 1
            y_pred[y_pred > 10] = 10

        return pd.Series(y_pred,index=np.asarray(postcodes, dtype=object))

    @staticmethod
    def get_flood_class_from_locations_methods():
        """
        Get a dictionary of available flood probablity classification methods for locations.

        Returns
        -------

        dict
            Dictionary mapping classification method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_flood_class_from_OSGB36_locations and
             get_flood_class_from_OSGB36_locations method.
        """

        return {'knn': 0, 'gbr': 1, 'knn_a': 2}

    def get_flood_class_from_OSGB36_locations(self, eastings, northings, method=0):
        """
        Generate series predicting flood probability classification for a collection of OSGB36_locations.

        Parameters
        ----------

        eastings : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : int (optional)
            optionally specify (via a value in
            self.get_flood_class_from_locations_methods) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by locations.
        """

        print('Generating flood risk labels...')

        X_pred = pd.DataFrame({'easting': eastings, 'northing': northings})

        if method not in list(self.get_flood_class_from_locations_methods().values()):
            method = 0

        if method == self.get_flood_class_from_locations_methods()["knn"] or  method == self.get_flood_class_from_locations_methods()["knn_a"]:
            self.model = KNeighborsClassifier(algorithm = 'auto', n_neighbors = 4, weights = 'distance')
        elif method == self.get_flood_class_from_locations_methods()["gbr"]:
            self.model = GradientBoostingRegressor(subsample = 0.4, n_estimators = 100, min_samples_split = 2, max_features = 'sqrt',max_depth = 45, loss = 'quantile',learning_rate = 0.1, alpha = 0.7)


        if method == self.get_flood_class_from_locations_methods()["knn"]:
            self.train(flag=0)
            y_pred = self.model.predict(X_pred)
        else:
            self.train(flag=3)
            y_pred = np.asarray(np.round(self.model.predict(self.augment_prediction_data(X_pred))), dtype=int)
            y_pred[y_pred < 1] = 1
            y_pred[y_pred > 10] = 10

        return pd.Series(y_pred, index=pd.MultiIndex.from_tuples((eastings, northings)))

    def get_flood_class_from_WGS84_locations(self, longitudes, latitudes, method=0):
        """
        Generate series predicting flood probability classification for a collection of WGS84 datum locations.

        Parameters
        ----------

        longitudes : sequence of floats
            Sequence of WGS84 longitudes.
        latitudes : sequence of floats
            Sequence of WGS84 latitudes.
        method : int (optional)
            optionally specify (via a value in
            self.get_flood_class_from_locations_methods) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by locations.
        """

        print('Generating flood risk labels...')

        eastings, northings = get_easting_northing_from_gps_lat_long(latitudes, longitudes)

        X_pred = pd.DataFrame({'easting': eastings, 'northing': northings})

        if method not in list(self.get_flood_class_from_locations_methods().values()):
            method = 0

        if method == self.get_flood_class_from_locations_methods()["knn"] or  method == self.get_flood_class_from_locations_methods()["knn_a"]:
            self.model = KNeighborsClassifier(algorithm = 'auto', n_neighbors = 4, weights = 'distance')
        elif method == self.get_flood_class_from_locations_methods()["gbr"]:
            self.model = GradientBoostingRegressor(subsample = 0.4, n_estimators = 100, min_samples_split = 2, max_features = 'sqrt',max_depth = 45, loss = 'quantile',learning_rate = 0.1, alpha = 0.7)

        if method == self.get_flood_class_from_locations_methods()["knn"]:
            self.train(flag=0)
            y_pred = self.model.predict(X_pred)
        else:
            self.train(flag=3)
            y_pred = np.asarray(np.round(self.model.predict(self.augment_prediction_data(X_pred))), dtype=int)
            y_pred[y_pred < 1] = 1
            y_pred[y_pred > 10] = 10

        return pd.Series(y_pred, index=pd.MultiIndex.from_tuples((longitudes, latitudes)))

    @staticmethod
    def get_house_price_methods():
        """
        Get a dictionary of available flood house price regression methods.

        Returns
        -------

        dict
            Dictionary mapping regression method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_median_house_price_estimate method.
        """

        return {'rf': 0, 'knr': 1, 'kncr': 2}

    def get_median_house_price_estimate(self, postcodes, method=0):
        """
        Generate series predicting median house price for a collection of postcodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : int (optional)
            optionally specify (via a value in
            self.get_house_price_methods) the regression
            method to be used.

        Returns
        -------

        pandas.Series
            Series of median house price estimates indexed by postcodes.
        """

        print('Calculating property prices...')

        X_pred = self.get_easting_northing(postcodes)

        if method not in list(self.get_house_price_methods().values()):
            method = 0

        if method == self.get_house_price_methods()["rf"]:
            self.model = RandomForestRegressor(n_estimators=480)
        elif method == self.get_house_price_methods()["knr"]:
            self.model = KNeighborsRegressor(n_neighbors=9, weights='distance')
        elif method == self.get_house_price_methods()["kncr"]:
            classified_knn_model = knn_c_r.train_medianPrice_model(self.training_df)

        if method == self.get_house_price_methods()["kncr"]:
            y_pred = []
            for i, row in X_pred.iterrows():
                row_df = pd.DataFrame(data=row.values.reshape(1, 2), columns=['easting', 'northing'])
                if classified_knn_model[0].predict(row_df) == 1:
                    y_pred.append(classified_knn_model[1].predict(row_df))
                else:
                    y_pred.append(classified_knn_model[2].predict(row_df))
        else:
            self.train(flag=1)
            y_pred = self.model.predict(X_pred)

        return pd.Series(y_pred,index=np.asarray(postcodes, dtype=object))

    @staticmethod
    def get_local_authority_methods():
        """
        Get a dictionary of available local authority classification methods.

        Returns
        -------

        dict
            Dictionary mapping regression method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_altitude_estimate method.
        """

        return {'knn': 0}

    def get_local_authority_estimate(self, eastings, northings, method=0):
        """
        Generate series predicting local authorities for a sequence of OSGB36 locations.

        Parameters
        ----------

        eastings : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : int (optional)
            optionally specify (via a value in self.get_local_authority_methods)
            the regression method to be used.

        Returns
        -------

        pandas.Series
            Series of local authorities indexed by postcodes.
        """

        X_pred = pd.DataFrame({'easting': eastings, 'northing': northings})

        if method not in list(self.get_local_authority_methods().values()):
            method = 0

        if method == self.get_local_authority_methods()["knn"]:
            self.model = KNeighborsClassifier(algorithm = 'auto', n_neighbors = 4, weights = 'distance')

        self.train(flag=2)

        y_pred = self.model.predict(X_pred)

        return pd.Series(y_pred, index=pd.MultiIndex.from_tuples((eastings, northings)))

    def calc_households_per_postcode(self, household_file, postcode_file):
        """
        Calculates the average number of households per postcode

        Parameters
        ----------

        household_file : csv file
            filename and path of  csv files containing number of household per sector and number of postcode units.
        postcode_file : csv file
            filename and path of csv files containing postcode sectors.

        Returns
        -------

        pandas.DataFrame
            Combined Dataframe with both mean of river level from Wet Day  and Typical day for each station.
        """

        household_file['average_household_per_postcode'] = np.round(household_file['households'] / household_file['number of postcode units'], 0)

        sector_array = np.unique(postcode_file['sector'].apply(lambda x: x.replace(' ', '')))
        sector_names = '|'.join(sector_array)

        sector_index = household_file[(household_file['postcode sector'].apply(lambda x: x.replace(' ', '')).str.contains(sector_names))]
        sector_index = sector_index[['postcode sector', 'average_household_per_postcode']].reset_index(drop=True)

        sector_index = sector_index.append({'postcode sector': str(None), 'average_household_per_postcode': np.nan},ignore_index=True)

        sector_dict = dict(zip(sector_index['postcode sector'].apply(lambda x: x.replace(' ', '')), sector_index.index))

        postcode_houses = postcode_file.copy()
        postcode_houses['sector_index'] = postcode_file['sector'].apply(lambda x: x.replace(' ', '')).apply(lambda x: sector_dict.get(x))
        postcode_houses['sector_index'] = postcode_houses['sector_index'].fillna(len(sector_dict) - 1).apply(lambda x: int(x))
        postcode_houses['average_households'] = sector_index.loc[postcode_houses['sector_index']].reset_index(drop=True)['average_household_per_postcode']
        df1_house = postcode_houses[['postcode','sector', 'average_households']]

        return df1_house

    def get_total_value(self, postal_data):
        """
        Return a series of estimates of the total property values
        for a sequence of postcode units or postcode sectors.

        Parameters
        ----------

        postal_data : sequence of strs
            Sequence of postcode units or postcode sectors

        Returns
        -------

        pandas.Series
            Series of total property value estimates indexed by locations.
        """

        houses_per_postcode = self.calc_households_per_postcode(self.household_df, self.training_df).dropna()
        houses_per_postcode['postcode'] = houses_per_postcode['postcode'].apply(lambda x: x.replace(' ', ''))
        houses_per_postcode['sector'] = houses_per_postcode['sector'].apply(lambda x: x.replace(' ', ''))
        average_num_houses = np.mean(houses_per_postcode['average_households'])

        median_price_preds = self.get_median_house_price_estimate(postal_data)
        median_price_preds.to_csv(os.sep.join((os.path.dirname(__file__), 'predictions', 'median_price_predictions.csv')), index=True, header=False)

        house_nums = []
        for postcode in postal_data:
            try:
                house_nums.append((houses_per_postcode[houses_per_postcode['postcode'] == postcode.replace(' ', '')]['average_households']).values[0])
            except:
                try:
                    house_nums.append(np.mean(houses_per_postcode[houses_per_postcode['sector'] == postcode.replace(' ', '')[:-2]]['average_households']).values)
                except:
                    house_nums.append(average_num_houses)

        return house_nums*median_price_preds

    def get_annual_flood_risk(self, supplied_postcodes=None,  risk_labels=None):
        """
        Return a series of estimates of the total property values of a
        collection of postcodes.

        Risk is defined here as a damage coefficient multiplied by the
        value under threat multiplied by the probability of an event.

        Parameters
        ----------

        supplied_postcodes : sequence of strs
            Sequence of postcodes.
        risk_labels: pandas.Series (optional)
            Series containing flood risk classifiers, as predicted by get_flood_class_from_postcodes.

        Returns
        -------

        pandas.Series
            Series of total annual flood risk estimates indexed by locations.
        """

        if supplied_postcodes is None:
            postcodes = self.postcode_df.index.values
        else:
            postcodes = supplied_postcodes

        cost = self.get_total_value(postcodes)

        risk_labels = risk_labels or self.get_flood_class_from_postcodes(postcodes)

        risk_labels.to_csv(os.sep.join((os.path.dirname(__file__), 'predictions', 'risk_label_predictions.csv')),index=True, header=False)

        print('Generating flood risk predictions...')

        probs = pd.DataFrame({1: 0.0001, 2: 0.0005, 3: 0.001, 4: 0.005, 5: 0.01, 6: 0.015, 7: 0.02, 8: 0.03, 9: 0.04, 10: 0.05},index=list(range(1, 11))).iloc[1]

        annual_flood_risk = (0.05 * cost.values * probs[risk_labels.values]).reset_index(drop=True)
        annual_flood_risk.index = postcodes

        print('Done.')

        return annual_flood_risk
