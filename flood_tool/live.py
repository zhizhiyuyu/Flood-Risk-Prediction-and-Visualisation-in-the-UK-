"""Interactions with rainfall and river data."""

import pandas as pd

__all__ = ["get_station_data_from_csv"]

def get_station_data_from_csv(filename, station_reference = None):
    """Return readings for a specified recording station from .csv file.

    Parameters
    ----------

    filename: str
        filename to read
    station_reference
        station_reference to return.
    """

    station_data = pd.read_csv(filename)
    if station_reference is not None:
        station_data = station_data.loc[station_data.stationReference == station_reference]

    return station_data.values

def get_live_station_data(station_reference):
    """
    Return readings for a specified recording station from live API.

    Parameters
    ----------

    filename: str
        filename to read
    station_reference
        station_reference to return.

    >>> data = get_live_station_data('0184TH')
    """

    url = f'https://environment.data.gov.uk/flood-monitoring/id/stations/{station_reference}'
    csv = pd.read_json(url)

    try:
        return csv.loc['measures'].values[2]['latestReading']['value']
    except:
        return 0

def get_live_stations_data(filename):
    """
    Return readings for a specified list of recording stations from live API.

    Parameters
    ----------

    filename: str
        filename to read

    """

    def abnormal_val(data, column_name):
        """
        Find abnormal values with '|' included in the column

        Parameters
        ----------
        data: pandas.Dataframe
            pandas.Dataframe containing unitName and value column.
        column_name: str
            column name to find the abnormal value in.

        Returns
        -------
        pandas.DataFrame
            Original input dataframe with value removed and input
        """

        abn_index = data[(data[column_name].str.contains(r"\|") == 1)][column_name].index

        abn_value = data.loc[abn_index][column_name]

        data.loc[(data[column_name].str.contains(r"\|") == 1), [column_name]] = abn_value.astype(str).apply(
            lambda x: x[0:5]).astype(float)
        return data

    station_data = pd.read_csv(filename)
    station_data['value'] = station_data['stationReference'].apply(get_live_station_data)
    station_data = abnormal_val(station_data, 'value')

    return station_data