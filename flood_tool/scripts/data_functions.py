import numpy as np
import pandas as pd

def extract_riverdata(typical_day, wet_day):
    """
    Getting river_data from typical_day and wet_day

    Parameters
    ----------
    wet_day : str data
        filename and path of csv files containing reading data(value and its unitName), type of qualifier, stationReference.
    typical_day : str data
        filename and path of csv files containing reading data(value and its unitName), type of qualifier, stationReference.
    Returns
    -------
    pandas.DataFrame
        Combined Dataframe with both mean wet and typical day river levels for each station.
    """
    
    def convert_mm(data) :
        """

        Convert data with in mm units to m units

        Parameters
        ----------
        data: pandas.Dataframe
            pandas.Dataframe containing unitName and value column.

        Returns
        -------
        pandas.DataFrame
            Original input dataframe with mm data converted
        """
            
        # finding index of data with mm as unit
        mm_index=data[(data['unitName']=='mm')]['value'].index

        # extracting value to convert
        mm_value=data.loc[mm_index]['value']

        # converting value and changing unit name
        data.loc[data['unitName']=='mm',['value']]=mm_value.astype(float).apply(lambda x : x/1000)
        data.loc[data['unitName']=='mm',['unitName']]=mm_value.apply(lambda x :'mASD')

        return data
    
    def abnormal_val (data,column_name):
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
        
        abn_index=data[(data[column_name].str.contains(r"\|")==1)][column_name].index

        abn_value=data.loc[abn_index][column_name]

        data.loc[(data[column_name].str.contains(r"\|")==1),[column_name]]=abn_value.astype(str).apply(lambda x : x[0:5]).astype(float)
        return data

    wet_data=pd.read_csv(wet_day)
    typical_data=pd.read_csv(typical_day)

    river_wet_data=wet_data[(wet_data['qualifier']=='Stage')]
    river_typical_data=typical_data[(typical_data['qualifier']=='Stage')]
    
    river_wet_data=convert_mm(river_wet_data)
    river_wet_data=abnormal_val(river_wet_data,'value')
    
    river_wet_data=river_wet_data[['stationReference','value']]
    river_wet_data.rename(columns={'value':'score'},inplace=True)
    river_wet_data.reset_index(drop=True,inplace=True)

    river_typical_data.loc[river_typical_data['value']>10,['value']]=np.nan
    river_typical_data=convert_mm(river_typical_data)
    river_typical_data=river_typical_data.groupby(['stationReference'])[['value','stationReference']].mean()
    river_typical_data.rename(columns={'value':'score'},inplace=True)

    river_data=river_wet_data.join(river_typical_data,on='stationReference',how='inner',lsuffix='_wetday',rsuffix='_typicalday')
    river_data.reset_index(drop=True,inplace=True)
    
    return river_data.to_csv('resources/Stage.csv',index=False)


def get_tide(typical_day, wet_day):
    """
    Get tidal data by input typical_day.csv and wet_day.csv

    Parameters
    ----------
    typical_day : str data
        filename and path of csv files containing reading data(value and its unitName), type of qualifier, stationReference.
    wet_day : str data
        filename and path of csv files containing reading data(value and its unitName), type of qualifier, stationReference.

   Returns
    -------
   pandas.DataFrame
        Combined Dataframe with both maximum wet and typical day tidal levels for each station.
    """

    data_t = pd.read_csv(typical_day)
    data_w = pd.read_csv(wet_day)

    def remove_outliers(dataset):
        """Returns a dataset excluding outliers"""

        dataset.loc[dataset['value'] > 10, ['value']] = np.nan
        dataset.loc[dataset['value']< -10, ['value']] = np.nan
            
        return dataset

    data_t = remove_outliers(data_t) # drop the outliers

    def prepare_data(dataset):
        """Drops duplicates and missing values, and converts the datetime value to pandas datetime object """

        dataset = dataset.drop_duplicates(ignore_index=True)
        dataset = dataset.dropna()
        dataset['dateTime']=pd.to_datetime(dataset['dateTime'])

        return dataset

    data_t = prepare_data(data_t)
    data_w = prepare_data(data_w)

    tidal_t = data_t[(data_t['qualifier']=='Tidal Level')]
    tidal_w = data_w[(data_w['qualifier']=='Tidal Level')]

    tidal_t = tidal_t[(tidal_t['unitName']=='mAOD')]
    tidal_w = tidal_w[(tidal_w['unitName']=='mAOD')]

    max_t = tidal_t.groupby(['stationReference']).max()
    max_w = tidal_w.groupby(['stationReference']).max()
    max_w = max_w['value'].astype(np.float64)

    data = {'stationReference': max_w.index,
        'max_tide(typical)': np.nan,
        'max_tide(wet)': np.nan}

    df = pd.DataFrame(data, index=None)

    for i in range(len(df)):
        if df['stationReference'].values[i] in max_w.index:
            df['max_tide(typical)'].values[i] = max_w[max_w.index == df['stationReference'].values[i]]
            
        if df['stationReference'].values[i] in max_t.index:
            df['max_tide(wet)'].values[i] = max_t['value'][max_t.index == df['stationReference'].values[i]].values[0]    

    return df.to_csv('resources/tidal_level.csv',index=False)
    
def rainfall_process(typical, wet):
    """
    Get rainfall data by input typical_day.csv and wet_day.csv

    Parameters
    ----------
    typical_day : str data
        filename and path of csv files containing reading data (value and its unitName), type of qualifier, stationReference.
    wet_day : str data
        filename and path of csv files containing reading data (value and its unitName), type of qualifier, stationReference.

   Returns
    -------
   pandas.DataFrame
        Combined Dataframe with both wet and typical day rainfall levels for each station.
    """

    df1 = pd.read_csv(typical)
    df2 = pd.read_csv(wet)
    
    rainfall1 = df1[df1['parameter']=='rainfall']
    rainfall2 = df2[df2['parameter']=='rainfall']

    rainfall2.value = rainfall2.value.astype(float)

    rainfall1.drop(index = rainfall1[rainfall1.value.isnull() == True].index, inplace = True)
    rainfall2.drop(index = rainfall2[rainfall2.value.isnull() == True].index, inplace = True)
    
    mean_value1 = rainfall1.groupby(['stationReference']).mean()*4
    mean_value2 = rainfall2.groupby(['stationReference']).mean()*4
    
    mean_value1.reset_index(inplace = True)
    mean_value2.reset_index(inplace = True)

    rainfall_station = mean_value2.merge(mean_value1, how='outer', on= 'stationReference') 
    rainfall_station.columns = ['stationReference','rainfall_wet','rainfall_typical']
    rainfall_station = rainfall_station[['stationReference','rainfall_typical','rainfall_wet']]
    return rainfall_station.to_csv('resources/rain.csv',index=False)