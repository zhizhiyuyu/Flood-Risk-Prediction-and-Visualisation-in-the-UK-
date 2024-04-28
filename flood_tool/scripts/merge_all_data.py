# import geopandas as gpd
# import pandas as pd
# import glob, os
# import geo
#
# def Data_Merge(path = "ads-deluge-Clyde/flood_tool/resources"):
#     """ Merge all csvs in a folder into one GeoDataFrame
#         Expects CSVs are named as follows:
#         postcodes_unlabelled.csv
#         stage.csv
#         tidal_level.csv
#         rain.csv
#         wet_day.csv
#         households_per_sector.csv
#         typical_day.csv
#         postcodes_sampled.csv
#
#     Parameters
#     ----------
#     path: string
#         File path to the folder which contains the data
#     Returns
#     -------
#     GeoPandas Dataframe
#     """
#     os.chdir(path)
#
#     #Create a dictionary of geodataframes from CSVs in the path
#     df_dict = {os.path.splitext(os.path.basename(f))[0] : pd.read_csv(f) for f in glob.glob('*.csv')}
#
#     def proc_Postcodes(postcode_Data):
#         postcode_Data['latitude'], postcode_Data['longitude'] = geo.get_gps_lat_long_from_easting_northing(postcode_Data['easting'], postcode_Data['northing'])
#         postcode_Data = gpd.GeoDataFrame(postcode_Data, geometry = gpd.points_from_xy(postcode_Data['longitude'], postcode_Data['latitude']), crs={'init': 'epsg:4326'})
#         postcode_Data.dropna(subset=['longitude', 'latitude'], how='any', inplace=True)
#         postcode_Data.reset_index(drop=True)
#         return postcode_Data
#
#     df_dict['stations'] = df_dict['stations'][['stationReference', 'latitude', 'longitude']].reset_index(drop=True)
#     def proc_Stations(station_Data):
#         station_Data = station_Data.merge(df_dict['stations'], on='stationReference')
#         station_Data = gpd.GeoDataFrame(station_Data, geometry = gpd.points_from_xy(station_Data['longitude'], station_Data['latitude']), crs={'init': 'epsg:4326'})
#         station_Data.dropna(subset=['longitude', 'latitude'], how='any', inplace=True)
#         station_Data.reset_index(drop=True)
#         return station_Data
#
#     def join_Nearest(DataFrame, stations_DF):
#         out_DF = gpd.sjoin_nearest(left_df=DataFrame, right_df=stations_DF, how='left')
#         out_DF.reset_index(drop=True)
#         return out_DF
#
#     pc_Unlabelled = proc_Postcodes(df_dict['postcodes_unlabelled'])
#     pc_Labelled = proc_Postcodes(df_dict['postcodes_sampled'])
#     stage = proc_Stations(df_dict['stage'])
#     tidal_level = proc_Stations(df_dict['tidal_level'])
#     rain = proc_Stations(df_dict['rain'])
#
#     #Begin Combining the data
#     #Combine unlabelled/labelled data (These are split again at the end, this is just to make the processing neater)
#     combined_Data = gpd.GeoDataFrame(pd.concat([pc_Labelled, pc_Unlabelled], ignore_index=True) )
#
#     #Join Households per sector data
#     #df_dict['households_per_sector'].rename(columns = {'postcode sector': 'sector'}, inplace=True) #Rename to facilitate the join
#     #combined_Data = combined_Data.merge(df_dict['households_per_sector'], on='sector', how='left', validate='m:1')
#     combined_Data = combined_Data.merge(df_dict['postcode_average_household'], on=['postcode', 'sector'], how='left', validate='m:1')
#
#     #Join nearest station
#     combined_Data = join_Nearest(combined_Data, stage)
#     combined_Data.drop(columns=['index_right'], inplace=True)
#     combined_Data = join_Nearest(combined_Data, tidal_level)
#     combined_Data.drop(columns=['index_right'], inplace=True)
#     combined_Data = join_Nearest(combined_Data, rain)
#
#     combined_Data = combined_Data[combined_Data.columns.drop(list(combined_Data.filter(regex='index')))]
#     combined_Data = combined_Data[combined_Data.columns.drop(list(combined_Data.filter(regex='latitude')))]
#     combined_Data = combined_Data[combined_Data.columns.drop(list(combined_Data.filter(regex='longitude')))]
#     combined_Data = combined_Data[combined_Data.columns.drop(list(combined_Data.filter(regex='Unnamed')))]
#     combined_Data = combined_Data[combined_Data.columns.drop(list(combined_Data.filter(regex='stationReference')))]
#     combined_Data.rename(columns=({'sector_x':'Postcode Sector'}))
#     combined_Data = combined_Data.reset_index(drop=True)
#
#     #Split the labelled/unlabelled data back out
#     unLabelled_GDF = combined_Data[combined_Data['riskLabel'].isna()].reset_index(drop=True)
#     Labelled_GDF = combined_Data[~combined_Data['riskLabel'].isna()].reset_index(drop=True)
#
#     return unLabelled_GDF, Labelled_GDF
#
