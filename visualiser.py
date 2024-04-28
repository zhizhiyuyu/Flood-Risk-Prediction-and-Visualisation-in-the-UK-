import argparse
import sys

import numpy as np
import pandas as pd
from flood_tool import *
import folium
import webbrowser
import os
from folium.plugins import HeatMap
import branca.colormap as cm


def get_typical_day_levels(data):
    """
    Get a map with two different layers illustrating
    typical rainfall and river levels

    Parameters
    -------------------

    data: pandas.DataFrame
        Geographic location data with rainfall and river levels

    Returns
    -------------------

    folium.Map
        Map containing two different layers illustrating
        typical rainfall and river levels
    """

    this_map = folium.Map(prefer_canvas=True)

    pointgroup_1 = folium.FeatureGroup(name='Typical_Rain', control=True)
    pointgroup_2 = folium.FeatureGroup(name='Typical_River', control=True)

    def plot_point(point):
        location = [point.latitude, point.longitude]

        folium.Circle(location=location,
                        popup=f'rainfall level:{point.rainfall_level}',
                        color=rainfall_c(point.rainfall_level),
                        radius=2,
                        weight=5).add_to(pointgroup_1)

        folium.Circle(location=location,
                      popup=f'river level:{point.river_level}',
                      color=river_c(point.river_level),
                      radius=2,
                      weight=5).add_to(pointgroup_2)

    east = np.array(data['easting'])
    north = np.array(data['northing'])
    lat, lon = geo.get_gps_lat_long_from_easting_northing(east, north)
    data['latitude'] = lat
    data['longitude'] = lon

    rainfall_c = cm.LinearColormap(["lightblue", "blue", "darkblue"], vmin=data['rainfall_level'].min(), vmax=data['rainfall_level'].max())
    river_c = cm.StepColormap(["lightblue", 'cadetblue', "blue", "darkblue"], vmin=data['river_level'].min(), vmax=data['river_level'].max()+1)

    data.apply(plot_point, axis=1)

    this_map.add_child(pointgroup_1)
    this_map.add_child(pointgroup_2)

    folium.LayerControl().add_to(this_map)
    this_map.fit_bounds(this_map.get_bounds())

    return this_map


def rain_heatmap(data):
    """
    Get a heatmap of typical rainfall level

    Parameters
    -------------------

    data: pandas.DataFrame
        Geographic location data with river levels

    Returns
    -------------------

    folium.Map
        Heatmap containing typical rainfall level
    """

    easting = data['easting'].values
    northing = data['northing'].values
    lat, lon = geo.get_gps_lat_long_from_easting_northing(easting, northing)

    rainfall_level = np.array(data['rainfall_level'])
    rainfall_level[np.isnan(rainfall_level)] = 0

    data = []
    for i in range(len(lat)):
        data.append([lat[i], lon[i], rainfall_level[i]])

    this_map = folium.Map(prefer_canvas=True)

    pointgroup_1 = folium.FeatureGroup(name='rainfall level', control=True)
    HeatMap(data).add_to(pointgroup_1)
    this_map.add_child(pointgroup_1)

    folium.LayerControl().add_to(this_map)

    this_map.fit_bounds(this_map.get_bounds())

    return this_map


def river_heatmap(data):
    """
    Get a heatmap of typical river level

    Parameters
    -------------------

    data: pandas.DataFrame
        Geographic location data with river levels

    Returns
    -------------------

    folium.Map
        Heatmap containing typical river level
    """

    easting = data['easting'].values
    northing = data['northing'].values
    lat, lon = geo.get_gps_lat_long_from_easting_northing(easting, northing)

    river_level = np.array(data['river_level'])
    river_level[np.isnan(river_level)] = 0

    data = []
    for i in range(len(lat)):
        data.append([lat[i], lon[i], river_level[i]])

    this_map = folium.Map(prefer_canvas=True)

    pointgroup_1 = folium.FeatureGroup(name='river level', control=True)
    HeatMap(data).add_to(pointgroup_1)
    this_map.add_child(pointgroup_1)

    folium.LayerControl().add_to(this_map)

    this_map.fit_bounds(this_map.get_bounds())

    return this_map

def tidal_heatmap(data):
    """
    Get a heatmap of typical tidal level

    Parameters
    -------------------

    data: pandas.DataFrame
        Geographic location data with river levels

    Returns
    -------------------

    folium.Map
        Heatmap containing typical tidal level
    """

    easting = data['easting'].values
    northing = data['northing'].values
    lat, lon = geo.get_gps_lat_long_from_easting_northing(easting, northing)

    tidal_level = np.array(data['tide_level'])
    tidal_level[np.isnan(tidal_level)] = 0

    data = []
    for i in range(len(lat)):
        data.append([lat[i], lon[i], tidal_level[i]])

    this_map = folium.Map(prefer_canvas=True)

    pointgroup_1 = folium.FeatureGroup(name='rainfall level', control=True)
    HeatMap(data).add_to(pointgroup_1)
    this_map.add_child(pointgroup_1)

    folium.LayerControl().add_to(this_map)

    this_map.fit_bounds(this_map.get_bounds())

    return this_map

def get_predicted(data):
    """
    Get a map with three different layers illustrating
    predicted flood risk, risk label, and median property price

    Parameters
    -------------------

    data: pandas.DataFrame
        Geographic location data with predicted flood risk, risk label, and median property price

    Returns
    -------------------

    folium.Map
        Map containing three different layers illustrating
        predicted flood risk, risk label, and median property price
    """

    pointgroup_1 = folium.FeatureGroup(name='Overall Risk', control=True)
    pointgroup_2 = folium.FeatureGroup(name='Median Property Price', control=True)
    pointgroup_3 = folium.FeatureGroup(name='Flood Risk Label', control=True)

    def plot_point(point):
        location = [point.latitude, point.longitude]
        risk = point.risk
        price = point.price
        label = point.label

        if risk >=0 and risk <= 200:
            color = 'green'
        elif risk > 200 and risk <= 500:
            color = 'orange'
        elif risk > 500:
            color = 'red'

        folium.Circle(location=location,
                      popup=f'overall risk:{risk}',
                      color=color,
                      radius=2,
                      weight=5).add_to(pointgroup_1)

        if price >=0 and price <= 200000:
            color = 'lightgreen'
        elif price > 200000 and price <= 500000:
            color = 'green'
        elif price > 500000 and price <= 1000000:
            color = 'orange'
        elif price > 10000000:
            color = 'red'

        folium.Circle(location=location,
                      popup=f'median property price:{price}',
                      color=color,
                      radius=2,
                      weight=5).add_to(pointgroup_2)

        folium.Circle(location=location,
                      popup=f'risk label:{label}',
                      color=label_c(label),
                      radius=2,
                      weight=5).add_to(pointgroup_3)

    this_map = folium.Map(prefer_canvas=True)

    label_c = cm.LinearColormap(["green", "orange", "red"], vmin=data.label.min(), vmax=data.label.max())

    lat, long = geo.get_gps_lat_long_from_easting_northing(data.easting, data.northing)
    data['latitude'] = lat
    data['longitude'] = long

    data.apply(plot_point, axis=1)

    this_map.add_child(pointgroup_1)
    this_map.add_child(pointgroup_2)
    this_map.add_child(pointgroup_3)

    folium.LayerControl().add_to(this_map)

    this_map.fit_bounds(this_map.get_bounds())

    return this_map

def plot_visualisation():
    """Parses command line input and visualises data."""

    sys.stdout = open(os.devnull, 'w')

    parser = argparse.ArgumentParser()
    parser.add_argument("-hr", "--heatmap_rain",help="Filename of a .csv containing rain level data for geographic locations.", type=str)
    parser.add_argument("-hrr", "--heatmap_river",help="Filename of a .csv containing river level data for geographic locations.", type=str)
    parser.add_argument("-ht", "--heatmap_tidal", help="Filename of a .csv containing tidal level data for geographic locations.", type=str)
    parser.add_argument("-cmrr", "--colormap_rr", help="Filename of a .csv containing rain, river, and tide level data for geographic locations.", type=str)
    parser.add_argument("-cmr", "--colormap_r",help="Filename of a .csv containing predicted flood risk, risk label, and median property value for geographic locations.", type=str)
    args = parser.parse_args()

    if args.heatmap_rain is not None and os.path.exists(args.heatmap_rain):
        heatmap_rain = args.heatmap_rain
        heatmap_format = pd.read_csv(heatmap_rain)

        req_columns = list(item in heatmap_format.columns.to_list() for item in ['easting','northing','rainfall_level'])

        if len(set(req_columns)) == 1:
            mapit = rain_heatmap(heatmap_format)
        else:
            sys.exit('Invalid rainfall level data')

        filename = os.getcwd() + '/visualisations/' + 'heatmap_rain.html'
        mapit.save(filename)
        webbrowser.open_new_tab('file:///' + filename)

    if args.heatmap_river is not None and os.path.exists(args.heatmap_river):
        heatmap_river = args.heatmap_river
        heatmap_format = pd.read_csv(heatmap_river)

        req_columns = list(item in heatmap_format.columns.to_list() for item in ['easting','northing','river_level'])

        if len(set(req_columns)) == 1:
            mapit = river_heatmap(heatmap_format)
        else:
            sys.exit('Invalid river level data')

        filename = os.getcwd() + '/visualisations/' + 'heatmap_river.html'
        mapit.save(filename)
        webbrowser.open_new_tab('file:///' + filename)

    if args.heatmap_tidal is not None and os.path.exists(args.heatmap_tidal):
        heatmap_tidal = args.heatmap_tidal
        heatmap_format = pd.read_csv(heatmap_tidal)

        req_columns = list(item in heatmap_format.columns.to_list() for item in ['easting','northing','tide_level'])

        if len(set(req_columns)) == 1:
            mapit = tidal_heatmap(heatmap_format)
        else:
            sys.exit('Invalid tide level data')

        filename = os.getcwd() + '/visualisations/' + 'heatmap_tidal.html'
        mapit.save(filename)
        webbrowser.open_new_tab('file:///' + filename)

    if args.colormap_rr is not None and os.path.exists(args.colormap_rr):
        colormap_rr = args.colormap_rr
        colormap_format = pd.read_csv(colormap_rr)

        req_columns = list(item in colormap_format.columns.to_list() for item in ['easting', 'northing', 'rainfall_level', 'river_level','tide_level'])

        if len(set(req_columns)) == 1:
            mapit = get_typical_day_levels(colormap_format)
        else:
            sys.exit('Invalid rainfall and river level data')

        filename = os.getcwd() + '/visualisations/' + 'typical_day_levels.html'
        mapit.save(filename)
        webbrowser.open_new_tab('file:///' + filename)

    if args.colormap_r is not None and os.path.exists(args.colormap_r):
        colormap_r = args.colormap_r
        colormap_format = pd.read_csv(colormap_r)

        req_columns = list(item in colormap_format.columns.to_list() for item in ['easting', 'northing', 'risk', 'price','label'])

        if len(set(req_columns)) == 1:
            mapit = get_predicted(colormap_format)
        else:
            sys.exit('Invalid rainfall level data')

        filename = os.getcwd() + '/visualisations/' + 'risk_predictions.html'
        mapit.save(filename)
        webbrowser.open_new_tab('file:///' + filename)

if __name__=='__main__':
    plot_visualisation()