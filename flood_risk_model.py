#!/usr/bin/env python3

""" A script to run the flood risk prediction model"""

import argparse
import sys
import os
import pandas as pd
from flood_tool import *

def run_model():
    """Parses command line input and runs flood risk prediction model."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--postcode_file",help="Filename of a .csv file containing geographic location data for postcodes.", type=str)
    parser.add_argument("-l","--sample_labels", help="Filename of a .csv containing labelled geographic location data for postcodes.", type=str)
    parser.add_argument("-hh","--household_file", help="Filename of a .csv file containing information on households by postcode.", type=str)
    args = parser.parse_args()

    flag = 0
    postcode_file = ''
    if args.postcode_file is None:
        flag = 1
    elif args.postcode_file is not None and os.path.exists(args.postcode_file):
        postcode_file = args.postcode_file
        postcode_format = pd.read_csv(postcode_file)

        req_columns1 = list(item in postcode_format.columns.to_list() for item in ['postcode','easting','northing'])
        req_columns2 = list(item in postcode_format.columns.to_list() for item in ['postcode', 'latitude', 'longitude'])

        if len(set(req_columns1)) == 1:
            flag = 2
        elif len(set(req_columns2)) == 1:
            flag = 3
        else:
            sys.exit('Invalid postcode_file')

    sample_labels = ''
    if args.sample_labels is not None and os.path.exists(args.sample_labels):
        sample_labels = args.sample_labels
        labels_format = pd.read_csv(sample_labels)

        req_columns = list(item in labels_format.columns.to_list() for item in ['postcode','easting','northing','riskLabel','medianPrice','localAuthority'])

        if not len(set(req_columns)) == 1:
            sys.exit('Invalid sample_labels')

    household_file = ''
    if args.household_file is not None and os.path.exists(args.household_file):
        household_file = args.household_file
        household_format = pd.read_csv(household_file)

        req_columns = list(item in household_format.columns.to_list() for item in ['postcode sector', 'households', 'number of postcode units'])

        if not len(set(req_columns)) == 1:
            sys.exit('Invalid household_file')

    model = Tool(postcode_file, sample_labels, household_file)

    postcode_filename = postcode_file.split('/')[-1]

    if flag == 1:
        flood_risk_predictions = model.get_annual_flood_risk()
        flood_risk_predictions.to_csv(os.sep.join((os.path.dirname(__file__), 'flood_tool/resources', 'flood_risk_predictions.csv')),header=False)
    elif flag == 2:
        flood_risk_predictions = model.get_annual_flood_risk(postcode_format['postcode'].values)
        flood_risk_predictions.to_csv(os.sep.join((os.path.dirname(__file__), 'flood_tool/resources', f'flood_risk_predictions_{postcode_filename}.csv')),header=False)
    elif flag == 3:
        risk_labels = model.get_flood_class_from_WGS84_locations(postcode_format['latitude'].values,postcode_format['longitude'].values)
        flood_risk_predictions = model.get_annual_flood_risk(risk_labels = risk_labels)
        flood_risk_predictions.to_csv(os.sep.join((os.path.dirname(__file__), 'flood_tool/resources', f'flood_risk_predictions_{postcode_filename}.csv')),header=False)
    else:
        sys.exit('Model failed - incompatible input data')

if __name__ == '__main__':
    run_model()