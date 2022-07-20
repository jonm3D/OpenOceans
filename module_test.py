from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure
from bokeh.sampledata.sea_surface_temperature import sea_surface_temperature
from bokeh.server.server import Server
from bokeh.themes import Theme


import pandas as pd
import numpy as np
import geopandas as gpd
import h5py
import os
import re
import sys
import datetime
import warnings
import pyproj

from tqdm import tqdm 
from io import BytesIO
from pybase64 import b64decode
from pathlib import Path

from scipy.signal import peak_widths, find_peaks
from astropy.time import Time, TimeDatetime

from bokeh.plotting import figure, curdoc
from bokeh.models import  CheckboxGroup, Button, Circle, TextInput, FileInput, RadioButtonGroup, Select, MultiChoice
from bokeh.models import DateRangeSlider, DatePicker
from bokeh.models import Paragraph, Div, Panel, Tabs
from bokeh.models import Range1d, ColumnDataSource, MultiLine, HoverTool
from bokeh.layouts import column, row
from bokeh.tile_providers import get_provider
from bokeh.server.server import Server

from sliderule import icesat2
from bokeh.models import BoxEditTool
import logging

def photon_refraction(W, Z, ref_az, ref_el,
                       n1=1.00029, n2=1.34116):
    '''
    ICESat-2 refraction correction implented as outlined in Parrish, et al. 
    2019 for correcting photon depth data.

    Highly recommended to reference elevations to geoid datum to remove sea
    surface variations.

    https://www.mdpi.com/2072-4292/11/14/1634

    Code Author: 
    Jonathan Markel
    Graduate Research Assistant
    3D Geospatial Laboratory
    The University of Texas at Austin
    jonathanmarkel@gmail.com

    Parameters
    ----------
    W : float, or nx1 array of float
        Elevation of the water surface.

    Z : nx1 array of float
        Elevation of seabed photon data. Highly recommend use of geoid heights.

    ref_az : nx1 array of float
        Photon-rate reference photon azimuth data. Should be pulled from ATL03
        data parameter 'ref_azimuth'. Must be same size as seabed Z array.

    ref_el : nx1 array of float
        Photon-rate reference photon azimuth data. Should be pulled from ATL03
        data parameter 'ref_elev'. Must be same size as seabed Z array.

    n1 : float, optional
        Refractive index of air. The default is 1.00029.

    n2 : float, optional
        Refractive index of water. Recommended to use 1.34116 for saltwater 
        and 1.33469 for freshwater. The default is 1.34116.

    Returns
    -------
    dE : nx1 array of float
        Easting offset of seabed photons.

    dN : nx1 array of float
        Northing offset of seabed photons.

    dZ : nx1 array of float
        Vertical offset of seabed photons.

    '''

    # compute uncorrected depths
    D = W - Z
    H = 496  # mean orbital altitude of IS2, km
    Re = 6371  # mean radius of Earth, km

    # angle of incidence (wout Earth curvature)
    theta_1_ = (np.pi / 2) - ref_el

    # incidence correction for earths curvature
    theta_EC = np.arctan(H * np.tan(theta_1_) / Re)

    # angle of incidence
    theta_1 = theta_1_ + theta_EC

    # angle of refraction
    theta_2 = np.arcsin(n1 * np.sin(theta_1) / n2)  # eq 1

    phi = theta_1 - theta_2

    # uncorrected slant range to the uncorrected seabed photon location
    S = D / np.cos(theta_1)  # eq 3

    # corrected slant range
    R = S * n1 / n2  # eq 2

    P = np.sqrt(R**2 + S**2 - 2*R*S*np.cos(theta_1 - theta_2))  # eq 6

    gamma = (np.pi / 2) - theta_1  # eq 4

    alpha = np.arcsin(R * np.sin(phi) / P)  # eq 5

    beta = gamma - alpha  # eq 7

    # cross-track offset
    dY = P * np.cos(beta)  # eq 8

    # vertical offset
    dZ = P * np.sin(beta)  # eq 9

    kappa = ref_az

    # UTM offsets
    dE = dY * np.sin(kappa)  # eq 10
    dN = dY * np.cos(kappa)  # eq 11

    return dE, dN, dZ

def convert_seg_to_ph_res(segment_res_data,
                          ph_index_beg,
                          segment_ph_cnt,
                          total_ph_cnt):

    '''Upsamples segment-rate data to the photon-rate.'''

    # initialize photon resolution array
    ph_res_data = pd.Series(np.nan, index=list(range(total_ph_cnt)))

    # trim away segments without photon data
    segs_with_data = segment_ph_cnt != 0

    segment_ph_cnt = segment_ph_cnt[segs_with_data]
    ph_index_beg = ph_index_beg[segs_with_data]
    segment_res_data = segment_res_data[segs_with_data]

    # set value for first photon in segment
    ph_res_data.iloc[ph_index_beg - 1] = segment_res_data
    ph_res_data.fillna(method='ffill', inplace=True)

    return ph_res_data.values

def read_h5_as_df(h5_file_bytes_io, gtxx, conf_type=1, verbose=False):

    if ((conf_type <= 0) or (conf_type >= 4)):
        raise Exception('''Requested confidence type must be between 
            0 and 4 (land, ocean, sea ice, land ice and inland water).''')

    # checks on beam input

    if isinstance(gtxx, str):
        # user input a single string like 'gt1r'

        # check its the right format
        gtxx = gtxx.lower()

        if re.search('gt[1-3][lr]', gtxx) is None:
            raise Exception('Unrecognized beam format. Use GT#(L/R) - for example, \'GT1R\'.')

        # set up in iterable
        gtxx = [gtxx]


    elif isinstance(gtxx, list):
        # user input a list of strings

        # make all inputs lowercase and remove duplicates

        gtxx = [g.lower() for g in gtxx]

        gtxx = list(np.unique(gtxx))

        # check all elements are the right format
        if any(re.search('gt[1-3][lr]', g) is None for g in gtxx):

            raise Exception('Unrecognized beam format. Use GT#(L/R) - for example, \'GT1R\'.')

        pass

    else:
        raise Exception('''Requested beams must be either a string (e.g.\'gt1r\')
                        or list (e.g. [\'gt1r\', \'gt2l\']).''')

    df_all = []
    # if verbose:
    #     print(h5_file_path)
    #     print('File size: {:.2f} GB'.format(os.path.getsize(h5_file_path) / 1e9))

    with h5py.File(h5_file_bytes_io, 'r') as f:
        for beam in tqdm(gtxx):
            pho = dict()
            seg = dict()
            anc= {}

            if verbose: print(beam.upper())

            if verbose: print('Reading photon-resolution data...')
            ##### PHOTON RESOLUTION DATA #####

            lat_ph = np.array(f[beam + '/heights/lat_ph'])

            lon_ph = np.array(f[beam + '/heights/lon_ph'])

            pho['geometry'] = gpd.points_from_xy(lon_ph, lat_ph)

            pho['delta_time'] = np.array(f[beam + '/heights/delta_time'])

            pho['height'] = np.array(f[beam + '/heights/h_ph'])

            # sliderule documentation not clear on this - skipping
            #df['distance'] = np.array(f[gtxx + '/heights/dist_ph_along'])

            pho['quality_ph'] = np.array(f[beam + '/heights/quality_ph'])

            signal_conf = np.array(f[beam + '/heights/signal_conf_ph'])

            pho['atl03_cnf'] = signal_conf[:, conf_type] # user input signal type

            if verbose: print('Reading ancillary data...')
            ##### ANCILLARY / ORBIT DATA #####

            # variable resolution 

            anc['atlas_sdp_gps_epoch'] = np.array(
                f['/ancillary_data/atlas_sdp_gps_epoch'])

            anc['sc_orient'] = np.array(
                f['/orbit_info/sc_orient'])

            anc['sc_orient_time'] = np.array(
                f['/orbit_info/sc_orient'])

            anc['rgt'] = np.array(
                f['/orbit_info/rgt'])

            pho['rgt'] = np.int64(anc['rgt'][0] * np.ones(len(lat_ph)))

            anc['orbit_number'] = np.array(
                f['/orbit_info/orbit_number'])

            anc['cycle_number'] = np.array(
                f['/orbit_info/cycle_number'])

            pho['cycle'] = np.int64(anc['cycle_number'][0] * np.ones(len(lat_ph)))

            if len(anc['sc_orient']) == 1:
                # no spacecraft transitions during this granule
                pho['sc_orient'] = np.int64(anc['sc_orient'][0] * np.ones(len(lat_ph)))
            else:
                warnings.warn('Spacecraft in transition detected! sc_orient parameter set to -1.')
                pho['sc_orient'] = np.int64(-1 * np.ones(len(lat_ph)))

            # track / pair data from user input

            pho['track'] = np.int64(beam[2]) * np.ones(len(lat_ph), dtype=np.int64)

            if beam[3].lower() == 'r':
                pair_val = 1

            elif beam[3].lower() == 'l':
                pair_val = 0

            pho['pair'] = pair_val * np.ones(len(lat_ph), dtype=np.int64)

            if verbose: print('Reading segment resolution data and upsampling...')
            ##### SEGMENT RESOLUTION DATA #####
            seg['ref_azimuth'] = np.array(
                f[beam + '/geolocation/ref_azimuth'])

            seg['ref_elev'] = np.array(
                f[beam + '/geolocation/ref_elev'])

            seg['segment_id'] = np.array(
                f[beam + '/geolocation/segment_id'])

            seg['segment_dist_x'] = np.array(
                f[beam + '/geolocation/segment_dist_x'])

            seg['solar_azimuth'] = np.array(
                f[beam + '/geolocation/solar_azimuth'])

            seg['solar_elev'] = np.array(
                f[beam + '/geolocation/solar_elevation'])

            seg['ph_index_beg'] = np.array(
                f[beam + '/geolocation/ph_index_beg'])

            seg['segment_ph_cnt'] = np.array(
                f[beam + '/geolocation/segment_ph_cnt'])

            seg['geoid'] = np.array(
                f[beam + '/geophys_corr/geoid'])

            ##### UPSAMPLE SEGMENT RATE DATA #####
            pho['segment_id']    = np.int64(convert_seg_to_ph_res(segment_res_data=seg['segment_id'],
                                       ph_index_beg=seg['ph_index_beg'],
                                       segment_ph_cnt=seg['segment_ph_cnt'],
                                       total_ph_cnt=len(lat_ph)))

            pho['segment_dist']    = convert_seg_to_ph_res(segment_res_data=seg['segment_dist_x'],
                                       ph_index_beg=seg['ph_index_beg'],
                                       segment_ph_cnt=seg['segment_ph_cnt'],
                                       total_ph_cnt=len(lat_ph))

            pho['ref_azimuth']    = convert_seg_to_ph_res(segment_res_data=seg['ref_azimuth'],
                                       ph_index_beg=seg['ph_index_beg'],
                                       segment_ph_cnt=seg['segment_ph_cnt'],
                                       total_ph_cnt=len(lat_ph))
            pho['ref_elev']       = convert_seg_to_ph_res(segment_res_data=seg['ref_elev'],
                                       ph_index_beg=seg['ph_index_beg'],
                                       segment_ph_cnt=seg['segment_ph_cnt'],
                                       total_ph_cnt=len(lat_ph))

            pho['geoid_z']       = convert_seg_to_ph_res(segment_res_data=seg['geoid'],
                                       ph_index_beg=seg['ph_index_beg'],
                                       segment_ph_cnt=seg['segment_ph_cnt'],
                                       total_ph_cnt=len(lat_ph))

            if verbose: print('Converting times...')
            ##### TIME CONVERSION / INDEXING #####

            t_gps = Time(anc['atlas_sdp_gps_epoch'] + pho['delta_time'], format='gps')
            t_utc = Time(t_gps, format='iso', scale='utc')
            pho['time'] = t_utc.datetime

            # sorting into the same order as sliderule df's purely for asthetics
            if verbose: print('Concatenating data...')
            df = gpd.GeoDataFrame(pho)

            df.set_index('time', inplace=True)

            df = df.loc[:, ['rgt', 'cycle', 'track', 'segment_id', 'segment_dist',
                            'sc_orient', 'atl03_cnf', 'height', 'quality_ph', 'delta_time', 'pair', 'geometry',
                           'ref_azimuth', 'ref_elev', 'geoid_z'] ]

            df_all.append(df)

    return pd.concat(df_all)


def reduce_gdf(gdf, RGT=None, track=None, pair=None, cycle=None):
    ''''''
    D = gdf.copy()
    print('fibufsdbsdvnds')
    if RGT is not None:
        D = D[D.loc[:, 'rgt'] == RGT]
    if track is not None:
        D = D[D.loc[:, 'track'] == track]
    if pair is not None:
        D = D[D.loc[:, 'pair'] == pair]
    if cycle is not None:
        D = D[D.loc[:, 'cycle'] == cycle] 

    return D

def bkapp(doc):

    df = sea_surface_temperature.copy()
    source = ColumnDataSource(data=df)
    reduce_gdf(df)
    plot = figure(x_axis_type='datetime', y_range=(0, 25), y_axis_label='Temperature (Celsius)',
                  title="Sea Surface Temperature at 43.18, -70.43")
    plot.line('time', 'temperature', source=source)

    def callback(attr, old, new):
        if new == 0:
            data = df
        else:
            data = df.rolling(f"{new}D").mean()
        source.data = ColumnDataSource.from_df(data)

    slider = Slider(start=0, end=30, value=0, step=1, title="Smoothing by N Days")
    slider.on_change('value', callback)

    doc.add_root(column(slider, plot))

# Setting num_procs here means we can't touch the IOLoop before now, we must
# let Server handle that. If you need to explicitly handle IOLoops then you
# will need to use the lower level BaseServer class.
server = Server({'/': bkapp}, num_procs=4)
server.start()
server.show('/')

# if __name__ == '__main__':
#     print('Opening Bokeh application on http://localhost:5006/')

server.io_loop.add_callback(server.show, "/")
server.io_loop.start()
