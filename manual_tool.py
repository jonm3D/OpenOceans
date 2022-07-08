# must pass input --websocket-max-message-size=500000000
# otherwise typical h5 file will not be able to be passed and the server will close

""
import pandas as pd
import numpy as np
import geopandas as gpd
import h5py
import os
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pyproj
from datetime import datetime

from scipy.stats import norm
from scipy.signal import peak_widths, find_peaks
from astropy.time import Time, TimeDatetime
import laspy

import ipywidgets as widgets
import warnings
from tqdm import tqdm 
from ipyfilechooser import FileChooser

from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Paragraph, CheckboxGroup, Button, Circle, TextInput, FileInput, RadioButtonGroup, LassoSelectTool
from bokeh.io import show, output_notebook
from bokeh.layouts import column, row
from bokeh.models import Range1d
from bokeh.palettes import Category10


import io
from pybase64 import b64decode
from bokeh.tile_providers import get_provider

# from tkinter import filedialog
# from tkinter import *


tile_provider = get_provider('ESRI_IMAGERY')

color_palette = Category10[4]

unclass_color = color_palette[0]
surface_color = color_palette[1]

surface_class = 41
no_bottom_class = 45
bathy_class = 40

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
    
    if RGT is not None:
        D = D[D.loc[:, 'rgt'] == RGT]
    if track is not None:
        D = D[D.loc[:, 'track'] == track]
    if pair is not None:
        D = D[D.loc[:, 'pair'] == pair]
    if cycle is not None:
        D = D[D.loc[:, 'cycle'] == cycle] 
        
    return D


""
panel_width = 350

df_cols = ['lon', 'lat', 'rgt', 'cycle', 'track', 'segment_id', 'segment_dist', 'sc_orient',
       'atl03_cnf', 'height', 'quality_ph', 'delta_time', 'pair',
       'ref_azimuth', 'ref_elev', 'geoid_z', 'x_wm', 'y_wm'
           'height_ortho', 'classification', 'surface_height', 'surface_sigma', 'dZ', 'height_ortho_c']

gdf_empty = pd.DataFrame([], columns=df_cols)

# source to hold all h5 data
src = ColumnDataSource(data=gdf_empty)

# source to hold data for individual beams
src_gt = ColumnDataSource(data=gdf_empty)

# source to manage plotting of all photon data
plt_src = ColumnDataSource(data=gdf_empty)

# source to hold data for individual beams - subsampled for plotting
src_gt_sub = ColumnDataSource(data=gdf_empty)

# sources for individual track classified data
sfc_src = ColumnDataSource(data=gdf_empty)
subsfc_src = ColumnDataSource(data=gdf_empty)
bathy_src = ColumnDataSource(data=gdf_empty)



fig_full_profile = figure(title='Full Track Profile', 
                         tools='zoom_in,zoom_out,pan,box_zoom,wheel_zoom,undo,redo,reset,lasso_select',
                          sizing_mode="scale_width", height=500,
                          x_axis_type='mercator')#, x_range=fig_imagery.y_range)
                        
# glyphs

wm_axis = 'x_wm'
fig_full_profile.circle(x=wm_axis, y='height_ortho', 
                        source=plt_src, size=0.5,
                        color='black')

fig_full_profile.circle(x=wm_axis, 
                        y='height_ortho', 
                        source=sfc_src, 
                        size=0.5,
                        color='blue', 
                        legend_label='Water Surface (Class {})'.format(surface_class))

fig_full_profile.line(x=wm_axis, y='surface_height', source=plt_src, 
                        color='lime', line_width=3,
                        name='surface_model')

fig_full_profile.circle(x=wm_axis, 
                        y='height_ortho',
                        source=subsfc_src,
                        size=0.5,
                        color='grey', 
                        legend_label='Subsurface Noise (Class {})'.format(no_bottom_class), 
                        name='subsurface',
                        selection_color='red',
                        selection_fill_alpha=1,
                        selection_line_color='red',
                        nonselection_fill_alpha=0.9,
                        nonselection_line_color='black',
                        nonselection_fill_color="black")

fig_full_profile.circle(x=wm_axis, 
                        y='height_ortho',
                        source=bathy_src,
                        size=0.5,
                        color='black', 
                        legend_label='Uncorrected Bathymetry (Class {})'.format(bathy_class))

fig_full_profile.circle(x=wm_axis, 
                        y='height_ortho_c',
                        source = bathy_src,
                        size=0.5,
                        color='red', 
                        legend_label='Bathymetry, Refraction Corrected'.format(bathy_class))

fig_full_profile.legend.location = "bottom_left"
fig_full_profile.legend.visible = False

# figures
fig_imagery = figure(title='', 
                     x_range=(-6000000, 6000000), 
                     y_range=(-1000000, 7000000),
                     x_axis_type="mercator", 
                     y_axis_type="mercator", 
                     tools='zoom_in,zoom_out,pan,box_zoom,wheel_zoom,undo,redo,reset,', sizing_mode="stretch_height", 
                     width=panel_width, toolbar_location='below')

fig_imagery.add_tile(tile_provider)
trackline = fig_imagery.line(x='y_wm', y='x_wm', source=src_gt, color='red', line_width=1)
gt_labels = ['gt1r', 'gt1l', 'gt2r', 'gt2l', 'gt3r', 'gt3l']

def import_h5(attr, old, new):
    decoded = b64decode(file_in_w.value)
    f = io.BytesIO(decoded)
    # h = h5py.File(f,'r')
    gdf1 = read_h5_as_df(f, 
                       gt_labels, 
                       conf_type=1, verbose=False)
    
    # coords/conversions
    gdf1.insert(0, 'lat', gdf1.geometry.y, False) 
    gdf1.insert(0, 'lon', gdf1.geometry.x, False)
    
    wgs84 = pyproj.crs.CRS.from_epsg(4979)
    wgs84_egm08 = pyproj.crs.CRS.from_epsg(3855)
    tform = pyproj.transformer.Transformer.from_crs(crs_from=wgs84, crs_to=wgs84_egm08)
    _, _, z_g = tform.transform(gdf1.geometry.y, gdf1.geometry.x, gdf1.height)
    gdf1.insert(0, 'height_ortho', z_g, False) 
        
    # allocation of to be used arrays
    gdf1.insert(0, 'classification', np.int64(np.zeros_like(z_g)), False)
    gdf1.insert(0, 'surface_height', np.nan*np.zeros_like(z_g), False)
    gdf1.insert(0, 'surface_sigma', np.nan*np.zeros_like(z_g), False)
    gdf1.insert(0, 'height_ortho_c', np.nan*np.zeros_like(z_g), False)
    gdf1.insert(0, 'dZ', np.zeros_like(z_g), False)
    # color_arr = np.chararray(np.shape(z_g), itemsize=7)
    # color_arr[:] = unclass_color
    # gdf1.insert(0, 'color', color_arr, False)
    
    src.data = gdf1.drop('geometry', axis=1)
    
    # enable the gtxx selector boxes
    gt_w.disabled=False
    

def select_profile(attr, old, new):
    # start fresh with classes in case the user is coming from another tab
    src_gt.data = gdf_empty
    plt_src.data = gdf_empty
    src_gt_sub.data = gdf_empty
    sfc_src.data = gdf_empty
    subsfc_src.data = gdf_empty
    bathy_src.data = gdf_empty
    
    src_gt.selected.indices = []
    plt_src.selected.indices = []
    src_gt_sub.selected.indices = []
    sfc_src.selected.indices = []
    subsfc_src.selected.indices = []
    bathy_src.selected.indices = []
    
    save_out_w.disabled=False
    surface_line = fig_full_profile.select_one({'name': 'surface_model'})    
    surface_line.visible = True
    
    ######## 
    
    pair_char = gt_labels[new][3]
    track_ = np.int64(gt_labels[new][2])
    if pair_char == 'l': pair_ = 0
    elif pair_char == 'r': pair_ = 1
    
    # print(pair_, track_)
    
    gdf_ = reduce_gdf(pd.DataFrame(src.data), track=track_, pair=pair_)
    src_gt.data = gdf_ # profile data
    # profile data, subsampled for plotting
    src_gt_sub.data = gdf_.sample(min(int(1e5), gdf_.shape[0])) 
    

    # convert sampled gdf lat lon to web mercator for plotting
    wgs84 = pyproj.crs.CRS.from_epsg(4979)
    web_merc = pyproj.crs.CRS.from_epsg(3857)
    tform = pyproj.transformer.Transformer.from_crs(crs_from=wgs84, crs_to=web_merc)
    # src_gt_sub.data['y_wm'], src_gt_sub.data['x_wm'] = tform.transform(src_gt_sub.data['lat'], 
    #                                                                    src_gt_sub.data['lon'])
    src_gt.data['y_wm'], src_gt.data['x_wm'] = tform.transform(src_gt.data['lat'], src_gt.data['lon'])
        
    # update range
    #fig_full_profile.x_range = Range1d(min(src_gt.data['lat']), max(src_gt.data['lat']))
    
    #fig_full_profile.x_range = Range1d(min(src_gt.data['lat']), max(src_gt.data['lat']))  
    fig_full_profile.xaxis.axis_label = 'Latitude (deg)'
    fig_full_profile.yaxis.axis_label = 'Orthometric Height (m)'
        
    # enable water surface calculation button
    model_surface_w.disabled = False
    select_points_w.disabled = True
    calc_refract_w.disabled = True
    
    # update the base filename field
    out_name_w.value = re.sub('ATL03', 'BATHY', 
                              file_in_w.filename[:-3] + '_' + gt_labels[gt_w.active].upper())

    # update plot/remove legend
    plt_src.data = dict(src_gt.data)
    fig_full_profile.legend.visible = False

def model_surface():

    if src_gt.data['segment_id'] != []:
        # calculate water surface model
        # tweakable parameters
        z_bin_size = 0.1

        # vertical bins from 100m depth to 50m above 0 datum
        z_bin_edges = np.arange(-25, 75 + z_bin_size, z_bin_size)
        z_centered_bins = z_bin_edges[:-1] + z_bin_size

        # at each 20m segment evaluate the 400m on either side
        process_chunk_half_width = 20

        for seg_i in tqdm( np.arange(src_gt.data['segment_id'].min(), src_gt.data['segment_id'].max(), 5) ):

            # index for segment processing window
            chunk_i = (src_gt.data['segment_id'] >= (seg_i-process_chunk_half_width)) \
                & (src_gt.data['segment_id'] <= (seg_i+process_chunk_half_width))

            #gdf_seg = gdf1.loc[ chunk_i , : ]

            depth = -src_gt.data['height_ortho'][chunk_i]

            hist, _ = np.histogram(depth, bins=z_bin_edges)
            #print(hist)
            # find peaks
            pk_i, pk_dict = find_peaks(hist)

            pk_dict['fwhm'], pk_dict['width_heights_hm'], pk_dict['left_ips_hm'] ,pk_dict['right_ips_hm'] \
                = peak_widths(hist , pk_i , rel_height=0.4)

            pk_dict['i'] = pk_i

            pk_dict['heights'] = hist[pk_dict['i']]

            # estimate standard deviation
            pk_dict['sigma_est'] = z_bin_size * (pk_dict['fwhm'] / 2.35) # std. dev. est.

            # get z value of peak
            pk_dict['z'] = z_centered_bins[pk_dict['i']]

            pk_df = pd.DataFrame.from_dict(pk_dict, orient='columns')

            pk_df.sort_values(by='heights', inplace=True, ascending=False)

            if pk_df.shape[0] == 0:
                # no peaks in segment
                src_gt.data['surface_height'][chunk_i] = -99
                continue

            else:
                surf_pk = pk_df.iloc[0]

                src_gt.data['surface_height'][chunk_i] = -surf_pk.z

                src_gt.data['surface_sigma'][chunk_i] = surf_pk.sigma_est    
        
        lower_bound = (src_gt.data['surface_height'] - 3*src_gt.data['surface_sigma'])
        upper_bound = (src_gt.data['surface_height'] + 3*src_gt.data['surface_sigma'])
        
        surf_idx = (src_gt.data['height_ortho'] >= lower_bound ) \
                 & (src_gt.data['height_ortho'] <= upper_bound )
        subsurf_idx = (src_gt.data['height_ortho'] < lower_bound )
        
        # update surface and subsurface classifications
        src_gt.data['classification'][surf_idx] = surface_class
        src_gt.data['classification'][subsurf_idx] = no_bottom_class
        
        # update plotting data with new surface data for passing later
        plt_src.data = dict(src_gt.data)

        select_points_w.disabled = False
        model_surface_w.disabled = True
    
def select_bathy():
    # make the lasso tool active
    
    # remove the water surface model from the plot
    # or get the glyph from the Figure:
    surface_line = fig_full_profile.select_one({'name': 'surface_model'})    
    surface_line.visible = False
    
    # remove the legend from the plot
    fig_full_profile.legend.visible = False
                        
    # change the subsurface points to black just for this selection step
    subsurface_renderer = fig_full_profile.select_one({'name': 'subsurface'})
    subsurface_renderer.glyph.fill_color='black'
    subsurface_renderer.glyph.line_color='black'
    subsurface_renderer.selection_glyph.size=0.5
    subsurface_renderer.selection_glyph.fill_color='red'
    subsurface_renderer.selection_glyph.line_color='red'
    subsurface_renderer.nonselection_glyph.size=0.5
    subsurface_renderer.nonselection_glyph.fill_color='black'
    subsurface_renderer.nonselection_glyph.line_color='black'

    
    # get index of surface classifications
    surf_idx = plt_src.data['classification'] == surface_class
            
    # add all subsurface data to subsurface plot source and remove the intermediate plotting data
    rm_idx = ((plt_src.data['height_ortho'] >= plt_src.data['surface_height']) | surf_idx)
    subsfc_src.data = {key: value[~rm_idx] for (key, value) in plt_src.data.items()}
    plt_src.data = gdf_empty
    
    # enable selection tool
    calc_refract_w.disabled = False
    select_points_w.disabled = True
    
def correct_refraction():
    # switch subsurface photons back to gray for background
    subsurface_renderer = fig_full_profile.select_one({'name': 'subsurface'})
    subsurface_renderer.glyph.fill_color='grey'
    subsurface_renderer.glyph.line_color='grey'
    subsurface_renderer.selection_glyph.size=0.5
    subsurface_renderer.selection_glyph.fill_color='grey'
    subsurface_renderer.selection_glyph.line_color='grey'
    subsurface_renderer.nonselection_glyph.size=0.5
    subsurface_renderer.nonselection_glyph.fill_color='grey'
    subsurface_renderer.nonselection_glyph.line_color='grey'
    # disable lasso?
    
    # disable point selection button
    select_points_w.disabled = True
    
    # for all selected points
    bathy_idx = subsfc_src.selected.indices
    subsfc_src.selected.indices = []
    bathy_bool = np.zeros((len(subsfc_src.data['height']),), dtype=bool)
    bathy_bool[bathy_idx] = True
    
    # update classifications before passing data along to bathy_src holder
    subsfc_src.data['classification'][bathy_bool] = bathy_class
    
    bathy_src.data = {key: value[bathy_bool] for (key, value) in subsfc_src.data.items()}
    
    # remove bathy from subsurface noise data handler
    subsfc_src.data = {key: value[~bathy_bool] for (key, value) in subsfc_src.data.items()}
    
    # calculate refraction
    _, _, bathy_src.data['dZ'] = photon_refraction(W=bathy_src.data['surface_height'],
                                                   Z=bathy_src.data['height_ortho'], 
                                                   ref_az=bathy_src.data['ref_azimuth'], 
                                                   ref_el=bathy_src.data['ref_elev'],
                                                   n1=1.00029, n2=1.34116)
    
    bathy_src.data['height_ortho_c'] = bathy_src.data['height_ortho'] + bathy_src.data['dZ']

    
    # reactivate surface photons on plot
    surf_idx = (src_gt.data['classification'] == surface_class)
    sfc_src.data = {key: value[surf_idx] for (key, value) in src_gt.data.items()}
    
    fig_full_profile.legend.visible = True
    
    # disable calculate refraction button
    calc_refract_w.disabled=True
    save_out_w.disabled=False
    

    
def check_out_dir_exists(attr, old, new):
    if not Path(out_path_w.value_input).exists():
        out_path_w.background = 'red'
    else:
        out_path_w.background = 'white'
        
def save_output():
    csv_output_path = os.path.join(out_path_w.value, 
                                   out_name_w.value + '.csv')
    
    #combine surface data, bathy data, subsurface data
    df_comb = pd.concat([pd.DataFrame(bathy_src.data),
                         pd.DataFrame(sfc_src.data), 
                         pd.DataFrame(subsfc_src.data)])
    
    if 0 in out_types_w.active:
        # csv checked
        df_out = df_comb.loc[:, 
                             ['lon', 'lat', 
                              'height_ortho', 'classification', 
                              'surface_height','surface_sigma', 'dZ'] ]
        df_out.to_csv(csv_output_path, index=False)
        
    save_out_w.button_type='success'
    save_out_w.label='Output Saved!'
    save_out_w.disabled=True
    
    print()
    
def close_app():
    sys.exit()
                            
file_in_w = FileInput(accept=".h5", sizing_mode="stretch_width")
gt_w = RadioButtonGroup(labels=gt_labels, 
                        disabled=True, sizing_mode="stretch_width")
    
gt_w.on_change('active', select_profile)
file_in_w.on_change('value', import_h5)

# Widget setup
model_surface_w = Button(label='Model Water Surface', disabled=True, sizing_mode="stretch_width")
model_surface_w.on_click(model_surface)

select_points_w = Button(label='Begin Point Selection', 
                         disabled=True, 
                         sizing_mode="stretch_width")

select_points_w.on_click(select_bathy)

calc_refract_w = Button(label='Calculate Refraction', 
                        disabled=True, 
                        sizing_mode="stretch_width")

calc_refract_w.on_click(correct_refraction)

out_text_w = Paragraph(text="""Select output file types (more coming soon...)""")

out_name_w = TextInput(title="Output file name base:", 
                       disabled=False, 
                       sizing_mode="stretch_width")

out_path_w = TextInput(title="Output directory:", 
                       value=os.getcwd(), 
                       disabled=False, 
                       sizing_mode="stretch_width")

out_path_w.on_change('value_input', check_out_dir_exists)

out_types_w = CheckboxGroup(labels=['csv'], active=[0], disabled=True, sizing_mode="stretch_width")
save_out_w = Button(label='Save Data to Output', disabled=False, sizing_mode="stretch_width")
save_out_w.on_click(save_output)
quit_w = Button(label='QUIT', disabled=False, sizing_mode="stretch_width")
quit_w.on_click(close_app)
# Organizing layout
left_half = column(file_in_w, gt_w, model_surface_w, select_points_w, 
                   calc_refract_w, out_path_w, out_name_w, out_text_w, 
                   out_types_w, save_out_w, quit_w, fig_imagery, 
                   sizing_mode="fixed", height=800, width=panel_width)

curdoc().add_root(row(left_half, fig_full_profile))
curdoc().title = 'Manual Classification'
