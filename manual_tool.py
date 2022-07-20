# must pass input --websocket-max-message-size=500000000 
# or whatever number is closest to the size of the h5 files youre reading in from local
# otherwise typical h5 file will not be able to be passed and the server will close

# example command
# bokeh serve --show manual_tool.py --websocket-max-message-size=500000000

""
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

# ########################## ICESAT2 FUNCTIONS ##############################


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

def bkapp(doc):
    # ########################## DATA SETUP ##############################

    surface_class = 41
    no_bottom_class = 45
    bathy_class = 40

    gt_labels = ['gt1r', 'gt1l', 'gt2r', 'gt2l', 'gt3r', 'gt3l']


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

    # bounding box
    bbox = pd.DataFrame([], index=['x', 'y', 'w', 'h']).T
    bbox_src = ColumnDataSource(bbox)



    # ########################## WIDGET / CALLBACK SETUP ##############################

    # Select input h5 file
    w_file_input = FileInput(accept=".h5", sizing_mode="stretch_width")

    def import_h5(attr, old, new):
        w_status_box.text='Loading h5...'
        decoded = b64decode(w_file_input.value)
        f = BytesIO(decoded)
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
        w_gt_select.disabled=False
        w_status_box.text='Successfully imported H5 file. Use tabs on left to select a beam.'

    w_file_input.on_change('value', import_h5)

    # Buttons to select which beam youre classifying
    w_gt_select = RadioButtonGroup(labels=gt_labels, 
                            disabled=True, sizing_mode="stretch_width")

    def select_profile(attr, old, new):
        # make the imagery window visible 
        imagery_renderer = fig_imagery.select_one({'name': 'tile_renderer'})    
        imagery_renderer.visible = True

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

        w_save_button.disabled=False
        surface_line = fig_primary.select_one({'name': 'surface_model'})    
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
        src_gt.data['x_wm'], src_gt.data['y_wm'] = tform.transform(src_gt.data['lat'], src_gt.data['lon'])

        # update imagery window
        fig_imagery.x_range.start = min(src_gt.data['x_wm'])
        fig_imagery.x_range.end = max(src_gt.data['x_wm'])    
        fig_imagery.y_range.start = min(src_gt.data['y_wm'])
        fig_imagery.y_range.end = max(src_gt.data['y_wm'])

        # update primary window details
        fig_primary.xaxis.axis_label = 'Latitude (deg)'
        fig_primary.yaxis.axis_label = 'Orthometric Height (m)'

        # enable water surface calculation button
        w_surface_button.disabled = False
        w_select_button.disabled = True
        w_refract_button.disabled = True
        w_save_button.disabled=True
        w_save_button.button_type='default'
        w_save_button.label='Save Data to Output'

        # update the base filename field
        w_out_name.value = re.sub('ATL03', 'BATHY', 
                                  w_file_input.filename[:-3] + '_' + gt_labels[w_gt_select.active].upper())

        # update primary plot/remove legend
        plt_src.data = dict(src_gt.data)

        fig_primary.legend.visible = False

    w_gt_select.on_change('active', select_profile)

    # Button to start water surface modeling
    w_surface_button = Button(label='Model Water Surface', disabled=True, sizing_mode="stretch_width")

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

            w_select_button.disabled = False
            w_surface_button.disabled = True

    w_surface_button.on_click(model_surface)

    # Button to begin point selection
    w_select_button = Button(label='Begin Point Selection', 
                             disabled=True, 
                             sizing_mode="stretch_width")

    def select_bathy():
        # make the lasso tool active

        # remove the water surface model from the plot
        # or get the glyph from the Figure:
        surface_line = fig_primary.select_one({'name': 'surface_model'})    
        surface_line.visible = False

        # remove the legend from the plot
        fig_primary.legend.visible = False

        # change the subsurface points to black just for this selection step
        subsurface_renderer = fig_primary.select_one({'name': 'subsurface'})
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
        w_refract_button.disabled = False
        w_select_button.disabled = True

    w_select_button.on_click(select_bathy)

    # Button to start refraction correction and update plots/data
    w_refract_button = Button(label='Calculate Refraction', 
                            disabled=True, 
                            sizing_mode="stretch_width")

    def correct_refraction():
        # switch subsurface photons back to gray for background
        subsurface_renderer = fig_primary.select_one({'name': 'subsurface'})
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
        w_select_button.disabled = True

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

        fig_primary.legend.visible = True

        # disable calculate refraction button
        w_refract_button.disabled=True
        w_save_button.disabled=False


    w_refract_button.on_click(correct_refraction)

    # Text box for specifying directory to output data
    w_out_path = TextInput(title="Output directory:", 
                           value=os.getcwd(), 
                           disabled=False, 
                           sizing_mode="stretch_width")


        # trying to change the text background red if the output directory doesnt exist - buggy
    def check_out_dir_exists(attr, old, new):
        if not Path(w_out_path.value_input).exists():
            w_out_path.background = 'red'
        else:
            w_out_path.background = 'white'

    w_out_path.on_change('value_input', check_out_dir_exists)


    # Text box for specifying what the output file name should be
    w_out_name = TextInput(title="Output file name base:", 
                           disabled=False, 
                           sizing_mode="stretch_width")

    # Check boxes to select output file types
    w_checkbox_text = Paragraph(text="""Select output file types (more coming soon...)""")
    w_checkbox_out_type = CheckboxGroup(labels=['csv'], active=[0], disabled=True, sizing_mode="stretch_width")

    # Button to write refraction corrected data to output files
    w_save_button = Button(label='Save Data to Output', 
                           disabled=True, 
                           sizing_mode="stretch_width")

    def save_output():
        csv_output_path = os.path.join(w_out_path.value, 
                                       w_out_name.value + '.csv')

        #combine surface data, bathy data, subsurface data
        df_comb = pd.concat([pd.DataFrame(bathy_src.data),
                             pd.DataFrame(sfc_src.data), 
                             pd.DataFrame(subsfc_src.data)])

        if 0 in w_checkbox_out_type.active:
            # csv checked
            df_out = df_comb.loc[:, 
                                 ['lon', 'lat', 
                                  'height_ortho', 'classification', 
                                  'surface_height','surface_sigma', 'dZ'] ]
            df_out.to_csv(csv_output_path, index=False)

        w_save_button.button_type='success'
        w_save_button.label='Output Saved!'
        w_save_button.disabled=True

        print()

    w_save_button.on_click(save_output)

    # Button to end the underlying bokeh server
    # need to add a lifecycle hook to close it on tab closure too
    w_quit_button = Button(label='END SESSION', 
                           disabled=False, 
                           button_type='danger', 
                           sizing_mode="stretch_width")

    def close_app():
        sys.exit

    w_quit_button.on_click(close_app)

    w_status_box = Div(text="""status updates to go here...""",
                       height=30)


    # Widgets/callbacks for Sliderule query tab
    # w_date_slider = DateRangeSlider(value=(datetime.date(2020, 6, 1), datetime.date(2020, 7, 1)),
    #                                     start=datetime.date(2018, 9, 15), end=datetime.date.today())

    w_date_picker_start = DatePicker(title='Start Date', 
                                     value="2021-06-01", 
                                     min_date="2018-09-15", 
                                     max_date=datetime.date.today(),
                                     width=150)
    w_date_picker_end = DatePicker(title='End Date', 
                                   value="2021-06-15", 
                                   min_date="2018-09-15", 
                                   max_date=datetime.date.today(), 
                                   width=150)

    # when start date is selected, update min date possible in end widget
    def update_end_date_widget(attr, old, new):
        w_date_picker_end.min_date = w_date_picker_start.value

    w_date_picker_start.on_change('value', update_end_date_widget)

    # when end date is selected, update max date possible in start widget
    def update_start_date_widget(attr, old, new):
        w_date_picker_start.max_date = w_date_picker_end.value

    w_date_picker_end.on_change('value', update_start_date_widget)

    # window to select bounding box
    # see figure call (fig_bbox) in format/plot section 

    # this here doesnt actually work
    # def bbox_updated(attr, old, new):
    #     w_status_box.text = 'New bounding box selected...'

    # bathy_src.on_change('data', bbox_updated)

    # which data release to download
    w_release_select = Select(title="Release Version:", value="005", options=["005"])
    w_surftype_select = Select(title='ATL03 Photon Surface Type', value='1', options=['0','1','2','3','4'])

    conf_list = ["atl03_tep", "atl03_not_considered", "atl03_background", "atl03_within_10m", "atl03_low", "atl03_medium", "atl03_high"]

    w_conf_select = MultiChoice(title="ATL03 Photon Confidence", value = conf_list, options=conf_list)

    # button to finalize bounding box and begin query
    w_query_button = Button(label='Submit Query', 
                             disabled=False, 
                             button_type='success', 
                             sizing_mode=None)

    def query_sliderule():

        # first, convert bokeh height/width format to bbox corners
        # upper left, bottom left, bottom right, upper right, upper left    

        x_wm_bbox = [bbox_src.data['x'][0] - bbox_src.data['w'][0]/2,
               bbox_src.data['x'][0] - bbox_src.data['w'][0]/2,
               bbox_src.data['x'][0] + bbox_src.data['w'][0]/2,
               bbox_src.data['x'][0] + bbox_src.data['w'][0]/2,
               bbox_src.data['x'][0] - bbox_src.data['w'][0]/2] 

        y_wm_bbox = [bbox_src.data['y'][0] + bbox_src.data['h'][0]/2,
               bbox_src.data['y'][0] - bbox_src.data['h'][0]/2,
               bbox_src.data['y'][0] - bbox_src.data['h'][0]/2,
               bbox_src.data['y'][0] + bbox_src.data['h'][0]/2,
               bbox_src.data['y'][0] + bbox_src.data['h'][0]/2] 

        # convert bbox from web mercator to lat/lon
        wgs84 = pyproj.crs.CRS.from_epsg(4979)
        web_merc = pyproj.crs.CRS.from_epsg(3857)
        tform = pyproj.transformer.Transformer.from_crs(crs_from=web_merc, crs_to=wgs84)

        lat, lon = tform.transform(x_wm_bbox, y_wm_bbox)

        # actual querying code
        url="icesat2sliderule.org"
        icesat2.init(url, verbose=True, loglevel=logging.DEBUG)
        asset = "nsidc-s3" 

        # convert bbox corners to Sliderule compatible region data
        sr_reg = icesat2.toregion( gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat)) )

        # Select release
        time_start = datetime.datetime.strptime(w_date_picker_start.value, "%Y-%m-%d").date().strftime('%Y-%m-%dT%H:%M:%SZ')
        time_end = datetime.datetime.strptime(w_date_picker_end.value, "%Y-%m-%d").date().strftime('%Y-%m-%dT%H:%M:%SZ')


        granules_list = icesat2.cmr(polygon=sr_reg[0], version=w_release_select.value, short_name='ATL03', 
                                    time_start=time_start, 
                                    time_end=time_end)
        print(granules_list)
        print(w_surftype_select.value, w_conf_select.value, w_release_select.value)
        params = {}
        params['poly'] = sr_reg[0]
        params['srt'] = int(w_surftype_select.value)
        params['cnf'] = w_conf_select.value
        print('querying...')
        gdf = icesat2.atl03sp(params, asset=asset, version=w_release_select.value, resources=granules_list)
        print('DONE')
        print(gdf.head())

        # ADD track lines to plot with hover tool
        # coords/conversions
        gdf.insert(0, 'lat', gdf.geometry.y, False) 
        gdf.insert(0, 'lon', gdf.geometry.x, False)

        # convert from wm to lat lon
        wgs84 = pyproj.crs.CRS.from_epsg(4979)
        web_merc = pyproj.crs.CRS.from_epsg(3857)
        tform = pyproj.transformer.Transformer.from_crs(crs_from=wgs84, crs_to=web_merc)

        x_wm, y_wm = tform.transform(gdf.lat, gdf.lon)
        gdf.insert(0, 'x_wm', x_wm, False) 
        gdf.insert(0, 'y_wm', y_wm, False)

        gdf.reset_index(inplace=True)

        print('collecting data for multiline plotting...')

        # must speed this up, but ok for testing plots

        data = dict(x=[], y=[], lon=[], lat=[], rgt=[], cycle=[], pair=[], track=[], date=[])

        for rgt_ in gdf.rgt.unique():
            for cycle_ in gdf.cycle.unique():
                for pair_ in gdf.pair.unique():
                    for track_ in gdf.track.unique():
                        print(rgt_, cycle_, pair_, track_)
                        gdf_ = reduce_gdf(gdf, RGT=rgt_, track=track_, cycle=cycle_, pair=pair_)

                        if gdf_.shape[0] > 0:
                            # downsample...
                            if gdf_.shape[0] > 10:
                                #get evenly spaced indices of about 1,000 points
                                gdf_ = gdf_.iloc[np.floor(np.linspace(0, gdf_.shape[0]-1, np.int64(10))), :]


                            data['date'].append(gdf_.iloc[0].time.date())
                            data['rgt'].append(rgt_)
                            data['track'].append(track_)
                            data['cycle'].append(cycle_)
                            data['pair'].append(pair_)
                            data['x'].append(gdf_.x_wm.values)
                            data['y'].append(gdf_.y_wm.values)

                            data['lat'].append(gdf_.lat.values)
                            data['lon'].append(gdf_.lon.values)

                        # append data array 
        rake_src = ColumnDataSource(data)
        glyph = MultiLine(xs="x", ys="y", line_width=2, line_color='lawngreen', name='track_rake')
        gr = fig_bbox.add_glyph(rake_src, glyph)
        hover = HoverTool(tooltips =[
            ("Date", "@date"),
            ("RGT", "@rgt"),
            ("Cycle", "@cycle"),
            ("Track", "@track"),
            ("Pair", "@pair")
            ])

        fig_bbox.add_tools(hover)

        # Make bbox invisible so it doesnt show up on hover tools
        bb_render.visible = False

    w_query_button.on_click(query_sliderule)

    ########################### FORMATTING / PLOTTING ##############################
    panel_width = 350
    panel_height = 800

    # Imagery window
    tile_provider = get_provider('ESRI_IMAGERY')

    fig_imagery = figure(title='', 
                         x_range=(-6000000, 6000000), 
                         y_range=(-1000000, 7000000),
                         x_axis_type="mercator", 
                         y_axis_type="mercator", 
                         tools='zoom_in,zoom_out,pan,box_zoom,wheel_zoom,undo,redo,reset,', sizing_mode="stretch_height", 
                         width=panel_width, toolbar_location='below')

    fig_imagery.add_tile(tile_provider, name='tile_renderer', visible=False)
    trackline = fig_imagery.line(x='x_wm', y='y_wm', source=src_gt, color='red', line_width=1)

    # Bounding box selection window
    fig_bbox = figure(x_range=(-9240000, -8460000), 
                      y_range=(2450000, 3000000),
                      x_axis_type="mercator", 
                      y_axis_type="mercator", 
                      tools='zoom_in,zoom_out,pan,box_zoom,wheel_zoom,undo,redo,reset', height=350, sizing_mode='stretch_width')
    fig_bbox.add_tile(tile_provider)

    bb_render = fig_bbox.rect(x="x", y="y", width="w", height="h", source=bbox_src, 
                              color='red', fill_alpha=0.1, line_width=5)

    box_edit_tool = BoxEditTool(renderers=[bb_render], num_objects=1)
    fig_bbox.add_tools(box_edit_tool)
    # fig_bbox.toolbar.active_drag = box_edit_tool


    # Primary window 
    fig_primary = figure(title='', 
                         tools='zoom_in,zoom_out,pan,box_zoom,wheel_zoom,undo,redo,reset,lasso_select',
                              sizing_mode="scale_width", height=500,
                              x_axis_type='mercator', x_range=fig_imagery.y_range)#, x_range=fig_imagery.y_range)

        # initialize glyphs from data so that changes to data update plots automatically

    # plots are actually using web-mercator coords under the hood
    # helps with mapping axes to the imagery window 

    wm_axis = 'y_wm'

    fig_primary.circle(x=wm_axis, y='height_ortho', 
                            source=plt_src, size=0.5,
                            color='black')

    fig_primary.circle(x=wm_axis, 
                            y='height_ortho', 
                            source=sfc_src, 
                            size=0.5,
                            color='blue', 
                            legend_label='Water Surface (Class {})'.format(surface_class))

    fig_primary.line(x=wm_axis, y='surface_height', source=plt_src, 
                            color='lime', line_width=3,
                            name='surface_model')

    fig_primary.circle(x=wm_axis, 
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

    fig_primary.circle(x=wm_axis, 
                            y='height_ortho',
                            source=bathy_src,
                            size=0.5,
                            color='black', 
                            legend_label='Uncorrected Bathymetry (Class {})'.format(bathy_class))

    fig_primary.circle(x=wm_axis, 
                            y='height_ortho_c',
                            source = bathy_src,
                            size=0.5,
                            color='red', 
                            legend_label='Bathymetry, Refraction Corrected'.format(bathy_class))

    fig_primary.legend.location = "bottom_left"
    fig_primary.legend.visible = False

    # Organizing layout of widgets
    left_column = column(w_file_input, w_gt_select, w_surface_button, w_select_button, 
                       w_refract_button, w_out_path, w_out_name, w_checkbox_text, 
                       w_checkbox_out_type, w_save_button, w_quit_button, fig_imagery, 
                       sizing_mode="fixed", height=panel_height, width=panel_width)

    sliderule_layout=column(fig_bbox,
                            row(w_date_picker_start, w_date_picker_end, w_release_select, w_surftype_select),
                            w_conf_select,
                            w_query_button) # 

    # combining layouts into panels
    sliderule_panel = Panel(child=sliderule_layout, title='Data Source')

    fig_primary_panel = Panel(child=fig_primary, title='Photon Cloud')

    # combining panels into a tab unit
    window_tabs = Tabs(tabs=[fig_primary_panel, sliderule_panel])

    # callback that triggers when the user activates different tabs
    def tab_switched(attr, old, new):
        if window_tabs.active == 1:
            w_status_box.text = '''Select box edit tool on right, hold shift, then click and drag anywhere on the plot or double tap once to start drawing. 
            Move the mouse and double tap again to finish drawing bounding box. '''

    window_tabs.on_change('active', tab_switched)


    right_column = column(w_status_box, window_tabs, sizing_mode="stretch_width")
    print('this is the end')
    doc.add_root(row(left_column, right_column))
    doc.title = 'OpenOceans Manual Classification Tool'

# Setting num_procs here means we can't touch the IOLoop before now, we must
# let Server handle that. If you need to explicitly handle IOLoops then you
# will need to use the lower level BaseServer class.
server = Server({'/': bkapp}, num_procs=4, websocket_max_message_size=3000000000)
server.start()

if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()