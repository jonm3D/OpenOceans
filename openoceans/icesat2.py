import re
import numpy as np
import h5py
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from astropy.time import Time, TimeDatetime
import progressbar
import warnings
import time
import pyproj
from shapely.geometry import Point
import simplekml
import os
import matplotlib.pyplot as plt
import hdbscan
import seaborn as sns
import pprint
import json

from bokeh.plotting import figure
from bokeh.io import show, output_notebook
from bokeh.layouts import row
from bokeh.tile_providers import get_provider
tile_provider = get_provider('ESRI_IMAGERY')

# Need to address section number in ATL03 name detail

def beam_info(gtxx, sc_orient):

    # check gt format
    gtxx = gtxx.lower()
    if re.search('gt[1-3][lr]', gtxx) is None:
        raise Exception(
            'Unrecognized beam format. Use GT#(L/R) - for example, \'GT1R\'.')

    # check sc_orient format
    if len(sc_orient) == 1:
        sc_orient = sc_orient[0]
        assert (sc_orient == 0) or (sc_orient == 1)
    else:
        raise Exception(
            'Spacecraft transition detected, unable to get beam info.')

    if sc_orient == 1:
        # forward orientation

        if gtxx == 'gt1l':
            strength = 'weak'
            spot = 6
            pair = 1

        elif gtxx == 'gt1r':
            strength = 'strong'
            spot = 5
            pair = 1

        elif gtxx == 'gt2l':
            strength = 'weak'
            spot = 4
            pair = 2

        elif gtxx == 'gt2r':
            strength = 'strong'
            spot = 3
            pair = 2

        elif gtxx == 'gt3l':
            strength = 'weak'
            spot = 2
            pair = 3

        elif gtxx == 'gt3r':
            strength = 'strong'
            spot = 1
            pair = 3

    elif sc_orient == 0:
        # backward orientation

        if gtxx == 'gt1l':
            strength = 'strong'
            spot = 1
            pair = 1

        elif gtxx == 'gt1r':
            strength = 'weak'
            spot = 2
            pair = 1

        elif gtxx == 'gt2l':
            strength = 'strong'
            spot = 3
            pair = 2

        elif gtxx == 'gt2r':
            strength = 'weak'
            spot = 4
            pair = 2

        elif gtxx == 'gt3l':
            strength = 'strong'
            spot = 5
            pair = 3

        elif gtxx == 'gt3r':
            strength = 'weak'
            spot = 6
            pair = 3

    beam_info_dict = {'beam_strength': strength,
                      'atlas_spot': spot,
                      'track_pair': pair}

    return beam_info_dict


class Profile:
    def __init__(self, data=None, aoi=None, info=None):

        # area of interest
        # if aoi is provided, clip photon data
        if aoi is not None:
            data = self._clip_to_aoi(data, aoi)

        self.aoi = aoi

        # details about the track (date, gtxx, rgt, cycle)
        self.info = info

        # photon resolution satellite data
        # add land classifications

        if aoi is not None:
            shoreline_bbox = aoi.poly.bounds
        else:
            shoreline_bbox = None

        try:

            land_point_labels = self._find_land_points(
                data, bbox=shoreline_bbox)

            data.loc[:, 'is_land'] = land_point_labels

            # data.loc[land_idx, 'is_land'] = 0

        except:
            pass

        # save to Profile object
        self.data = data

        self.signal_finding = False
        # have a property for height datum and/or x,y crs?

    def __str__(self):
        '''Human-readable description.'''
        desc = f"""
TRACK DETAILS
    D/M/Y: {self.info['date'].strftime('%d/%m/%Y')}
    Reference Ground Track: {self.info['rgt']}
    Cycle: {str(self.info['cycle'])}
    Region: {str(self.info['region'])}
    Orbit Direction: {self.info['orbit_dir']}
    Release: {str(self.info['release'])}

BEAM DETAILS
    {self.info['gtxx'].upper()}
    {self.info['beam_strength'].upper()} BEAM
    ATLAS SPOT {str(self.info['atlas_spot'])} 
    
ICESat-2 Profile: {self.get_atl03_name_pattern()}
Photons Returned: {self.data.shape[0]}
"""
        return desc

    @classmethod
    def from_h5(cls, filepath, gtxx, conf_type=1, aoi=None, verbose=False):
        # read in a single track from an h5 file
        # def read_h5_as_df(h5_file_bytes_io, gtxx, conf_type=1, verbose=False):

        # check validity of  input values
        if ((conf_type <= 0) or (conf_type >= 4)):
            raise Exception('''Requested confidence type must be between 
                0 and 4 (land, ocean, sea ice, land ice and inland water).''')

        if isinstance(gtxx, str):
            # user input a single string like 'gt1r'
            gtxx = gtxx.lower()

            if re.search('gt[1-3][lr]', gtxx) is None:
                raise Exception(
                    'Unrecognized beam format. Use GT#(L/R) - for example, \'GT1R\'.')

            # # set up in iterable
            # gtxx = [gtxx]

        else:
            raise Exception(
                '''Requested beams must be a string (e.g.\'gt1r\')''')

        if verbose:
            ProgressBar = progressbar.ProgressBar(max_value=7)
        else:
            ProgressBar = progressbar.NullBar(max_value=7)

        with ProgressBar as bar:
            with h5py.File(filepath, 'r') as f:

                # initialize data storage variables
                pho = dict()
                seg = dict()
                anc = {}

                if verbose:
                    print(gtxx.upper())

                if verbose:
                    print('Reading photon-resolution data...')

                bar.update(1)

                ##### PHOTON RESOLUTION DATA #####

                lat_ph = np.array(f[gtxx + '/heights/lat_ph'])

                lon_ph = np.array(f[gtxx + '/heights/lon_ph'])

                # this function is a significant slowdown without pygeos
                pho['geometry'] = gpd.points_from_xy(lon_ph, lat_ph)

                pho['delta_time'] = np.array(f[gtxx + '/heights/delta_time'])

                pho['height'] = np.array(f[gtxx + '/heights/h_ph'])

                # sliderule documentation not clear on this - skipping
                # photons distance from the start of the segment (ref photon)
                pho['dist_ph_along_from_seg'] = np.array(
                    f[gtxx + '/heights/dist_ph_along'])

                pho['quality_ph'] = np.array(f[gtxx + '/heights/quality_ph'])

                signal_conf = np.array(f[gtxx + '/heights/signal_conf_ph'])

                # user input signal type
                pho['atl03_cnf'] = signal_conf[:, conf_type]

                # update progress bar

                bar.update(2)

                if verbose:
                    print('Reading ancillary data...')

                ##### ANCILLARY / ORBIT DATA #####
                anc['gtxx'] = gtxx

                anc['filepath'] = filepath

                # variable resolution

                anc['atlas_sdp_gps_epoch'] = np.array(
                    f['/ancillary_data/atlas_sdp_gps_epoch'])

                anc['region'] = np.array(f['/ancillary_data/start_region'])[0]

                # use region to determine track direction
                if (anc['region'] <= 3) or (anc['region'] >= 12):
                    anc['orbit_dir'] = 'ASCENDING'

                elif (anc['region'] == 4) or (anc['region'] == 11):
                    anc['orbit_dir'] = 'POLE'

                else:
                    anc['orbit_dir'] = 'DESCENDING'

                # check on ATL03 behavior
                if np.array(f['/ancillary_data/start_region'])[0] != np.array(f['/ancillary_data/end_region'])[0]:
                    warnings.warn(
                        'Start/end region numbers differ - figure out why and handle this behavior.')

                # replace() fixes whitespace at end of string
                anc['release'] = np.array(
                    f['/ancillary_data/release'])[0].decode('UTF-8').replace(" ", "")

                anc['sc_orient'] = np.array(
                    f['/orbit_info/sc_orient'])

                anc['sc_orient_time'] = np.array(
                    f['/orbit_info/sc_orient'])

                anc['rgt'] = np.array(
                    f['/orbit_info/rgt'])[0]

                pho['rgt'] = np.int64(anc['rgt'] * np.ones(len(lat_ph)))

                anc['orbit_number'] = np.array(
                    f['/orbit_info/orbit_number'])[0]

                anc['cycle_number'] = np.array(
                    f['/orbit_info/cycle_number'])[0]

                pho['region'] = np.int64(
                    anc['region'] * np.ones(len(lat_ph)))

                pho['cycle'] = np.int64(
                    anc['cycle_number'] * np.ones(len(lat_ph)))

                if len(anc['sc_orient']) == 1:
                    # no spacecraft transitions during this granule
                    pho['sc_orient'] = np.int64(
                        anc['sc_orient'][0] * np.ones(len(lat_ph)))
                else:
                    warnings.warn(
                        'Spacecraft in transition detected! sc_orient parameter set to -1.')
                    pho['sc_orient'] = np.int64(-1 * np.ones(len(lat_ph)))

                # track / pair data from user input
                # is this correct
                pho['track'] = np.int64(gtxx[2]) * \
                    np.ones(len(lat_ph), dtype=np.int64)

                if gtxx[3].lower() == 'r':
                    pair_val = 1

                elif gtxx[3].lower() == 'l':
                    pair_val = 0

                pho['pair'] = pair_val * np.ones(len(lat_ph), dtype=np.int64)

                if verbose:
                    print('Reading segment resolution data and upsampling...')

                bar.update(3)

                ##### SEGMENT RESOLUTION DATA #####
                seg['ref_azimuth'] = np.array(
                    f[gtxx + '/geolocation/ref_azimuth'])

                seg['ref_elev'] = np.array(
                    f[gtxx + '/geolocation/ref_elev'])

                seg['segment_id'] = np.array(
                    f[gtxx + '/geolocation/segment_id'])

                seg['segment_dist_x'] = np.array(
                    f[gtxx + '/geolocation/segment_dist_x'])

                seg['solar_azimuth'] = np.array(
                    f[gtxx + '/geolocation/solar_azimuth'])

                seg['solar_elev'] = np.array(
                    f[gtxx + '/geolocation/solar_elevation'])

                seg['ph_index_beg'] = np.array(
                    f[gtxx + '/geolocation/ph_index_beg'])

                seg['segment_ph_cnt'] = np.array(
                    f[gtxx + '/geolocation/segment_ph_cnt'])

                seg['geoid'] = np.array(
                    f[gtxx + '/geophys_corr/geoid'])

                surf_type = np.array(
                    f[gtxx + '/geolocation/surf_type'])

                seg['surf_type_land'] = surf_type[:, 0]
                seg['surf_type_ocean'] = surf_type[:, 1]
                seg['surf_type_inland_water'] = surf_type[:, 4]

                seg['full_sat_fract'] = np.array(
                    f[gtxx + '/geolocation/full_sat_fract'])

                seg['near_sat_fract'] = np.array(
                    f[gtxx + '/geolocation/full_sat_fract'])

                bar.update(4)

                ##### UPSAMPLE SEGMENT RATE DATA #####
                pho['segment_id'] = np.int64(cls._convert_seg_to_ph_res(segment_res_data=seg['segment_id'],
                                                                        ph_index_beg=seg['ph_index_beg'],
                                                                        segment_ph_cnt=seg['segment_ph_cnt'],
                                                                        total_ph_cnt=len(lat_ph)))

                pho['segment_dist'] = cls._convert_seg_to_ph_res(segment_res_data=seg['segment_dist_x'],
                                                                 ph_index_beg=seg['ph_index_beg'],
                                                                 segment_ph_cnt=seg['segment_ph_cnt'],
                                                                 total_ph_cnt=len(lat_ph))

                pho['ref_azimuth'] = cls._convert_seg_to_ph_res(segment_res_data=seg['ref_azimuth'],
                                                                ph_index_beg=seg['ph_index_beg'],
                                                                segment_ph_cnt=seg['segment_ph_cnt'],
                                                                total_ph_cnt=len(lat_ph))
                pho['ref_elev'] = cls._convert_seg_to_ph_res(segment_res_data=seg['ref_elev'],
                                                             ph_index_beg=seg['ph_index_beg'],
                                                             segment_ph_cnt=seg['segment_ph_cnt'],
                                                             total_ph_cnt=len(lat_ph))

                pho['geoid_z'] = cls._convert_seg_to_ph_res(segment_res_data=seg['geoid'],
                                                            ph_index_beg=seg['ph_index_beg'],
                                                            segment_ph_cnt=seg['segment_ph_cnt'],
                                                            total_ph_cnt=len(lat_ph))

                pho['surf_type_land'] = np.int64(cls._convert_seg_to_ph_res(segment_res_data=seg['surf_type_land'],
                                                                            ph_index_beg=seg['ph_index_beg'],
                                                                            segment_ph_cnt=seg['segment_ph_cnt'],
                                                                            total_ph_cnt=len(lat_ph)))

                pho['surf_type_ocean'] = np.int64(cls._convert_seg_to_ph_res(segment_res_data=seg['surf_type_ocean'],
                                                                             ph_index_beg=seg['ph_index_beg'],
                                                                             segment_ph_cnt=seg['segment_ph_cnt'],
                                                                             total_ph_cnt=len(lat_ph)))

                pho['surf_type_inland_water'] = np.int64(cls._convert_seg_to_ph_res(segment_res_data=seg['surf_type_inland_water'],
                                                                                    ph_index_beg=seg['ph_index_beg'],
                                                                                    segment_ph_cnt=seg['segment_ph_cnt'],
                                                                                    total_ph_cnt=len(lat_ph)))

                # calculating photon along track distance from upsampled segment distance
                pho['dist_ph_along'] = pho['segment_dist'] + \
                    pho['dist_ph_along_from_seg']

                bar.update(5)

                if verbose:
                    print('Converting times...')

                ##### TIME CONVERSION / INDEXING #####

                t_gps = Time(anc['atlas_sdp_gps_epoch'] +
                             pho['delta_time'], format='gps')
                t_utc = Time(t_gps, format='iso', scale='utc')
                pho['time'] = t_utc.datetime

                bar.update(6)

                # sorting into the same order as sliderule df's purely for asthetics
                if verbose:
                    print('Concatenating data...')

                df = gpd.GeoDataFrame(pho)

                # df.set_index('time', inplace=True)

                df = df.loc[:, ['time', 'rgt', 'cycle', 'region', 'track', 'segment_id', 'segment_dist',
                                'sc_orient', 'atl03_cnf', 'height', 'quality_ph', 'delta_time', 'pair', 'geometry',
                                'ref_azimuth', 'ref_elev', 'dist_ph_along',
                                'surf_type_land', 'surf_type_ocean', 'surf_type_inland_water', 'geoid_z']]

                bar.update(7)

                df.insert(0, 'lat', df.geometry.y, False)
                df.insert(0, 'lon', df.geometry.x, False)

                # wgs84 = pyproj.crs.CRS.from_epsg(4979)
                # wgs84_egm08 = pyproj.crs.CRS.from_epsg(3855)
                # tform = pyproj.transformer.Transformer.from_crs(crs_from=wgs84, crs_to=wgs84_egm08)
                # _, _, z_g = tform.transform(df.geometry.y, df.geometry.x, df.height)
                # df.insert(0, 'height_ortho', z_g, False)

                # allocation of to be used arrays
                zero_int_array = np.int64(np.zeros_like(df.geometry.x))

                # insert() usage reminder: loc, column, value, allow_duplications
                df.insert(0, 'classification',
                          zero_int_array - 999, False)

                df.insert(0, 'signal',
                          zero_int_array, False)

                # Land flag initialized as -1
                # If shorelines downloaded already, will be set to 0 or 1
                df.insert(0, 'is_land',
                          zero_int_array - 1, False)

                df.set_crs("EPSG:4326", inplace=True)

                track_info_dict = {'date': df.time[0].date(),
                                   'rgt': anc['rgt'],
                                   'gtxx': gtxx,
                                   'cycle': anc['cycle_number'],
                                   'region': anc['region'],
                                   'orbit_dir': anc['orbit_dir'],
                                   'release': anc['release'],
                                   'path': os.path.abspath(filepath)}

                beam_info_dict = beam_info(gtxx, anc['sc_orient'])

                # combine the two dicts
                info_dict = {**track_info_dict, **beam_info_dict}

        return cls(data=df, info=info_dict, aoi=aoi)

    def get_atl03_name_pattern(self):
        '''Returns a regex pattern matching the original ATL03 filename.'''
        date = self.info['date'].strftime('%Y%m%d')
        rgt = str(self.info['rgt']).zfill(4)
        cc = str(self.info['cycle']).zfill(2)
        reg = str(self.info['region']).zfill(2)
        rel = self.info['release']
        return "ATL03_{}{}_{}{}{}_{}".format(date, '*', rgt, cc, reg, rel)

    def get_formatted_filename(self):
        '''Returns a descriptive string that uniquely defines the ATL03 file/track.'''
        date = self.info['date'].strftime('%Y-%m-%d')
        rgt = str(self.info['rgt']).zfill(4)
        cc = str(self.info['cycle']).zfill(2)
        reg = str(self.info['region']).zfill(2)
        rel = self.info['release']
        gtxx = self.info['gtxx'].upper()
        return "{}_rgt{}_cyc{}_reg{}_rel{}_beam{}".format(date, rgt, cc, reg, rel, gtxx)

    @classmethod
    def load_sample(cls, aoi=None):

        # relative path to atl03 sample data
        try:

            filepath = os.path.join(
                'sample_data', 'gbr_reef_ATL03_20210817155409_08401208_005_01.h5')

            absolute_filepath = os.path.abspath(filepath)

            sample_profile = cls.from_h5(absolute_filepath, 'gt2r', aoi=aoi)

            return sample_profile

        except:
            print('Sample data not found.')

            return None

    @staticmethod
    def _along_track_subsample(dist_ph_along, meters=1000):
        '''Gets indices to subsample data at an approximate along-track resolution. 
        This is primarily used for reducing the quantity of data when plotting tracks over imagery.'''

        # reset starting along track distance (0, instead of equator crossing)
        ph_at = dist_ph_along - dist_ph_along.iloc[0]

        # round all data to nearest X km, creating duplicate values within each chunk
        at_rounded = meters * np.round(ph_at/meters)

        # reset index from time to plain integers
        # these integers are later used as indices for the original data
        at_rounded = at_rounded.reset_index(drop=True)

        # get only the first of each duplicate value
        at_dropped = at_rounded.drop_duplicates()

        # data sampled at approximately every X km
        i_sample = at_dropped.index.values

        # add in last value in array for completeness if not there by chance
        if i_sample[-1] != len(dist_ph_along)-1:
            i_sample = np.append(i_sample, len(dist_ph_along)-1)

        # indices of photons approximately X meters apart + start/end photons
        # i_sample

        # how far off the distances between photons are from the requested spacing
        sampling_residuals = np.diff(dist_ph_along.iloc[i_sample]) - meters

        return i_sample, sampling_residuals

    # this is the quick and easy interface for the user
    def along_track_subsample(self, meters=1000):

        # directly subsample the photon data at the closest points to some rate
        # might be weird depending on the data
        # returns the residuals for your own validation step

        i_sample, residuals = self._along_track_subsample(
            self.data.dist_ph_along, meters=meters)

        return self.data.iloc[i_sample]

    @staticmethod
    def _convert_seg_to_ph_res(segment_res_data,
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

    def to_kml(self, filename=None, meters=1000, verbose=True):
        """Generates a kml file of the ICESat-2 ground track from lon/lat data.

        Args:
            filename (str, optional): Filename to save KML data. Defaults to the built-in formatted filename from the Profile class.
            meters (int, optional): Resolution to export photon data. Defaults to a photon every 1000m.
            verbose (bool, optional): Whether to print location that file is saved. Defaults to True.
        """

        # if unspecified, use unique readable filename
        if filename == None:
            filename = self.get_formatted_filename()+'.kml'

        '''Export track metadata and lon/lat (no elevations) at x km along track to kml'''
        i_sample, _ = self._along_track_subsample(
            self.data.dist_ph_along, meters=meters)

        # the folder will be open in the table of contents
        kml = simplekml.Kml(open=1)
        linestring = kml.newlinestring(
            name=self.get_formatted_filename(), description=self.__str__())
        linestring.coords = self.data.iloc[i_sample].loc[:, [
            'lon', 'lat']].values
        linestring.style.linestyle.color = 'ff0000ff'  # Red

        # linestring.style.linestyle.color = '39ff14'  # green

        linestring.style.linestyle.width = 2  # 5 pixels
        kml.save(filename)
        if verbose:
            print(f'Saved! At {os.path.abspath(filename)}')

    # def to_csv(self):
    #     '''Export photon data to CSV'''
    #     pass

    # def to_las(self):
    #     '''Export photon data to LAS 1.4'''
    #     pass

    def explore(self):
        '''Open bokeh plot of track photons vs imagery'''
        # should open html if run from terminal, widget if notebook

        # subsample photon data for quicker plotting? simpler seems easier for now
        df = self.data.loc[:, ['segment_id', 'lat', 'lon']]
        df_ = df.groupby('segment_id').median()

        # convert sampled gdf lat lon to web mercator for plotting
        wgs84 = pyproj.crs.CRS.from_epsg(4979)
        web_merc = pyproj.crs.CRS.from_epsg(3857)
        tform = pyproj.transformer.Transformer.from_crs(
            crs_from=wgs84, crs_to=web_merc)
        # underscore denotes subsampled datarate
        y_wm_, x_wm_ = tform.transform(
            df_.lat, df_.lon)

        y_wm, x_wm = tform.transform(
            self.data.geometry.y, self.data.geometry.x)
        h = self.data.height

        # plot trackline over imagery
        title_string = self.info['date'].strftime(
            '%Y/%m/%d') + ' ' + self.info['beam_strength'].upper() + ' BEAM, PAIR ' + str(self.info['track_pair'])
        fig_map = figure(title=title_string,
                         x_range=(min(y_wm), max(y_wm)),
                         y_range=(min(x_wm)-(max(y_wm)-min(y_wm)) * 0.2,
                                  max(x_wm)+(max(y_wm)-min(y_wm)) * 0.2),  # zooms out proportionally
                         #  width=400, height=400,
                         sizing_mode='stretch_both',
                         x_axis_type="mercator", y_axis_type="mercator",
                         tools='zoom_in, zoom_out, wheel_zoom, pan')

        fig_map.add_tile(tile_provider)

        fig_map.line(y_wm_, x_wm_, color='red', line_width=1)

        # plot photon data
        fig = figure(x_axis_type="mercator",
                     #  width=400, height=400,
                     sizing_mode='stretch_both',
                     x_range=fig_map.y_range,
                     tools='zoom_in,zoom_out,pan,box_zoom,wheel_zoom, undo, redo')

        fig.circle(x_wm, h, size=0.15, color='black')

        # format layout and show
        fig_grid = row([fig_map, fig])

        show(fig_grid)

    def label_photons(self):
        '''Stripped down generic photon labeler'''
        print('Not yet implemented!')
        return None
        # dropdown menu to select label for assigning photons

    def show_photons(self, x_unit='along track', y_lim=None, x_lim=None):

        plt.figure()
        if x_unit == 'along track':
            x = self.data.dist_ph_along - self.data.dist_ph_along.min()
            xlabel = 'Along Track Distance (m)'

        elif x_unit == 'lat':
            x = self.data.lat
            xlabel = 'Latitude (deg)'

        else:
            print('unknown x unit - supports along track or lat')
            x = self.data.lat
            xlabel = 'Latitude (deg)'

        z = self.data.height - self.data.geoid_z

        plt.plot(x, z, 'k.', markersize=1, alpha=0.6, label='All Photons')

        # plotting specific classes

        # check to see if the photons have been classified
        if np.all(self.data.classification == -999):
            # proceed without plotting specific classes
            pass

        else:
            labels = self.data.classification

            surf_idx = labels == 41
            bathy_idx = labels == 40
            column_idx = labels == 45

            # plot classes
            plt.plot(x[surf_idx], z[surf_idx], 'r.',
                     markersize=1, alpha=0.75, label='Surface (41)')
            plt.plot(x[bathy_idx], z[bathy_idx], 'bo',
                     markersize=1, alpha=0.75, label='Bathymetry (41)')
            plt.plot(x[column_idx], z[column_idx], 'g.',
                     markersize=1, alpha=0.75, label='Column (45)')

        # tidying up
        plt.xlabel(xlabel)
        plt.ylabel('Elevation (m)')
        plt.ylim(y_lim)
        plt.xlim(x_lim)
        plt.legend()
        plt.title(self.get_formatted_filename())

        return plt.gcf(), plt.gca()

    def find_signal(self, min_cluster_size=10, min_samples=1, cluster_selection_epsilon=0.75,
                    along_track_scaling=33., in_place=False, show_plot=False):
        """Using HDBSCAN algorithm to determine signal. Note that we are not employing HDBSCAN for the segmentation of subsurface/surface clusters, only for the detection of clustered data. Any photons detected as part of a cluster are labeled signal.

        For more information, see https://hdbscan.readthedocs.io/en/latest/

        Args:
            along_track_scaling (int, optional): Scaling factor for along track distance in meters. Defaults to 33 (0.3 sampling * 3 ~ 1m density proportion).
            min_cluster_size (int, optional): Smallest size grouping which is considered a cluster. Defaults to 15.
            min_samples (int, optional): Determines clusterer conservativeness, with higher values meaning more conservative results. Defaults to 1.
            cluster_selection_epsilon (int, optional): Distance within which to merge clusers. Defaults to 0.75 to match average sea surface return width.
            in_place (bool, optional): Whether to update the profile dataframe with new signal labels or return the label array. Defaults to False.

        Returns:
            _type_: _description_
        """

        self.signal_finding = True

        # scale along track and vertical axes
        # 11m along track for every 1 meter vertical - is2 footprint
        # dist_ph_along_scaled = (p.data.dist_ph_along - p.data.dist_ph_along.min()) / 11.
        dist_ph_along_scaled = (
            self.data.dist_ph_along - self.data.dist_ph_along.min()) / along_track_scaling

        # # tidy up along track data
        # data = np.vstack([dist_ph_along_scaled, p.data.height - p.data.geoid_z]).T
        data = np.vstack(
            [dist_ph_along_scaled, self.data.height - self.data.geoid_z]).T

        # initialize/fit hdbscan clusterer
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    cluster_selection_epsilon=cluster_selection_epsilon).fit(data)

        labels = (clusterer.labels_)

        # remove any photons 50 meters above or 100m below the geoid
        labels[data[:, 1] > 50] = -1
        labels[data[:, 1] < -100] = -1

        if show_plot:
            # plotting things
            color_palette = sns.color_palette("hls", max(labels)+1)
            cluster_colors = [color_palette[x] if x >= 0
                              else (0.5, 0.5, 0.5)
                              for x in labels]
            cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                     zip(cluster_colors, clusterer.probabilities_)]

            plt.figure()
            plt.scatter(x=data[:, 0], y=data[:, 1], s=10,
                        linewidth=0, c=cluster_member_colors, alpha=0.8)
            plt.show()

        if in_place:

            self.data.loc[:, 'signal'] = labels >= 0

            return None

        else:

            return labels, clusterer

    def plot_signal(self):

        # along track distance
        x = self.data.dist_ph_along - min(self.data.dist_ph_along)

        # geoidal height
        y = self.data.height - self.data.geoid_z

        # signal
        s = self.data.signal

        plt.figure()
        plt.plot(x, y, 'k.', label='Noise')
        plt.plot(x[s], y[s], 'r.', label='Signal')
        plt.xlabel('Along Track Distance (m)')
        plt.ylabel('EGM08 Height (m)')
        plt.legend(loc='upper right')
        plt.show()

    @staticmethod
    def _find_land_points(gdf, bbox):

        # assuming use of geoaois usage for bbox is geoaoi.poly.bounds

        # For Natural earth data implementation as of 1/6/2023
        # requires that the input gdf has ranged index values i
        # will need to change if index is changed to time or something
        # this currently checks point in polygon for EVERY point
        # would be significantly sped up if evaluated at 10m or something similar
        # maybe later, fine for now
        ########

        # try loading the shoreline data
        try:
            this_file = os.path.abspath(__file__)

            # navigating relative imports - I think theres probably a better way than this
            # Get the directory where the current file is stored
            current_dir = os.path.dirname(__file__)
            # Get the parent directory of the current directory
            parent_dir = os.path.dirname(current_dir)
            parent_dir_path = os.path.abspath(parent_dir)

            shoreline_data_path = os.path.join(
                parent_dir_path, 'shorelines', 'ne_10m_land', 'ne_10m_land.shp')
            land_polygon_gdf = gpd.read_file(shoreline_data_path, bbox=bbox)

            # continue with getting a new array of 0-or-1 labels for each photon
            new_labels = np.zeros_like(gdf.is_land.values)

            # update labels for points in the land polygons
            pts_in = gpd.sjoin(gdf, land_polygon_gdf, predicate='within')
            land_loc = gdf.index.isin(pts_in.index)  # bool

            new_labels[land_loc] = 1
            new_labels[~land_loc] = 0

            return new_labels

        except Exception as e:

            print(e)

            print("Error loading shoreline data, returning -1s for is_land flag")

            # if the shoreline data is not available
            # return the original label array

            return -np.ones_like(gdf.is_land.values)

    def clip_to_truth(self, truthpolypath):

        truth_polys = gpd.read_file(truthpolypath).set_crs('epsg:4326')

        pts_in = gpd.sjoin(self.data, truth_polys, predicate='within')

        self.data = pts_in

        return None

    @staticmethod
    def _clip_to_aoi(data, aoi):

        pts_idx = data.within(aoi.poly)

        return data.loc[pts_idx, :].copy()

    @staticmethod # can be accessed without an instance of the class
    def compress_ground_track(along_track_in, lat_in, lon_in,
                              polynomial_order=6, plot=False, profile=None):

        # note that the polynomial fit coefficients are also depending on along track
        # so we scale the along track distance to start 0 when fitting
        # total along track defines the stop point
        # can resample however you want within 0 and total_along_track_distance

        total_along_track_distance = np.int64(
            along_track_in.max() - along_track_in.min())

        along_track_fit = along_track_in - along_track_in.min()

        # fit a polynomial to the lat / lon data
        lon_coeffs = np.polyfit(along_track_fit, lon_in, deg=polynomial_order)
        lat_coeffs = np.polyfit(along_track_fit, lat_in, deg=polynomial_order)

        # store values you can use to sensibly reconstruct the track lat / lon
        cparams = {'total_along_track_distance': total_along_track_distance,
                   'lat_coeffs': lat_coeffs,
                   'lon_coeffs': lon_coeffs}

        if plot==True:

            if profile is None:
                raise ValueError("Must provide a profile to plot compression")
                
            # plot the original data and the reconstructed track
            plot_compression(profile, cparams)


        return cparams

    @staticmethod
    def reconstruct_track(cparams, along_track_sampling_meters=1000):

        lat_poly_object = np.poly1d(cparams['lat_coeffs'])
        lon_poly_object = np.poly1d(cparams['lon_coeffs'])

        x_reconstructed = np.linspace(0,
                                      cparams['total_along_track_distance'],
                                      along_track_sampling_meters)

        lat_reconstructed = lat_poly_object(x_reconstructed)
        lon_reconstructed = lon_poly_object(x_reconstructed)

        return lon_reconstructed, lat_reconstructed

    def write_database_entry(self, output_directory, polynomial_order=6):

        along_track_ph = self.data.dist_ph_along.values
        lat_ph = self.data.lat.values
        lon_ph = self.data.lon.values

        # compressing ground track includes visualization, but isnt really necessary
        # its included in the write database entry
        cparams = self.compress_ground_track(along_track_ph,
                                            lat_ph, lon_ph,                                            
                                            polynomial_order,
                                            plot=False) # required for plotting

        # put together path to output
        if output_directory is None:
            output_directory = os.getcwd()

        output_path = os.path.join(
            output_directory, self.get_formatted_filename() + '.json')

        # combine track parameters with track metadata
        # includes the absolute path to the h5 file
        output_dict = {**self.info, **cparams}

        # convert the date object to a string
        output_dict['date'] = output_dict['date'].strftime('%Y%m%d')

        with open(output_path, "w") as outfile:
            json.dump(output_dict, outfile, cls=NumpyEncoder,
            sort_keys=False, indent=4, separators=(',', ': '))

        return output_path

    def read_database_entry(filepath, verbose=False):

        with open(filepath) as json_file:
            data = json.load(json_file)

        if verbose:
            print(f'Reading database entry from...\n {filepath}')
            pp = pprint.PrettyPrinter()
            pp.pprint(data)

        return data

    @classmethod
    def from_database_entry(cls, filepath, AOI=None):

        data = cls.read_database_entry(filepath)

        # change this when its a class method, dont need profile
        profile = cls.from_h5(data['path'], data['gtxx'],
                     aoi=AOI, verbose=False)

        return profile

def plot_compression(profile, cparams, along_track_sampling_meters=1000):

    lon_reconstructed, lat_reconstructed = Profile.reconstruct_track(
        cparams, along_track_sampling_meters)

    # subsample by the nearest along track of the actual photon data for easier plotting
    data = profile.along_track_subsample(meters=20)

    f, ax = plt.subplots(1, 1, sharex=False, figsize=[7, 7])
    ax.set_title(profile.get_formatted_filename())
    ax.plot(data.lon, data.lat, 'b',
            label='Original Photon Data')

    ax.plot(lon_reconstructed, lat_reconstructed, 'r-.',
            label='Reconstructed Track')

    ax.legend()
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid()
    f.tight_layout()

    return f, ax

# encoder for handling numpy types in json
# via https://github.com/hmallen/numpyencoder/numpyencoder/numpyencoder.py
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)



if __name__ == "__main__":

    import sys
    import os
    import json
    import pprint
    import numpy as np
    import matplotlib.pyplot as plt

    sys.path.append('/Users/jonathan/Documents/Research/OpenOceans')
    import openoceans as oo

    # testing compression
    # I dont think / know if this will work over the poles or antimeridian
    # Inherently assumes that data are ordered by time
    # I dont know what the best order

    # now reconstruct track from cparams only
    # given compression parameters




    # this should be a class method



    # boilerplate script
    nc_aoi = oo.GeoAOI.from_geojson(
        os.path.abspath('demos/demo_data/pamlico.geojson'))
    h5_filepath = os.path.join(os.getcwd(
    ), 'sample_data', 'north_carolina_ATL03_20191024144104_04230506_005_01.h5')
    profile = oo.Profile.from_h5(
        h5_filepath, 'gt2r', aoi=nc_aoi, verbose=False)
    print(profile)
    print('done! :)')

    # TESTING DATABASES BELOW HERE

    # user inputs
    polynomial_order = 3  # recall + 1 for constant term
    along_track_sampling_meters = 100  # m

    along_track_ph = profile.data.dist_ph_along.values
    lat_ph = profile.data.lat.values
    lon_ph = profile.data.lon.values

    # compressing ground track includes visualization, but isnt really necessary
    # its included in the write database entry
    cparams = oo.Profile.compress_ground_track(along_track_ph,
                                              lat_ph, lon_ph,
                                              polynomial_order,
                                              plot=True, 
                                              profile=profile) # required for plotting

    # writes sample json database to demo data directory
    output_directory = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'demos', 'demo_data')

    path_to_entry = profile.write_database_entry(output_directory)

    # read database entry
    compressed_track_data = oo.Profile.read_database_entry(path_to_entry, verbose=True)

    profile2 = oo.Profile.from_database_entry(path_to_entry)
    profile_aoi = oo.Profile.from_database_entry(path_to_entry, AOI=nc_aoi)
    
    done = True

    # # Specifying h5 file path for importing ICESat-2 Data
    # # h5_filepath = "/Users/jonathan/Documents/Research/OpenOceans/demos/data/ATL03_20210817155409_08401208_005_01.h5"
    # # h5_filepath = '/Users/jonathan/Documents/Research/HTHH_SUMMER_2022/ATL03/ATL03_20220817200811_08691608_005_01.h5'

    # # Use open oceans Profile class to import data
    # # p = Profile.from_h5(h5_filepath, 'gt2r', verbose=True)

    # # Alternatively, load directly from the sample dataset
    # p = Profile.load_sample()

    # print('')
