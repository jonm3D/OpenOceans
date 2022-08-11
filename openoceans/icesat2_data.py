from cgitb import strong
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

from bokeh.plotting import figure
from bokeh.io import show, output_notebook
from bokeh.layouts import row
from bokeh.tile_providers import get_provider
tile_provider = get_provider('ESRI_IMAGERY')

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

        # details about the track (date, gtxx, rgt, cycle)
        self.info = info

        # photon resolution satellite data
        self.data = data

        # area of interest
        self.aoi = aoi

        # have a property for height datum and/or x,y crs?

    def __str__(self):
        '''Human-readable description.'''
        desc = f"""
TRACK DETAILS
    D/M/Y: {self.info['date'].strftime('%d/%m/%Y')}
    Reference Ground Track: {self.info['rgt']}
    Cycle: {str(self.info['cycle'])}
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

        # if verbose:
        #     print(h5_file_path)
        #     print('File size: {:.2f} GB'.format(os.path.getsize(h5_file_path) / 1e9))
        with progressbar.ProgressBar(max_value=7) as bar:

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
                #df['distance'] = np.array(f[gtxx + '/heights/dist_ph_along'])

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

                anc['release'] = np.array(
                    f['/ancillary_data/release'])[0].decode('UTF-8')

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

                df.set_index('time', inplace=True)

                df = df.loc[:, ['rgt', 'cycle', 'track', 'segment_id', 'segment_dist',
                                'sc_orient', 'atl03_cnf', 'height', 'quality_ph', 'delta_time', 'pair', 'geometry',
                            'ref_azimuth', 'ref_elev']]

                bar.update(7)

                df.insert(0, 'lat', df.geometry.y, False) 
                df.insert(0, 'lon', df.geometry.x, False)

                # wgs84 = pyproj.crs.CRS.from_epsg(4979)
                # wgs84_egm08 = pyproj.crs.CRS.from_epsg(3855)
                # tform = pyproj.transformer.Transformer.from_crs(crs_from=wgs84, crs_to=wgs84_egm08)
                # _, _, z_g = tform.transform(df.geometry.y, df.geometry.x, df.height)

                # df.insert(0, 'height_ortho', z_g, False) 

                # allocation of to be used arrays
                df.insert(0, 'classification', 
                          np.int64(np.zeros_like(df.geometry.x)), False)

                track_info_dict = {'date': df.index[0].date(),
                                   'rgt': anc['rgt'],
                                   'gtxx': gtxx,
                                   'cycle': anc['cycle_number'],
                                   'release': anc['release']}

                beam_info_dict = beam_info(gtxx, anc['sc_orient'])

                # combine the two
                info_dict = {**track_info_dict, **beam_info_dict}

        return cls(data=df, info=info_dict)

    def get_atl03_name_pattern(self):
        '''Returns a regex pattern matching the original ATL03 filename.'''
        date = self.info['date'].strftime('%Y%m%d')
        rgt = str(self.info['rgt']).zfill(4)
        cc = str(self.info['cycle']).zfill(2)
        rel = self.info['release']
        return "ATL03_{}{}_{}{}{}_{}".format(date, '*', rgt, cc, '*', rel)



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

    def to_kml(self):
        '''Export track metadata and lon/lat (no elevations) at x km along track to kml'''
        pass

    def to_csv(self):
        '''Export photon data to CSV'''
        pass

    def to_las(self):
        '''Export photon data to LAS 1.4'''
        pass

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
        # dropdown menu to select label for assigning photons


if __name__ == "__main__":
    from utils import _is_notebook
    h5_filepath = "/Users/jonathan/Documents/Research/OpenOceans_Public/demos/data/ATL03_20210817155409_08401208_005_01.h5"

    p = Profile.from_h5(h5_filepath, 'gt1r')
    print(p)
    p.explore()

else:
    from .utils import _is_notebook