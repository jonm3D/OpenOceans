from waveform import Waveform
from icesat2 import Profile
import numpy as np
import copy
import pandas as pd
from tqdm import tqdm
from fast_histogram import histogram2d, histogram1d
import matplotlib.pyplot as plt
from matplotlib import colors

# to do
# - convert waveforms to usable model dataframes - DONE
# - remove afterpulses by depth/ratio/saturation
# - add refraction corrections
# - skewed bathy waveforms
# - compare against validation depths
# basic DBSCAN function - 1 starting point
# functions for automatic parameter fitting


class ModelMaker:

    def __init__(self, res_along_track, res_z, range_z, window_size, step_along_track):
        """Create an instance with specified processing parameters at which to process profiles..

        Args:
            res_along_track (float): Along-track bin size which to histogram the photon data, in meters.
            res_z (float): Size of elevation bins with which to histogram photon data, in meters. Will be approximated as closely as possible to evenly fit the specified range_z, so recommended using nicely dividing decimals (like 0.1m).
            range_z (tuple): Total range of elevation values to bin, in meters. DEPTH OR ELEVATION
            window_size (float): How large of an along-track window to aggregate for constructing pseudowaveforms, in multiples of res_along_track. Recommended value of 1, must be odd.
            step_along_track (float): How many along track bins from step to step when constructing waveforms. Must be <= window_size.

        """
        assert step_along_track <= window_size
        assert np.mod(window_size, 2) == 1

        self.res_along_track = res_along_track
        self.res_z = res_z
        self.range_z = range_z
        self.window_size = window_size
        self.step_along_track = step_along_track

        # vertical bin sizing
        bin_count_z = np.ceil((self.range_z[1] - self.range_z[0]) / self.res_z)
        res_z_actual = np.abs(self.range_z[1] - self.range_z[0]) / bin_count_z

        if res_z_actual != self.res_z:
            print(f'Adjusting z_res - new value: {res_z_actual:.4f}')
            self.z_res = res_z_actual

    def __str__():
        return

    def process(self, profile):

        z_min = self.range_z[0]
        z_max = self.range_z[1]
        z_bin_count = np.int64(np.ceil((z_max - z_min) / self.res_z))

        # along track bin sizing
        at_min = profile.data.dist_ph_along.min()

        # calculating the range st the histogram output maintains exact along track values
        last_bin_offset = (self.res_along_track - np.mod((profile.data.dist_ph_along.max() -
                           profile.data.dist_ph_along.min()), self.res_along_track))

        at_max = profile.data.dist_ph_along.max() + last_bin_offset
        at_bin_count = np.int64((at_max - at_min) / self.res_along_track)

        xh = (profile.data.dist_ph_along.values)

        yh = (profile.data.height.values - profile.data.geoid_z.values)

        bin_edges_at = np.linspace(at_min, at_max, num=at_bin_count+1)
        bin_edges_z = np.linspace(z_min, z_max, num=z_bin_count+1)

        # disctionary of waveform objects - (AT_bin_i, w)
        w_d = {}

        # list of organized/simple data series'
        d_list = []

        # at the center of each evaluation window
        for i in tqdm(np.arange(self.step_along_track, len(bin_edges_at)-1, self.step_along_track)):
            # get indices/at distance of evaluation window
            i_beg = np.int64(max((i - (self.window_size-1) / 2), 0))
            i_end = np.int64(
                min((i + (self.window_size-1) / 2), len(bin_edges_at)-2))
            at_beg = bin_edges_at[i_beg]
            at_end = bin_edges_at[i_end]

            # could be sped up with numba if this is an issue
            # subset data using along track distance window
            i_cond = ((xh > at_beg) & (xh < at_end))
            df_win = profile.data.loc[i_cond, :]

            # i_window = np.arange( i_bot, i_top + 1, dtype=np.int64)

            # subset of histogram data in the evaluation window
            h = histogram1d((df_win.height.values - df_win.geoid_z.values),
                            range=[z_min, z_max], bins=z_bin_count)

            # identify median lat lon values of photon data in this chunk
            x_win = df_win.lon.median()
            y_win = df_win.lat.median()
            at_win = df_win.dist_ph_along.median()

            # not happy with having to flip this around but fine for now
            w = Waveform(np.flip(h), -np.flip(bin_edges_z))

            # this next section is basically all reformatting and organizing the data
            # might be better elsewhere but keeping here so i dont have to
            # run through another loop

            # we also want to have the original photon data for photons in this chunk
            # may also be helpful to have track metadata/functions
            p = copy.copy(profile)
            p.data = df_win
            w.add_profile(p)
            w.set_sat_ph_any((df_win.quality_ph > 0).any())

            # combine useful data into a nice dict for turning into a df
            extra_dict = {'i': i,  # indexing/location fields
                          'x_med': x_win,
                          'y_med': y_win,
                          'at_med': at_win,
                          'sat_ph_any': w.sat_ph_any,  # quality fields
                          'quality_flag': w.quality_flag,
                          'n_photons': df_win.shape[0]}

            # getting fitted params and renaming with distinct keys before combining dicts
            fitted_param_dict = {k+'_f': v for k,
                                 v in w.params.fitted.to_dict().items()}

            # combine into unified dict/df
            data_dict = dict(
                extra_dict, **w.params.initial.to_dict(), **fitted_param_dict)

            # add formatted series to the list to combine when done
            # might be faster if you preallocate the dataframe instead
            # but i dont want to have to constantly tweak the number
            # of rows if i add fields to this

            d_list.append(pd.Series(data_dict))

            # update dictionary of waveforms
            w_d.update({i: w})

        # summary of model data
        D = pd.DataFrame(d_list)

        m = Model(data=D, bin_edges_z=bin_edges_z,
                  bin_edges_at=bin_edges_at,
                  window_size=self.window_size,
                  step_along_track=self.step_along_track,
                  waves=w_d, profile=profile)

        return m

# maybe attach the model to the profile object


class Model:
    def __init__(self, data, bin_edges_z, bin_edges_at,
                 window_size, step_along_track,
                 waves, profile):
        # stores data for a model over a full profile

        self.data = data
        self.bin_edges_z = bin_edges_z
        self.bin_edges_at = bin_edges_at
        self.window_size = window_size
        self.step_along_track = step_along_track
        self.waves = waves
        self.profile = profile

        # slightly different than the model resolution
        # depending on step size, window
        self.hist = self._compute_histogram()

    def __str__(self):
        pass

    def _organize_wave_dict(self):
        # convert dict of waveforms to somethign more usable
        # has index to go back and reference individual waveforms

        # index
        # mean/median lat, lon, along track dist
        # model params
        # fitted model params
        # quality flag
        # has saturated photons or not
        # number of photons included

        pass

    def _compute_histogram(self):

        # backing histogram params from inputs
        xh = self.profile.data.dist_ph_along.values
        yh = (self.profile.data.height.values -
              self.profile.data.geoid_z.values)
        at_min = np.min(self.bin_edges_at)
        at_max = np.max(self.bin_edges_at)

        z_min = np.min(self.bin_edges_z)
        z_max = np.max(self.bin_edges_z)
        z_bin_count = len(self.bin_edges_z) - 1
        at_bin_count = len(self.bin_edges_at) - 1

        # compute fast histogram over large photon data
        h = histogram2d(xh, yh,
                        range=[[at_min, at_max], [
                            z_min, z_max]],
                        bins=[at_bin_count, z_bin_count])

        return h

    def show(self):

        x_axis_values = np.arange(len(self.bin_edges_at)) - 0.5

        pltnorm = colors.LogNorm(vmin=0.5, vmax=self.hist.max())

        # histogram of original photon data
        im = plt.pcolormesh(x_axis_values, self.bin_edges_z,
                            self.hist.T, norm=pltnorm)

        # model features
        qual_thresh = 1
        surf = plt.plot(self.data.i, -self.data.surf_loc, 'c.', label='surface')
        bathy = plt.plot(self.data.i[self.data.quality_flag == 3],
                         -self.data.bathy_loc[self.data.quality_flag == 3], 'gx', 
                         label='high_conf_bathy')
        
        low_bathy = plt.plot(self.data.i[self.data.quality_flag == 2],
                    -self.data.bathy_loc[self.data.quality_flag == 2], 'rx', 
                    label='low_conf_bathy')


        # formatting
        plt.xlabel('i')
        plt.ylabel('Elevation (m)')
        plt.title('Actual data hist - check this is oriented right wrt index')
        plt.legend()
        pass


if __name__ == "__main__":
    from waveform import Waveform
    from icesat2 import Profile
    import os

    h5_filepath = "/Users/jonathan/Documents/Research/OpenOceans/demos/data/ATL03_20210817155409_08401208_005_01.h5"
    p = Profile.from_h5(h5_filepath, 'gt1r')
    # note range z input is elevation, not depth
    M = ModelMaker(res_along_track=100, res_z=0.1, range_z=(-50, 20),
                   window_size=3, step_along_track=1)

    # check memory load without SD in bg
    m = M.process(p)
    m.show()
