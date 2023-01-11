from .waveform import Waveform
from .icesat2 import Profile
import numpy as np
import copy
import pandas as pd
from tqdm import tqdm
from fast_histogram import histogram2d, histogram1d
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.ndimage import gaussian_filter1d
import scipy.stats as stats
from scipy.signal import savgol_filter

# to do
# - add refraction corrections


def round_if_integer(x, n=2):
    if round(x, n) == x:
        return np.int64(x)
    return x


def cosine_similarity(x, y):
    """
    Compute the cosine similarity between two signals.

    Parameters:
    x (1D array): First signal.
    y (1D array): Second signal.

    Returns:
    float: Cosine similarity between x and y.
    """
    # Compute dot product and magnitudes
    dot = np.dot(x, y)
    x_mag = np.linalg.norm(x)
    y_mag = np.linalg.norm(y)

    # Compute and return cosine similarity
    return np.real(dot / (x_mag * y_mag))

# use a ttest to check if a set of values are meaningfully different from zero


def effectively_zero(values):

    if len(values) < 2:
        # not enough data to make a meaningful comparison
        return False

    t, p = stats.ttest_1samp(values, 0)

    return p > 0.05


class ModelMaker:

    def __init__(self, res_along_track, res_z, range_z, window_size, step_along_track, fit=False, verbose=False):
        """Create an instance with specified processing parameters at which to process profiles..

        Args:
            res_along_track (float): Along-track bin size which to histogram the photon data, in meters.
            res_z (float): Size of elevation bins with which to histogram photon data, in meters. Will be approximated as closely as possible to evenly fit the specified range_z, so recommended using nicely dividing decimals (like 0.1m).
            range_z (tuple): Total range of elevation values to bin, in meters. DEPTH OR ELEVATION
            window_size (float): How large of an along-track window to aggregate for constructing pseudowaveforms, in multiples of res_along_track. Recommended value of 1, must be odd.
            step_along_track (float): How many along track bins from step to step when constructing waveforms. Must be <= window_size.

        """
        assert step_along_track <= window_size  # prevents gaps in data coverage
        assert step_along_track > 0
        assert window_size > 0
        # odd for symmetry about center bins
        assert np.mod(window_size, 2) == 1

        self.res_along_track = res_along_track
        self.res_z = res_z
        self.range_z = range_z
        self.window_size = window_size
        self.step_along_track = step_along_track
        self.fit = fit
        self.verbose = verbose

        # vertical bin sizing
        bin_count_z = np.ceil((self.range_z[1] - self.range_z[0]) / self.res_z)
        res_z_actual = np.abs(self.range_z[1] - self.range_z[0]) / bin_count_z

        if res_z_actual != self.res_z:
            print(f'Adjusting z_res - new value: {res_z_actual:.4f}')
            self.z_res = res_z_actual

    def __str__():
        return

    def get_formatted_filename(self):

        res_at = round_if_integer(self.res_along_track)
        res_z = self.res_z
        range_z_0 = -round_if_integer(self.range_z[0])
        range_z_1 = round_if_integer(self.range_z[1])
        win = self.window_size
        step_at = self.step_along_track

        # programmatically create string to concatenate self attributes res_along_track, res_z, range_z, window_size, step_along_track
        precision = 2
        model_id = f'{res_at:.0f}_\
{res_z:.2f}_{range_z_0:.0f}_\
{range_z_1:.0f}_{win:.0f}_\
{step_at:.0f}'

        return model_id

    def process(self, profile):

        z_min = self.range_z[0]
        z_max = self.range_z[1]
        z_bin_count = np.int64(np.ceil((z_max - z_min) / self.res_z))

        # if signal has been identified, only use those photons
        if profile.signal_finding == True:
            data = profile.data.loc[profile.data.signal == True, :]

        else:
            data = profile.data

        # along track bin sizing
        at_min = data.dist_ph_along.min()

        # calculating the range st the histogram output maintains exact along track values
        last_bin_offset = (self.res_along_track - np.mod((data.dist_ph_along.max() -
                           data.dist_ph_along.min()), self.res_along_track))

        at_max = data.dist_ph_along.max() + last_bin_offset
        at_bin_count = np.int64((at_max - at_min) / self.res_along_track)

        xh = (data.dist_ph_along.values)

        yh = (data.height.values - data.geoid_z.values)

        bin_edges_at = np.linspace(at_min, at_max, num=at_bin_count+1)
        bin_edges_z = np.linspace(z_min, z_max, num=z_bin_count+1)

        # disctionary of waveform objects - (AT_bin_i, w)
        w_d = {}

        # list of organized/simple data series'
        d_list = []

        # array to store actual interpolated model and fitted model
        hist_modeled = np.nan * np.zeros((at_bin_count, z_bin_count))
        hist_modeled_fit = np.nan * np.zeros((at_bin_count, z_bin_count))

        # at the center of each evaluation window
        # win centers needs to actually handle the center values unlike here
        # currently bugs out if win is 1

        # this step ensures that histograms at edges dont have lower 'intensity' just becasue the window exceeds the data range
        start_step = (self.window_size-1) / 2
        end_step = len(bin_edges_at) - (self.window_size-1) / 2 - 1

        win_centers = np.arange(np.int64(start_step), np.int64(
            end_step), self.step_along_track)

        for i in tqdm(win_centers):

            # get indices/at distance of evaluation window
            i_beg = np.int64(max((i - (self.window_size-1) / 2), 0))
            i_end = np.int64(
                min((i + (self.window_size-1) / 2), len(bin_edges_at)-2)) + 1
            # + 1 above pushes i_end to include up to the edge of the bin when used as index

            at_beg = bin_edges_at[i_beg]
            at_end = bin_edges_at[i_end]

            # could be sped up with numba if this is an issue
            # subset data using along track distance window
            i_cond = ((xh > at_beg) & (xh < at_end))
            df_win = data.loc[i_cond, :]

            # version of dataframe with only nominal photons
            # use this data for constructing waveforms
            df_win_nom = df_win.loc[df_win.quality_ph == 0]
            height_geoid = (df_win_nom.height.values -
                            df_win_nom.geoid_z.values)

            # subset of histogram data in the evaluation window
            h_ = histogram1d(height_geoid,
                             range=[z_min, z_max], bins=z_bin_count)

            # smooth the histogram with 0.2 sigma gaussian kernel
            h = gaussian_filter1d(h_, 0.2/self.res_z)

            # identify median lat lon values of photon data in this chunk
            x_win = df_win_nom.lon.median()
            y_win = df_win_nom.lat.median()
            at_win = df_win_nom.dist_ph_along.median()

            # not happy with having to flip this around but fine for now
            any_sat_ph = (df_win.quality_ph > 0).any()

            w = Waveform(np.flip(h), -np.flip(bin_edges_z),
                         fit=self.fit, sat_check=any_sat_ph)

            #################################################################

            # this next section is basically all reformatting and organizing the data
            # might be better elsewhere but keeping here so i dont have to
            # run through another loop

            # we also want to have the original photon data for photons in this chunk
            # may also be helpful to have track metadata/functions
            # this is for ALL photons, not just the nominal subset
            p = copy.copy(profile)
            p.data = df_win
            w.add_profile(p)

            # combine useful data into a nice dict for turning into a df
            extra_dict = {'i': i,  # indexing/location fields
                          'x_med': x_win,
                          'y_med': y_win,
                          'at_med': at_win,
                          'at_range': (at_beg, at_end),
                          'quality_flag': w.quality_flag,
                          'bathy_conf': w.bathy_quality_ratio,
                          'water_conf': -999,
                          'sat_flag': w.sat_flag,
                          'sat_check': w.sat_check,
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

            # testing
            # use -1 instead of nan in order to be able to ffill sparse matrix
            # -1 indicates we checked this chunk and it actually is empty, not just
            # that is got skipped by the step sizes
            if np.all(np.isnan(w.model.output)):
                w.model.output = -1 * np.ones_like(w.model.output)

            if np.all(np.isnan(w.model.output_f)):
                w.model.output_f = -1 * np.ones_like(w.model.output_f)

            hist_modeled[i, :] = np.flip(w.model.output)
            hist_modeled_fit[i, :] = np.flip(w.model.output_f)

        # summary of model params
        params = pd.DataFrame(d_list)

        m = Model(params=params, model_hist=hist_modeled, model_hist_fit=hist_modeled_fit,
                  bin_edges_z=bin_edges_z,
                  bin_edges_at=bin_edges_at,
                  window_size=self.window_size,
                  step_along_track=self.step_along_track,
                  waves=w_d, profile=profile, fitted=self.fit)

        return m

# maybe attach the model to the profile object


class Model:
    # more processing after the histgram model has been estimated

    def __init__(self, params, model_hist, model_hist_fit, bin_edges_z, bin_edges_at,
                 window_size, step_along_track,
                 waves, profile, fitted):
        # stores data for a model over a full profile

        self.params = params
        self.model_hist = model_hist
        self.model_hist_fit = model_hist_fit
        self.bin_edges_z = bin_edges_z
        self.bin_edges_at = bin_edges_at
        self.res_z = np.diff(bin_edges_z)[0]
        self.res_at = np.diff(bin_edges_at)[0]
        self.window_size = window_size
        self.step_along_track = step_along_track
        self.waves = waves
        self.profile = profile
        self.fitted = fitted

        # slightly different than the model resolution
        # depending on step size, window
        self.hist = self._compute_histogram()
        self.model_hist_interp = self._interp_model_hist(self.model_hist)
        self.model_hist_fit_interp = self._interp_model_hist(
            self.model_hist_fit)

        # compute turbidity score and append to model data
        turbidity_score = self._compute_turbidity_score()
        self.params.loc[:, 'turb_score'] = turbidity_score

        # compute labels for photon data and update m.profile.data.classifications
        self.profile.data['classification'] = self.label_by_pseudowaveform()

        # will probably want a second classification array for post filtering results later
        # do that here

        # PROCESSING SURFACE MODEL / OCEAN FLAGGING / MORE
        self.surface_data = self.process_surface_data(along_track_sight_distance_flat=1000,
                                                      both_window_threshold=0.25,
                                                      one_window_threshold=0.5,
                                                      along_track_sight_distance_spectra=1000,  # betting this does better lower
                                                      spectral_similarity_threshold=0.95)  # processed surface data

    # def process_turbidity_data(self):

    #     return turbidity_data

    def process_surface_data(self, along_track_sight_distance_flat=2000,
                             both_window_threshold=0.25,
                             one_window_threshold=0.5,
                             along_track_sight_distance_spectra=1000,
                             spectral_similarity_threshold=0.95):

        # main function that corrals all the surface modeling functions

        # compute smoothed surface
        x_smoothed, z_smoothed = self._smooth_surface(
            self.profile.data, smoothing_aggression=0.01, surface_class=41)

        # start constructing surface object and add to object
        # specific flags get added to this structure
        data_rows = [x_smoothed, z_smoothed]
        data_names = ['dist_ph_along', 'z', ]
        surface_data = pd.DataFrame(data_rows,
                                    index=data_names).T

        # remove duplicate along track points and get median elevations for these
        # easier now that we have a dataframe whoo boy i love dataframes
        surface_data = surface_data.groupby(
            'dist_ph_along').median().reset_index()

        flatness_score, rel_elev_data = self._compute_flatness_score(
            surface_data.dist_ph_along.values,
            surface_data.z.values,
            along_track_sight_distance_flat,
            both_window_threshold,
            one_window_threshold)

        surface_data['flatness_score'] = flatness_score
        surface_data = pd.concat([surface_data, rel_elev_data], axis=1)

        # append flatness score and relative elevation

        # use waviness to determine likely ocean

        similar_spectra_flag, spectral_similarity_score = self._compute_spectral_ocean_flag(
            surface_data.dist_ph_along.values,
            surface_data.z.values,
            along_track_sight_distance_spectra,
            spectral_similarity_threshold)

        surface_data['similar_spectra_flag'] = similar_spectra_flag

        surface_data['spectral_similarity_score'] = spectral_similarity_score

        return surface_data

    def _compute_flatness_score(self, surface_along_track, surface_z, along_track_sight_distance=2000, both_window_threshold=0.25, one_window_threshold=0.5):
        # both window threshold means how far apart are the average surface values on either side of you
        # one window threshold means how far is the average of each side from the photon at the center of the window

        # this is sloowwwwwwwwwww...

        # FLAT SURFACE SCORE
        # 0 = not flat
        # 1 = flat
        # 2 = buffer / coastal zone

        # RELATIVE ELEVATION DATA
        # rows are observations of surface model points
        # columns are 'looking_ahead' and 'looking_behind'
        # values are the difference between the average surface elevation of the points ahead or behind the observation point
        # and the observation points elevation
        # first val of looking ahead and last value of looking behind are NAN!

        looking_ahead_relelev = self._diff_from_nearby_average(surface_along_track, surface_z,
                                                               along_track_direction='ahead',
                                                               along_track_sight_distance=along_track_sight_distance)

        looking_behind_relelev = self._diff_from_nearby_average(surface_along_track, surface_z,
                                                                along_track_direction='behind',
                                                                along_track_sight_distance=along_track_sight_distance)

        # combine the two
        rel_elev_data = pd.DataFrame([looking_ahead_relelev, looking_behind_relelev],
                                     index=['looking_ahead', 'looking_behind']).T

        # require that both the surfaces on either side be relatively close to eachother
        # i.e. \./ or _._
        flat_condition_1 = np.abs(
            looking_ahead_relelev - looking_behind_relelev) < both_window_threshold

        # ensure that neither surface on either side changes elevation very much
        # m threshold to limit \./ false positive detection
        flat_condition_2 = (np.abs(looking_ahead_relelev) < one_window_threshold) \
            & (np.abs(looking_behind_relelev) < one_window_threshold)

        # WHY THIS WORKS
        # the first condition really only works because with either sides surface elevation averaged,
        # then the wave motion is captured by the center photon, and the surface elevation changes symmetrically
        # this makes most sense if you plot the mean relative height of the window on either side along track
        # then the second condition just helps filter out chance occurances over land
        # buffer still needs to be applied to this method

        # anywhere both these conditions are met is high conf flat surface
        flat_surface_bool = np.logical_and(flat_condition_1, flat_condition_2)

        flat_surface_score = flat_surface_bool.astype(
            np.int64) * 2  # label these with class 2 - high conf

        # buffer size is automatically determined by the look ahead distance

        # get indices where flat_surface_bool changes values
        buffer_idx = np.where(np.diff(flat_surface_bool))[0]

        for i_buff in range(len(buffer_idx)):
            # get info about this point along track
            i = buffer_idx[i_buff]
            i_x = surface_along_track[i]
            i_z = surface_z[i]

            # collect all nearby photons
            left_edge = i_x - along_track_sight_distance
            right_edge = i_x + along_track_sight_distance
            photons_in_window = (surface_along_track > left_edge) & (
                surface_along_track < right_edge)

            # but only photons that werent already classified as surface
            unclass_photons_in_window = photons_in_window & (
                flat_surface_bool == False)

            # give these photons a flat surface score of 1
            flat_surface_score[unclass_photons_in_window] = 1

        return flat_surface_score, rel_elev_data

    def _diff_from_nearby_average(self, x_along_track, y_surface, along_track_direction, along_track_sight_distance=100):
        # require x_along_track and y_surface to just be 1d numpy arrays

      # compute the flatness and spectral similarity for each point along track
        # require along track direction string is 'ahead' or 'behind'
        assert along_track_direction in ['ahead', 'behind']

        # storage array
        diff_from_nearby_average = np.zeros_like(x_along_track)

        for i in range(len(y_surface)):
            i_x = x_along_track[i]
            i_y = y_surface[i]

            # (center bin, look ahead/behind distance)
            if along_track_direction == 'ahead':
                at_range = (i_x, i_x + along_track_sight_distance)
            elif along_track_direction == 'behind':
                at_range = (i_x - along_track_sight_distance, i_x)

            # get all photons ahead/behind you
            photons_in_window = (x_along_track > at_range[0]) \
                & (x_along_track < at_range[1])

            diff_from_nearby_average[i] = np.mean(
                y_surface[photons_in_window] - i_y)

        return diff_from_nearby_average

    def _compute_spectral_ocean_flag(self, x_along_track, z_surface, along_track_sight_distance=2000, spectral_similarity_threshold=0.95):

        # compute the flatness and spectral similarity for each point along track
        # x_along_track = self.surface_data.dist_ph_along.values
        # z_surface = self.surface_data.z.values

        # storage array
        this_photon_is_over_water = np.zeros_like(x_along_track)
        not_yet_checked = np.ones_like(x_along_track)
        not_yet_classified = np.ones_like(x_along_track)
        spectral_similarity_score = -np.ones_like(x_along_track)

        for i in range(len(z_surface)):  # standard loop through surface photons

            # loop through only unprocessed/classified photons
            # assignes classes in groups where possible to speed up compute

            # while np.any(np.logical_and(not_yet_checked, not_yet_classified)):
            #     i = np.where(np.logical_and(
            #         not_yet_checked, not_yet_classified))[0][0]

            # get this surface models along track and elevation value
            i_x = x_along_track[i]

            not_yet_checked[i] = 0  # update to do list

            # (center bin, look ahead distance)
            at_range_ahead = (i_x, i_x + along_track_sight_distance)

            # (center bin, look behind distance)
            at_range_behind = (i_x - along_track_sight_distance, i_x)

            # get all photons ahead/behind you
            photons_ahead = (x_along_track > at_range_ahead[0]) \
                & (x_along_track < at_range_ahead[1])

            photons_behind = (x_along_track > at_range_behind[0]) \
                & (x_along_track < at_range_behind[1])

            # make these surface the same length of data, whichever is shorter
            trim_length = min(np.sum(photons_ahead), np.sum(photons_behind))

            if trim_length < 50:
                # require 50 photons, else skip
                # totally arbitrary just a debugging sanity check
                continue

            # isolate same length surface signal chunks
            signal_ahead = z_surface[photons_ahead][:trim_length]
            signal_behind = z_surface[photons_behind][-trim_length:]

            # compute spectra of smoothed surface model
            spectra_ahead = np.fft.fft(signal_ahead)
            spectra_behind = np.fft.fft(signal_behind)

            # compare spectra - ocean waves more likely to have consistent frequencies
            spectral_similarity_score[i] = cosine_similarity(
                spectra_ahead, spectra_behind)

            # compare similarity to threshold
            likely_ocean_by_spectra = spectral_similarity_score[i] > spectral_similarity_threshold

            # if deemed likely ocean, raise flag for all photons in the window to reduce compute
            if likely_ocean_by_spectra:
                this_photon_is_over_water[photons_ahead] = 1
                this_photon_is_over_water[photons_behind] = 1

                this_photon_is_over_water[i] = 1

                not_yet_classified[photons_ahead] = 0  # update to do list
                not_yet_classified[photons_behind] = 0

            # if not deemed likely ocean, move on to next photon
            # recall photons are presumed land to start, requires positive verification to switch to ocean

            # move next unclassified, not-yet-analyzed photon
            # should be a significant computational savings over the pure loop

            # save spectral similarity

            # add buffer to spectral similarity score

        return this_photon_is_over_water.astype(bool), spectral_similarity_score

    def _smooth_surface(self, profile_data, smoothing_aggression=0.025, surface_class=41):

        # get surface classified photons
        surface_data = profile_data.loc[profile_data.classification == surface_class]

        # sort by along track distance
        surface_data = surface_data.sort_values('dist_ph_along')
        z = surface_data.height - surface_data.geoid_z
        x_along_track = surface_data.dist_ph_along.values

        # smoothing with savitsky golay
        # window size set by number of photons
        # theres probably a more consistent way to do this but i think fine for now
        poly_order = 3
        window_length = np.int64(len(z) * smoothing_aggression)

        z_smooth = savgol_filter(z, window_length, poly_order)

        return x_along_track, z_smooth

    def __str__(self):

        # print out useful info about this model like params, the ranges of the z , AT values, win, step size, num waveforms

        pass

    def _compute_turbidity_score(self):

        # if window and step size are both 1, dont throw an error
        if (self.window_size != 1) or (self.step_along_track != 1):
            raise ValueError('Turbidity score can only be computed for window size and step size of 1. This should be fixed at some point, but its not immediately clear how or why to handle this score for overlapping window cases.')

        # index via wave keys or something else?
        n = len(self.waves.keys())
        turbidity_score = np.zeros((n, ))

        for i in range(n):

            # sum of turbidity component
            # turbidity_score[i] = np.sum(self.waves[i].model.turbidity)

            # would need to be made into a true integral to be consistent regardless of bin resolution
            turbidity_score[i] = np.trapz(self.waves[i].model.turbidity)

        return turbidity_score

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
        h_ = histogram2d(xh, yh,
                         range=[[at_min, at_max], [
                             z_min, z_max]],
                         bins=[at_bin_count, z_bin_count])

        g_sigma = 0.2 / np.diff(self.bin_edges_z)[0]

        h = gaussian_filter1d(h_, g_sigma, axis=1)

        return h

    def _interp_model_hist(self, hist):

        # move this to the actual model section

        # interpolating between steps of data
        hist_interp = pd.DataFrame(hist).interpolate(
            method='linear', axis=0, limit=None, inplace=False, limit_area='inside')

        # using the first/last steps to fill the edges
        hist_interp.iloc[:self.window_size] = hist_interp.iloc[:self.window_size].fillna(
            method='backfill')
        hist_interp.iloc[-self.window_size:] = hist_interp.iloc[-self.window_size:].fillna(
            method='ffill')

        return hist_interp

    def show(self, interp=True):

        # will show the fitted model if it has been calculated

        x_axis_values = np.arange(len(self.bin_edges_at)) - 0.5

        pltnorm = colors.LogNorm(vmin=1, vmax=self.hist.max())

        f, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8, 6))

        # histogram of original photon data
        im = ax[0].pcolormesh(x_axis_values, self.bin_edges_z,
                              self.hist.T, norm=pltnorm)
        cmap = im.get_cmap()
        ax[0].set_title(self.profile.get_formatted_filename())
        ax[0].set_ylabel('Elevation (m)')

        # histogram of model data
        # plt.plot()

        if self.fitted == True and interp == True:
            histm = self.model_hist_fit_interp
        elif self.fitted == False and interp == True:
            histm = self.model_hist_interp
        elif self.fitted == True and interp == False:
            histm = self.model_hist_fit
        elif self.fitted == False and interp == False:
            histm = self.model_hist

        histm[histm < 1] = 0

        im1 = ax[1].pcolormesh(x_axis_values, self.bin_edges_z,
                               histm.T, norm=pltnorm, cmap=cmap)
        # get cmap from this image

        ax[1].set_title(
            'modeled data')
        ax[1].set_ylabel('Elevation (m)')
        ax[1].set_xlabel('Along Track Step Index')

        # Descriptive title for the model based on model parameters for uniqueness
        # assume constant along track resolution
        at_res = np.diff(self.bin_edges_at)[0]
        # assume constant vertical bin resolution
        z_res = np.diff(self.bin_edges_z)[0]
        z_range = (round_if_integer(np.min(self.bin_edges_z)),
                   round_if_integer(np.max(self.bin_edges_z)))
        ax[1].set_title(
            f'AT_res={round_if_integer(at_res)}m, Z_res={z_res}m, Z_range=({z_range[0]}, {z_range[1]})m, WIN={self.window_size}, STEP={self.step_along_track}')

        # main title
        f.suptitle('Pseudowaveform-based Model')

        # adjusting subplot to make space for colorbar
        bottom, top = 0.1, 0.9
        left, right = 0.1, 0.8

        f.subplots_adjust(top=top, bottom=bottom, left=left,
                          right=right, hspace=0.15, wspace=0.25)

        # colorbar
        cbar_ax = f.add_axes([0.85, bottom, 0.05, top-bottom])
        f.colorbar(im, cax=cbar_ax, cmap=cmap, label='Num. of Photons Per Bin')

        return f, ax

    def compare_fit(self):

        # compare the stat model vs the

        pass

    def report(self, ModelMakerInit):

        along_track_delta = self.params.at_med - self.profile.data.dist_ph_along.min()

        f, ax = plt.subplots(5, 1, figsize=(8, 11), sharex=True)

        ax[0].plot(self.profile.data.dist_ph_along - self.profile.data.dist_ph_along.min(),
                 self.profile.data.height - self.profile.data.geoid_z,
                   'k.', alpha=0.4, markersize=0.25, label='Photon Data')
        ax[0].set_ylim(-self.params.bathy_loc.max() - 2, 20)
        ax[0].plot(along_track_delta, -self.params.surf_loc,
                   'b', label='Estimated Surface (m)')
        ax[0].legend()
        ax[0].grid()
        ax[0].set_ylabel('EGM08 Elevation (m)')
        ax[0].set_ylim([min(-self.params.surf_loc) - 5,
                       max(-self.params.surf_loc) + 10])

        ax[1].plot(along_track_delta, self.params.surf_std, 'r')
        ax[1].set_title('Surface Deviation')
        ax[1].set_ylabel('$\sigma$ (m)')
        ax[1].grid()

        ax[2].plot(along_track_delta, self.params.surf_prom, 'g')
        ax[2].set_ylabel('Photons at peak')
        ax[2].set_title('Surface Prominence')
        ax[2].grid()

        ax[3].plot(along_track_delta, self.params.bathy_prom /
                   self.params.surf_prom, 'm', label='Bathy/Surface Prominence')
        ax[3].fill_between(along_track_delta, 0.5, 1.1, color='g', alpha=0.2)
        ax[3].fill_between(along_track_delta, 0.25, 0.5, color='y', alpha=0.2)
        ax[3].fill_between(along_track_delta, 0.0, 0.25, color='r', alpha=0.2)

        ax[3].set_ylim([-0.1, 1.1])
        ax[3].set_title('Bathymetry Confidence TBD')
        # ax[3].grid()

        ax[4].plot(along_track_delta, self.params.turb_score, 'k')
        ax[4].set_title('Turbidity Score')
        ax[4].set_ylabel('photon-meters')
        ax[4].grid()
        ax[4].set_xlabel('Along Track Delta (m)')

        f.suptitle(ModelMakerInit.get_formatted_filename() +
                   ' - ' + self.profile.get_formatted_filename().upper())
        f.tight_layout()

        return f, ax

    def classify_land(self, window_size=21):

        # window size is total sliding windo in bins
        # ie, the number of bins in the window to a given side of the center bin is as follows
        # require window size to be odd
        assert window_size % 2 == 1, 'window size must be odd'

        win_half = int((window_size - 1) / 2)

        likely_land = np.full((self.params.shape[0], ), False, dtype=bool)

        for i_along_track in range(self.params.shape[0]):

            # left sided value
            if i_along_track - win_half < 0:
                left = 0
            else:
                left = i_along_track - win_half

            # right sided value
            if i_along_track + win_half > self.params.shape[0]:
                right = self.params.shape[0]
            else:
                right = i_along_track + win_half

            # collect values to some distance either side
            params_L = self.params.iloc[left:i_along_track, :]
            params_R = self.params.iloc[i_along_track:right, :]
            params_i = self.params.iloc[i_along_track, :]

            # get relative surface elevations to either side of central bin
            relative_surface_heights_left = params_L.surf_loc - params_i.surf_loc
            relative_surface_heights_right = params_R.surf_loc - params_i.surf_loc

            # check if data on either given side is 'flat'
            if effectively_zero(relative_surface_heights_left) \
                    or effectively_zero(relative_surface_heights_right):

                # likely_land[i_along_track] = False # already false
                pass

            else:
                # neither side is effectively very flat
                likely_land[i_along_track] = True

            # we can also apply a sanity check here to some of the model parameters
            # theyre allowed to vary a bit outside of normal expected values for stabilities sake
            # but if they're way off, we can flag them as likely land

            # if params_i.surf_std > 5.0:
            #     self.params.loc[i_along_track, 'likely_land'] = True

        self.likely_land = likely_land

        return likely_land

    def plot_land_classification(self):

        f, ax = plt.subplots(1, 1, figsize=(8, 5))

        x = self.profile.data.dist_ph_along
        z = self.profile.data.height - self.profile.data.geoid_z

        ax.plot(x, z, 'k.', label='All photons')

        # plot land as green background shading, water as blue shading

        for i in np.arange(len(self.bin_edges_at) - 1) + 1:

            # shift index by 1 to get correct bin edges

            if self.likely_land[i]:

                ax.fill_betweenx(z, x[i-1], x[i], color='g', alpha=0.2)

            else:
                ax.fill_betweenx(z, x[i-1], x[i], color='b', alpha=0.2)

                # ax.axvspan(self.params.loc[i, 'at_med'] - self.params.loc[i, 'at_std'],
                #             self.params.loc[i, 'at_med'] + self.params.loc[i, 'at_std'],
                #             color='b', alpha=0.2)

        ax.plot(self.params.at_med, self.params.likely_land, 'k.')
        ax.set_xlabel('Along Track Delta (m)')
        ax.set_ylabel('Likely Land?')
        ax.set_title('Land Classification')

        return f, ax

    def label_by_pseudowaveform(self):

        # set up array of labels using original photon indices
        labels = pd.DataFrame(-np.ones((self.profile.data.shape[0],), dtype=int),
                              index=self.profile.data.index)

        for i_along_track in self.waves.keys():

            # get the wave object and estimated model parameters for this bin
            i_wave = self.waves[i_along_track]
            i_params = i_wave.params.initial

            # getting useful elevation values from the model estimate
            # converting to standard elevation
            i_surf_z = (-i_params.surf_loc)
            i_bathy_z = (-i_params.bathy_loc)

            # get model estimates for the following elevation values:

            # elevation 3 sigma above the surface
            thresh_surf_top = i_surf_z + (3 * i_params.surf_std)

            # transition point from surface to column
            thresh_column_top = i_surf_z - i_params.trans_mult * i_params.surf_std

            # set boundaries for generally gaussian bathymetric surface returns
            thresh_bathy_bottom = i_bathy_z - (3 * i_params.bathy_std)
            thresh_bathy_top = i_bathy_z + (3 * i_params.bathy_std)

            # get profile.data indices of photons in this along track bin
            idx_in_this_bin = i_wave.profile.data.index

            # index photons in this wave bin and classify them by height

            # above surface
            i_background_idx = (i_wave.profile.data.height -
                                i_wave.profile.data.geoid_z) > thresh_surf_top

            # in surface return
            i_surface_idx = ((i_wave.profile.data.height-i_wave.profile.data.geoid_z) <= thresh_surf_top) & \
                            ((i_wave.profile.data.height -
                             i_wave.profile.data.geoid_z) > thresh_column_top)

            # in water column
            i_column_idx = ((i_wave.profile.data.height-i_wave.profile.data.geoid_z) <= thresh_column_top) & \
                ((i_wave.profile.data.height-i_wave.profile.data.geoid_z)
                 > thresh_bathy_top)

            # in bathy return
            i_bathy_idx = ((i_wave.profile.data.height-i_wave.profile.data.geoid_z) <= thresh_bathy_top) & \
                ((i_wave.profile.data.height-i_wave.profile.data.geoid_z)
                 > thresh_bathy_bottom)

            # below any bathy return
            i_noise_idx = (i_wave.profile.data.height -
                           i_wave.profile.data.geoid_z) < thresh_bathy_bottom

            # assign labels to photons
            labels.loc[idx_in_this_bin[i_background_idx]] = 1
            # LAS specification
            labels.loc[idx_in_this_bin[i_surface_idx]] = 41
            labels.loc[idx_in_this_bin[i_column_idx]] = 45  # LAS specification
            labels.loc[idx_in_this_bin[i_bathy_idx]] = 40  # LAS specification
            labels.loc[idx_in_this_bin[i_noise_idx]] = 2

        return labels  # also output them for convenience


if __name__ == "__main__":
    from waveform import Waveform
    from icesat2 import Profile
    import os

    # note range z input is elevation, not depth
    M = ModelMaker(res_along_track=50, res_z=0.25, range_z=(-50, 20),
                   window_size=1, step_along_track=1, fit=False)

    h5_filepath = "/Users/jonathan/Documents/Research/OpenOceans/demos/data/ATL03_20210817155409_08401208_005_01.h5"
    h5_filepath = '/Users/jonathan/Documents/Research/PAPER_Density_Heuristic_IS2_Bathy/nc_is2_2019/ATL03_20191206003613_10710502_005_01.h5'

    p = Profile.from_h5(h5_filepath, 'gt1r')
    p.clip_to_ocean()

    m = M.process(p)

    # self = m
    m.show()

    plt.figure()
    thr = 10
    okidx = m.params.bathy_conf > thr
    plt.plot(p.data.lat, p.data.height-p.data.geoid_z,
             'k.', markersize=0.5, alpha=.8)
    plt.plot(m.params.y_med[okidx], -m.params.bathy_loc[okidx], 'r.')
