import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, peak_widths, peak_prominences, find_peaks
from scipy.stats import exponnorm, norm, skewnorm, expon
import warnings
import pandas as pd


def find_transition_point(turb_intens, surf_prom, surf_loc, surf_std):
    transition_value = turb_intens

    # debugging aid
    if np.log(transition_value / (surf_prom)) * (-2*surf_std**2) < 0:
        print(transition_value, surf_prom, surf_std)

    transition_point = surf_loc \
        + np.sqrt(np.log(transition_value / (surf_prom))
                  * (-2*surf_std**2))

    return transition_point

# temporary breakout for plotting/debugging


def noise(depth_bins, surf_prom, surf_loc, surf_std,
          decay_param, turb_intens,
          noise_above, noise_below,
          bathy_prom, bathy_loc, bathy_std):

    transition_point = find_transition_point(
        turb_intens, surf_prom, surf_loc, surf_std)

    # add base noise rates above and below the water surface
    # ramp between two constant values on either side of the gaussian
    bound_l = surf_loc - 3*surf_std
    bound_r = transition_point

    return np.interp(depth_bins, [bound_l, bound_r], [noise_above, noise_below])


def bathy(depth_bins, surf_prom, surf_loc, surf_std,
          decay_param, turb_intens,
          noise_above, noise_below,
          bathy_prom, bathy_loc, bathy_std):

    return (bathy_prom * (np.sqrt(2 * np.pi) * bathy_std)) \
        * norm.pdf(depth_bins, bathy_loc, bathy_std)


def surface(depth_bins, surf_prom, surf_loc, surf_std,
            decay_param, turb_intens,
            noise_above, noise_below,
            bathy_prom, bathy_loc, bathy_std):

    # TRANSITION point from gaussian surface to exponential model
    # the max intensity of the exponential curve should match the value of the gaussian model at the transition
    transition_point = find_transition_point(
        turb_intens, surf_prom, surf_loc, surf_std)

    # SURFACE GAUSSIAN peak values
    y_out = np.zeros_like(depth_bins)

    y_out[depth_bins < transition_point] = (surf_prom * (np.sqrt(2 * np.pi) * surf_std)) \
        * norm.pdf(depth_bins[depth_bins < transition_point], surf_loc, surf_std)

    return y_out


def turbidity(depth_bins, surf_prom, surf_loc, surf_std,
              decay_param, turb_intens,
              noise_above, noise_below,
              bathy_prom, bathy_loc, bathy_std):

    # depth_bins is the array of elevations

    # TRANSITION point from gaussian surface to exponential model
    # the max intensity of the exponential curve should match the value of the gaussian model at the transition
    transition_point = find_transition_point(
        turb_intens, surf_prom, surf_loc, surf_std)

    # EXPONENTIAL decay values
    # model depth values are measured from the surface peak location
    # exponential curve begins where the surface peak value matches the input turbid intensity
    # decay parameter is mapped to depth in m of the bins supplied by counts
    y_out = np.zeros_like(depth_bins)

    z_depth = depth_bins[depth_bins >= transition_point] - surf_loc

    y_out[depth_bins >= transition_point] = (turb_intens / np.exp(decay_param * (transition_point - surf_loc))) \
        * np.exp(decay_param*z_depth)

    return y_out


def histogram_model(depth_bins, **kwargs):

    y_noise = noise(depth_bins, **kwargs)

    y_surface = surface(depth_bins, **kwargs)

    y_bathy = bathy(depth_bins, **kwargs)

    y_turbidity = turbidity(depth_bins, **kwargs)

    return y_noise + y_surface + y_turbidity + y_bathy

# break this into separate functions for the noise, surface peak, etc


# def bathy_model(depth_bins, surf_prom, surf_loc, surf_std,
#                 decay_param, turb_intens,
#                 noise_above, noise_below,
#                 bathy_prom, bathy_loc, bathy_std):
#     # original bathy model before getting split up, saved for debugging


#     # depth_bins is the array of elevations

#     # TRANSITION point from gaussian surface to exponential model
#     # the max intensity of the exponential curve should match the value of the gaussian model at the transition
#     transition_value = turb_intens

#     if np.log(transition_value / (surf_prom)) * (-2*surf_std**2) < 0:
#         print(transition_value, surf_prom, surf_std)

#     transition_point = surf_loc \
#         + np.sqrt(np.log(transition_value / (surf_prom))
#                   * (-2*surf_std**2))

#     # add base noise rates above and below the water surface
#     # ramp between two constant values on either side of the gaussian
#     bound_l = surf_loc - 3*surf_std
#     bound_r = transition_point
#     y_out = np.interp(depth_bins, [bound_l, bound_r], [
#                       noise_above, noise_below])

#     # SURFACE GAUSSIAN peak values
#     y_out[depth_bins < transition_point] = y_out[depth_bins < transition_point] \
#         + (surf_prom * (np.sqrt(2 * np.pi) * surf_std)) \
#         * norm.pdf(depth_bins[depth_bins < transition_point], surf_loc, surf_std)

#     # EXPONENTIAL decay values
#     # model depth values are measured from the surface peak location
#     # exponential curve begins where the surface peak value matches the input turbid intensity
#     # decay parameter is mapped to depth in m of the bins supplied by counts

#     #pseudodepth = np.arange(len(depth_bins[depth_bins >= transition_point]))
#     # y_out[depth_bins >= transition_point] =  y_out[depth_bins >= transition_point] \
#     #                                      + transition_value * np.exp(decay_param * pseudodepth)

#     if np.exp(decay_param * (transition_point - surf_loc)) == 0:
#         print(decay_param, transition_point, transition_value, surf_loc)

#     z_depth = depth_bins[depth_bins >= transition_point] - surf_loc
#     y_out[depth_bins >= transition_point] = y_out[depth_bins >= transition_point] \
#         + (transition_value / np.exp(decay_param * (transition_point - surf_loc))) \
#         * np.exp(decay_param*z_depth)

#     # BATHY gaussian guess
#     # (peak + 3std) bounded above by the surf/exp transition point
#     # peak bounded below by lowest bin
#     # magnitude bounded from 0 to surf mag
#     # std bounded from 0.1 to max bin depth/6

#     y_out = y_out \
#         + (bathy_prom * (np.sqrt(2 * np.pi) * bathy_std)) \
#         * norm.pdf(depth_bins, bathy_loc, bathy_std)

#     return y_out


def get_peak_info(hist, depth, verbose=False):
    ''' Finds peaks and calculates associated statistics for analysis, sorted by peak height'''

    depth_bin_size = np.unique(np.diff(depth))[0]

    # padding elevation mapping for peak finding at edges
    depth_padded = depth
    depth_padded = np.insert(depth_padded,
                             0,
                             depth[0] - depth_bin_size)  # use zbin for actual fcn

    depth_padded = np.insert(depth_padded,
                             len(depth_padded),
                             depth[-1] + depth_bin_size)  # use zbin for actual fcn

    dist_req_between_peaks = 0.5  # m

    if dist_req_between_peaks/depth_bin_size < 1:
        warn_msg = '''Vertical bin resolution is greater than the req. min. distance 
        between peak. Setting req. min. distance = depth_bin_size. Results may not be as expected.
        '''
        if verbose:
            warnings.warn(warn_msg)
        dist_req_between_peaks = depth_bin_size

    # note: scipy doesnt identify peaks at the start or end of the array
    # so zeros are inserted on either end of the histogram and output indexes adjusted after

    # distance = distance required between peaks - use approx 0.5 m, accepts floats >=1
    # prominence = required peak prominence - use 1 to return all
    pk_i, pk_dict = find_peaks(np.pad(hist, 1),
                               distance=dist_req_between_peaks/depth_bin_size,
                               prominence=1)

    # evaluating widths with find_peaks() seems to be using it as a threshold - not desired
    # width = required peak width (index) - use 1 to return all
    # rel_height = how far down the peak to measure its width
    # 0.5 is half way down, 1 is measured at the base
    # approximate stdm from the full width and half maximum
    pk_dict['fwhm'], pk_dict['width_heights_hm'], pk_dict['left_ips_hm'], pk_dict['right_ips_hm'] \
        = peak_widths(np.pad(hist, 1), pk_i, rel_height=0.4)

    # calculate widths at full prominence, more useful than estimating peak width by std
    pk_dict['widths_full'], pk_dict['width_heights_full'], pk_dict['left_ips_full'], pk_dict['right_ips_full'] \
        = peak_widths(np.pad(hist, 1), pk_i, rel_height=1)

    # organize into dataframe for easy sorting and reindex
    pk_dict['i'] = pk_i - 1
    pk_dict['heights'] = hist[pk_dict['i']]

    # draw a horizontal line at the peak height until it cross the signal again
    # min values within that window identifies the bases
    # preference for closest of repeated minimum values
    # ie. can give weird values to the left/right of the main peak, and to the right of a bathy peak
    # when theres noise in a scene with one 0 bin somewhere far
    pk_dict['left_z'] = depth_padded[pk_dict['left_bases']]
    pk_dict['right_z'] = depth_padded[pk_dict['right_bases']]
    pk_dict['left_bases'] -= 1
    pk_dict['right_bases'] -= 1

    # left/right ips = interpolated positions of left and right intersection points
    # of a horizontal line at the respective evaluation height.
    # mapped to input indices so needs adjustingpk_dict['left_ips'] -= 1
    pk_dict['right_ips_hm'] -= 1
    pk_dict['left_ips_hm'] -= 1
    pk_dict['right_ips_full'] -= 1
    pk_dict['left_ips_full'] -= 1

    # estimate gaussian STD from the widths
    # sigma estimate in terms of int indexes
    pk_dict['sigma_est_i'] = (pk_dict['fwhm'] / 2.35)
    # sigma estimate in terms of int indexes
    pk_dict['sigma_est_left_i'] = (
        2*(pk_dict['i'] - pk_dict['left_ips_hm']) / 2.35)
    # sigma estimate in terms of int indexes
    pk_dict['sigma_est_right_i'] = (
        2*(pk_dict['right_ips_hm'] - pk_dict['i']) / 2.35)

    # sigma estimate in terms of int indexes
    pk_dict['sigma_est'] = depth_bin_size * (pk_dict['fwhm'] / 2.35)

    # approximate gaussian scaling factor based on prominence or magnitudes
    # for gaussians range indexed
    pk_dict['prom_scaling_i'] = pk_dict['prominences'] * \
        (np.sqrt(2 * np.pi) * pk_dict['sigma_est_i'])
    pk_dict['mag_scaling_i'] = pk_dict['heights'] * \
        (np.sqrt(2 * np.pi) * pk_dict['sigma_est_i'])

    # for gaussians mapped to z
    pk_dict['prom_scaling'] = pk_dict['prominences'] * \
        (np.sqrt(2 * np.pi) * pk_dict['sigma_est'])
    pk_dict['mag_scaling'] = pk_dict['heights'] * \
        (np.sqrt(2 * np.pi) * pk_dict['sigma_est'])
    pk_dict['depth'] = depth[pk_dict['i']]

    pk_df = pd.DataFrame.from_dict(pk_dict, orient='columns')
    pk_df.sort_values(by='heights', inplace=True, ascending=False)

    # * should peaks be sorted by prominence instead?
    # worry that sorting by magnitude will bias towards shallow turbidity over sparse bathy
    # maybe sort by height for the surface, prominence for the subsurface

    return pk_df


def estimate_model_params(hist, depth, peaks, verbose=False):

    pk_df = peaks
    zero_val = 1e-31
    quality_flag = False
    params_out = {'surf_prom': np.nan,  # surface peak magnitude * consider using prom (count)
                  'surf_loc': np.nan,  # surface peak location (m)
                  'surf_std': np.nan,  # surface peak deviation (m)
                  'decay_param': np.nan,  # decay param, important to start with a low value
                  # starting intensity of turbid decay model element (count)
                  'turb_intens': np.nan,
                  # noise rate above the surface (count)
                  'noise_above': np.nan,
                  # noise rate below the surface (count)
                  'noise_below': np.nan,
                  # prominence of potential bathy peak (count)
                  'bathy_prom': np.nan,
                  # location of potentail bathy peak (m)
                  'bathy_loc': np.nan,
                  'bathy_std': np.nan}  # deviation of potential bathy peak (m)

    # set fitting bounds
    # default bounds are fixed until otherwise changed by this function
    # note: lower bounds must be strictly less than upper bounds, even when trying to 'fix' a value

    lower_bound = {'surf_prom': zero_val,  # surface peak magnitude * consider using prom (count)
                   'surf_loc': zero_val,  # surface peak location (m)
                   'surf_std': zero_val,  # surface peak deviation (m)
                   'decay_param': zero_val,  # decay param, important to start with a low value
                   # starting intensity of turbid decay model element (count)
                   'turb_intens': zero_val,
                   # noise rate above the surface (count)
                   'noise_above': zero_val,
                   # noise rate below the surface (count)
                   'noise_below': zero_val,
                   # prominence of potential bathy peak (count)
                   'bathy_prom': zero_val,
                   # location of potentail bathy peak (m)
                   'bathy_loc': zero_val,
                   'bathy_std': zero_val}  # deviation of potential bathy peak (m)

    upper_bound = {'surf_prom': 2*zero_val,  # surface peak magnitude * consider using prom (count)
                   'surf_loc': 2*zero_val,  # surface peak location (m)
                   'surf_std': 2*zero_val,  # surface peak deviation (m)
                   'decay_param': 2*zero_val,  # decay param, important to start with a low value
                   # starting intensity of turbid decay model element (count)
                   'turb_intens': 2*zero_val,
                   # noise rate above the surface (count)
                   'noise_above': 2*zero_val,
                   # noise rate below the surface (count)
                   'noise_below': 2*zero_val,
                   # prominence of potential bathy peak (count)
                   'bathy_prom': 2*zero_val,
                   # location of potentail bathy peak (m)
                   'bathy_loc': 2*zero_val,
                   'bathy_std': 2*zero_val}  # deviation of potential bathy peak (m)

    depth_bin_size = np.unique(np.diff(depth))[0]

    # ##################### QUALITY CHECKS ON HISTOGRAM ##################################

    # first, check that the histogram has at least 5 bins with actual data
    # also takes care of the 'no peaks case'
    if np.sum(hist != 0) <= 1/depth_bin_size:
        quality_flag = -1

        bounds = pd.DataFrame([lower_bound, upper_bound], index=[
            'lower', 'upper']).T
        params_out = pd.Series(params_out, name='initial')
        return params_out, bounds, quality_flag

    # next, check for clear primary/surface peak within bins with data
    if np.max(hist)/np.median(hist[np.nonzero(hist)]) < 3:
        # use nonzero because binning window may be larger than noise block
        # dont bother fitting, its probably noise only
        quality_flag = -2

        bounds = pd.DataFrame([lower_bound, upper_bound], index=[
            'lower', 'upper']).T
        params_out = pd.Series(params_out, name='initial')
        return params_out, bounds, quality_flag

    # ########################## ESTIMATING SURFACE PEAK PARAMETERS #############################

    # Surface return - largest peak
    pk_df.sort_values(by='prominences', inplace=True, ascending=False)
    surf_pk = pk_df.iloc[0]

    params_out['surf_loc'] = surf_pk.depth
    params_out['surf_std'] = surf_pk.sigma_est
    params_out['surf_prom'] = surf_pk.prominences

    # enable appropriate fitting bounds for surface model
    lower_bound['surf_loc'] = -np.inf
    lower_bound['surf_std'] = zero_val
    lower_bound['surf_prom'] = zero_val

    upper_bound['surf_loc'] = np.inf
    upper_bound['surf_std'] = np.inf
    upper_bound['surf_prom'] = 2*hist.max()

    # estimate noise rates above and below the surface peak

    # dont use ips, it will run to the edge of an array with a big peak
    # using std estimate of surface peak instead
    surface_peak_left_edge_i = np.int64(
        np.floor(surf_pk.i - 2.5 * surf_pk.sigma_est_left_i))
    surface_peak_right_edge_i = np.int64(
        np.ceil(surf_pk.i + 2.5 * surf_pk.sigma_est_right_i))

    if surface_peak_left_edge_i <= 0:
        # no bins above the surface
        params_out['noise_above'] = zero_val
        lower_bound['noise_above'] = zero_val
        upper_bound['noise_above'] = 2*zero_val
    else:
        # median of all bins left of the peak
        params_out['noise_above'] = np.median(
            hist[:int(surface_peak_left_edge_i)]) + zero_val  # eps to avoid 0
        lower_bound['noise_above'] = zero_val
        upper_bound['noise_above'] = hist.max()

    if surface_peak_right_edge_i >= len(hist):
        # surface peak on far right, no subsurface bins
        params_out['noise_below'] = zero_val
        lower_bound['noise_below'] = zero_val
        upper_bound['noise_below'] = 2*zero_val

    else:
        pass
        # will consider noise below later
        # want to avoid including shallow bathy in the noise count for now

    # ########################## ESTIMATE BATHYMETRY PEAK PARAMETERS ############################

    # Remove any histogram peaks above the surface from consideration
    bathy_pk_df = pk_df[pk_df.i > surf_pk.i]

    # If no peaks below the primary surface peak
    if bathy_pk_df.shape[0] == 0:

        # set quality flag to indicate no bathy,but good surface
        quality_flag = 1

        # leave bathy peak fitting params set to 0

        # set bathy peak values to 0
        params_out['bathy_loc'] = zero_val
        params_out['bathy_std'] = zero_val
        params_out['bathy_prom'] = zero_val

        # may still be subsurface noise
        params_out['noise_below'] = np.median(
            hist[int(surface_peak_right_edge_i):]) + zero_val
        lower_bound['noise_below'] = zero_val
        upper_bound['noise_below'] = hist.max()

        # may still have turbidity without subsurface peaks
        # e.g. in the case of a perfectly smooth turbid descent, or with coarsely binned data
        # continuing to turbidity estimation...
        water_column_right_edge_i = len(hist) - 1

    # If some peak below the primary peak, quick checks for indexing, add as needed
    if bathy_pk_df.shape[0] > 0:

        # Very shallow peak, with limited data below
        # Surface range can extend beyond the end of the array/bathy
        if surface_peak_right_edge_i >= bathy_pk_df.iloc[0].i:
            surface_peak_right_edge_i = int(bathy_pk_df.iloc[0].i) - 1

        # Bathymetry peak estimation

        # Use largest (or only) subsurface peak for bathy estimate
        bathy_pk = bathy_pk_df.iloc[0]

        params_out['bathy_loc'] = bathy_pk.depth
        params_out['bathy_std'] = bathy_pk.sigma_est
        params_out['bathy_prom'] = bathy_pk.prominences

        # Enable fitting bounds now we know theres at least one subsurface peak
        # Depth limited from bottom of surface to 5m past last bin below surface with any photons
        lower_bound['bathy_loc'] = depth[surface_peak_right_edge_i]
        upper_bound['bathy_loc'] = depth[np.nonzero(hist)[
            0][-1]] + 5  # m

        # Prominince limited by the surface peak height
        lower_bound['bathy_prom'] = zero_val
        upper_bound['bathy_prom'] = surf_pk.heights

        # Large bathy std max allows for potential fitting of deep afterpulsing/tep returns
        lower_bound['bathy_std'] = zero_val
        upper_bound['bathy_std'] = (
            upper_bound['bathy_loc'] - lower_bound['bathy_loc']) / 3

        # Estimate subsurface noise using bins excluding the bathy peak
        bathy_peak_left_edge_i = np.int64(
            np.floor(bathy_pk.i - 3 * bathy_pk.sigma_est_left_i))
        bathy_peak_right_edge_i = np.int64(
            np.ceil(bathy_pk.i + 3 * bathy_pk.sigma_est_right_i))
        water_column_right_edge_i = bathy_peak_left_edge_i + 1

        if bathy_peak_right_edge_i >= len(hist):
            pass
            # dont currently need to do this because array index slicing returns empty arrray
            # doing this when the bathy peak is at the edge includes the peak value
            #bathy_peak_right_edge_i = len(hist) - 1

        subsurface_wout_peak = np.concatenate((hist[surface_peak_right_edge_i: bathy_peak_left_edge_i],
                                               hist[bathy_peak_right_edge_i:]))

        if len(subsurface_wout_peak) == 0:
            params_out['noise_below'] = zero_val
            lower_bound['noise_below'] = zero_val
            upper_bound['noise_below'] = 2*zero_val
        else:
            params_out['noise_below'] = np.median(
                subsurface_wout_peak) + zero_val  # eps to avoid 0 in fit
            lower_bound['noise_below'] = zero_val
            upper_bound['noise_below'] = hist.max()

        # ##################### ESTIMATE QUALITY/BATHY CONFIDENCE #########################

        # If multiple subsurface peaks, compare prominences to get a confidence score
        if bathy_pk_df.shape[0] > 1:
            # more than one peak with similar prominance makes it more likely to be noise
            second_pk = bathy_pk_df.iloc[1]
            if bathy_pk.prominences <= (1.5 * second_pk.prominences):
                quality_flag = 2

            else:
                quality_flag = 3

        # If only 1 subsurface peak, compare prominence to subsurface noise to get conf.
        # suspect this wont actually happen often unless
        # - zbinsize is large
        # - bathy peak very close to surface, night time, no noise
        # - testing sample histograms
        else:
            if bathy_pk.prominences < (2 * params_out['noise_below']):
                quality_flag = 2
            else:
                quality_flag = 3

        ####################################################################################

    # ###################### ESTIMATE TURBIDITY EXPONENTIAL PARAMETERS ############################

    # Using water column photon data to replace an exponential somewhere the bottom of the water surface
    # Then mapping it back to physically meaningful parameters with some
    # back and forth between gaussians and exponentials and fun indexing
    # This is the roughest initialization in this model imo,
    # because it assumes that the exponential begins
    # near 1.5 STD below the surface peak. Recommend fitting for more precise outcomes.

    # - turbid intensity - the max photon count value at which the exponential decay begins
    # - decay parameter - exponential decay parameter in terms of DEPTH BELOW THE SURFACE PEAK

    # Consider only subsurface data between 1.5x std transition point and the bathy peak edge
    # approx the transition point at 1.5 std to roughly isolate the water column data
    transition_approx = 1.5
    water_column_left_edge_i = np.int64(np.ceil(surf_pk.i +
                                                transition_approx * surf_pk.sigma_est_right_i))

    water_column_i = np.arange(
        water_column_left_edge_i, water_column_right_edge_i)
    # remove 0 values from consideration so they dont blow up log values
    water_column_i = water_column_i[np.nonzero(hist[water_column_i])[0]]
    water_column_hist = hist[water_column_i]
    water_column_z = depth[water_column_i]
    water_column_depth = water_column_z - surf_pk.depth

    if len(water_column_i) < 2:
        params_out['turb_intens'] = 1
        params_out['decay_param'] = -zero_val

        # also loosening the bounds here so curve_fit doesnt get stuck
        lower_bound['decay_param'] = -1000  # -np.inf
        # too much flexibility makes it possible for curve fit to accidentally hit a divide by 0
        # hopefully this problem can disappear after things are reformulated but i want to know what works
        upper_bound['decay_param'] = -zero_val

        lower_bound['turb_intens'] = 1
        upper_bound['turb_intens'] = surf_pk.heights

        # use magnitude of bathy peak for very shallow peaks
        # likely that prominence is being skewed by edging up against subsurface noise
        if bathy_pk_df.shape[0] > 0:
            params_out['bathy_prom'] = bathy_pk.heights

        pass

        # this case is meant to address very very shallow bathymetry (1 or 2 bins below surface)
        # in which case, turbidity doesnt matter
        # but 'peak prominence' may be misrepresentative if the left side of the bathy peak
        # edges up against the primary peak
        # therefore, instead of using bathy prominence, consider using (magnitude - subsurface noise rate)
        # not doing this until all the other code is worked out at the least, though
        # i suspect it will bias towards the overclassification of turbidity as bathy

        # return

    else:
        # first, initialize turbidity fitting/values now that we can estimate turbidity
        lower_bound['decay_param'] = -1000  # -np.inf
        # too much flexibility makes it possible for curve fit to accidentally hit a divide by 0
        # after overfitting the decay parameter to -400 and more
        # hopefully this problem can disappear after things are reformulated but i want to know what works
        upper_bound['decay_param'] = -zero_val

        lower_bound['turb_intens'] = 1
        upper_bound['turb_intens'] = params_out['surf_prom'] * 0.5

        # next, get the decay parameter from the water column data

        # when plotted logarithmically, slope corresponds to the decay parameter
        # linear fit to get mean slope, use that

        # note: can reconstruct the original data with np.exp(b) * np.exp(m * water_column_z)

        # _i: mapped to integer indices, _z: mapped to height, _d: mapped to depth
        # may get rid of some versions later but currently useful for debugging

        # get point of intersection between modeled exp and modeled surface gauss
        # increase resolution for better intersection

        # decay modeled in terms of integer index
        m_i, b_i = np.polyfit(water_column_i, np.log(water_column_hist), 1)
        surf_col_i = np.arange(surf_pk.i, water_column_right_edge_i, 1e-2)
        surf_pk_model_i = surf_pk.mag_scaling_i * \
            norm.pdf(surf_col_i, surf_pk.i, surf_pk.sigma_est_i)
        turbid_model_i = np.exp(b_i) * np.exp(m_i * surf_col_i)
        model_diff_i = np.abs(surf_pk_model_i - turbid_model_i)

        # decay modeled in terms of depth
        m, b = np.polyfit(water_column_z, np.log(water_column_hist), 1)
        surf_col_z = np.arange(
            surf_pk.depth, depth[water_column_right_edge_i], 1e-2)
        surf_pk_model_z = surf_pk.mag_scaling * \
            norm.pdf(surf_col_z, surf_pk.depth, surf_pk.sigma_est)
        turbid_model_z = np.exp(b) * np.exp(m * surf_col_z)
        model_diff_z = np.abs(surf_pk_model_z - turbid_model_z)

        # decay modeled in terms of depth
        m_d, b_d = np.polyfit(water_column_depth,
                              np.log(water_column_hist), 1)
        surf_col_d = np.arange(0, max(water_column_depth), 1e-2)
        surf_pk_model_d = surf_pk.mag_scaling * \
            norm.pdf(surf_col_d + surf_pk.depth,
                     surf_pk.depth, surf_pk.sigma_est)
        turbid_model_d = np.exp(b_d) * np.exp(m_d * surf_col_d)
        model_diff_d = np.abs(surf_pk_model_d - turbid_model_d)

        # if models never intersect
        if model_diff_d.min() < 1e-2:
            i_inter_d = model_diff_d.argmin()
            intersect_d = surf_col_d[i_inter_d]

            i_inter_i = model_diff_i.argmin()
            intersect_i = surf_col_i[i_inter_i]

            i_inter_z = model_diff_z.argmin()
            intersect_z = surf_col_z[i_inter_z]
        else:
            # use the left most bound of the water column, in terms of depth
            intersect_d = water_column_depth.min()
            intersect_i = water_column_left_edge_i
            intersect_z = water_column_z.min()

        # photon count at intersection should be effectively the same between models
        intersect_val = np.exp(b_i) * np.exp(m_i * intersect_i)
        #intersec_val = surf_pk.mag_scaling_i * norm.pdf(intersect_i, surf_pk.i, surf_pk.sigma_est_i)

        params_out['turb_intens'] = intersect_val - \
            params_out['noise_below']

        # Checks / final setting of turbidity parameters
        if params_out['turb_intens'] < lower_bound['turb_intens']:

            warn_msg = '''Attempted to set turbidity intensity value below lower bound of {}. 
            Using lower bound for initial turbidity intensity value...'''.format(
                lower_bound['turb_intens'])

            if verbose:
                warnings.warn(warn_msg)

            params_out['turb_intens'] = lower_bound['turb_intens']

        elif params_out['turb_intens'] > upper_bound['turb_intens']:

            warn_msg = '''Attempted to set turbidity intensity value higher than the surface peak. 
            Using upper bound for initial turbidity intensity value...'''

            if verbose:
                warnings.warn(warn_msg)

            params_out['turb_intens'] = upper_bound['turb_intens']

        # using decay parameter in terms of depth because it has physical meaning built in
        # required to be negative, in bound
        if (m_d < -zero_val) & (m_d > lower_bound['decay_param']):
            params_out['decay_param'] = m_d

        else:
            params_out['decay_param'] = -zero_val

            if verbose:
                warnings.warn(
                    'Attempted to set decay parameter out of bounds, using zero value instead.')

    bounds = pd.DataFrame([lower_bound, upper_bound],
                          index=['lower', 'upper']).T
    params_out = pd.Series(params_out, name='initial')

    return params_out, bounds, quality_flag


class Waveform:
    def __init__(self, histogram, depth_bin_edges, verbose=False):

        # ensure that histogram and bin edges make sense
        assert len(histogram) == (len(depth_bin_edges) - 1)

        # require that depth bins be increasing
        if np.any(np.diff(depth_bin_edges) < 0):
            raise Exception('Elevation bins values must be increasing.')

        self.data = histogram

        # edges used to calculate the histogram
        self.bin_edges = depth_bin_edges

        # center of histogram bins
        self.depth = self.bin_edges[:-1] + np.diff(self.bin_edges)/2

        # get peak data about this waveform
        self.peaks = get_peak_info(self.data, self.depth, verbose=verbose)

        self.params, self.bounds, self.quality_flag = estimate_model_params(
            self.data, self.depth, self.peaks, verbose=verbose)

        # Backing out modeled histogram values for plotting and analysis
        self.model = histogram_model(self.depth, **self.params)
        self.noise_model = noise(self.depth, **self.params)
        self.bathy_model = bathy(self.depth, **self.params)
        self.surface_model = surface(self.depth, **self.params)
        self.turbidity_model = turbidity(self.depth, **self.params)

        # find a better term than high_res
        self.depth_high_res = np.linspace(
            self.depth.min(), self.depth.max(), 1000)
        self.noise_model_high_res = noise(self.depth_high_res, **self.params)
        self.bathy_model_high_res = bathy(self.depth_high_res, **self.params)
        self.surface_model_high_res = surface(
            self.depth_high_res, **self.params)
        self.turbidity_model_high_res = turbidity(
            self.depth_high_res, **self.params)

    def __str__(self):
        '''Human readable summary'''
        s = f"""----- PSEUDOWAVEFORM -----
TOTAL PHOTONS: {np.sum(self.data)}
Depth Range : [{min(self.bin_edges)}m, {max(self.bin_edges)}m]
Depth Bin Count : {len(self.bin_edges) - 1}
Peak Count : {self.peaks.shape[0]}
        """
        return s

    def deconvolve_atlas(self):
        # deconvolves the atlas response at whatever the depth resolution is
        pass

    def fit(self):
        # fits the estimated waveform with non linear least squares
        pass

    def show(self):
        # plots the pseudowaveform with model
        # dont make any new calculations here!
        f = plt.figure(figsize=[4.8, 6.4])

        # plt.plot(self.model, self.depth, 'ro-')
        plt.plot(self.noise_model_high_res + self.surface_model_high_res + self.turbidity_model_high_res +
                 self.bathy_model_high_res, self.depth_high_res, 'b.', label='Seafloor Return')

        plt.plot(self.noise_model_high_res + self.surface_model_high_res,
                 self.depth_high_res, 'r.', label='Surface Return')

        plt.plot(self.noise_model_high_res + self.turbidity_model_high_res,
                 self.depth_high_res, 'g.', label='Turbidity')

        plt.plot(self.noise_model_high_res,
                 self.depth_high_res, marker='.', linestyle='', color='grey', label='Noise')

        plt.plot(self.data, self.depth, 'kx-', linewidth=2, label='Histogram')
        plt.plot(self.model, self.depth, marker='s', linestyle='-',
                 color='darkorange', linewidth=2, alpha=0.75, label='Histogram Model')

        plt.legend()
        plt.grid('both')
        plt.gca().invert_yaxis()

        plt.ylabel('Depth (m)')
        plt.xlabel('Photons (count)')
        plt.title('Pseudowaveform Model Components')

        plt.tight_layout()
        plt.show()

        return f


if __name__ == "__main__":
    import numpy as np

    # increasing in depth, decreasing in elevation
    hist = np.array([1, 2, 10, 25, 8, 5, 5, 4, 8, 2, 2, 1])

    # depth bins must be increasing L-R
    depth_bin_edges = np.linspace(-5, 15, len(hist)+1)

    w = Waveform(hist, depth_bin_edges)
    print(w)
    w.show()

    print('')
