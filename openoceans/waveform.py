import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd

from scipy.signal import medfilt, peak_widths, peak_prominences, find_peaks
from scipy.stats import exponnorm, norm, skewnorm, expon
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d


# to do
# - try out slope based peak finding
# - add error calculations to the waveform
# - remove noise components


def gauss(x, c, mu, sigma):
    return c * (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x - mu)**2 / (2 * sigma**2))


def is_peak_ap(test_peak_mag, test_peak_loc, surf_peak_mag, surf_peak_loc):
    # removing checks on ratios
    # saturation can only make up part of a window, lowering the ratio
    # rely on quality_ph from atl03 to indiciate sensor saturation
    # remove any peaks that roughly match

    check = False
    # get dist from surface
    dist_ap = surf_peak_loc - test_peak_loc
    ratio_ap = test_peak_mag / surf_peak_mag
    # print(dist_ap, ratio_ap)
    # check if distance is within +-0.2m of rough loc
    # recall dist ap should be negative
    dist_thresh = 0.25

    # check if ratio is close (+- 2%)
    # given this as error relative to total surface peak photons
    # actually a pretty generous bound...
    ratio_thresh = 0.01

    # (dist, ratio)
    aps = [(0.45, 2e-2),
           (0.9, 4e-3),
           (1.35, 1.9e-3),
           (1.8, 1e-3),
           (2.25, 1.5e-3),
           (4.2, 7.2e-4),
           (6.45, 3.5e-5),
           (8.4, 1.7e-5)]
    i = 0
    for ap in aps:

        dist_check = np.abs(dist_ap + ap[0]) < dist_thresh

        # see note at top of function
        # ratio_check = np.abs(ratio_ap - ap[1]) / ap[1] < ratio_thresh

        # # make the ratio error a function of the total peak
        # ratio_check = np.abs(test_peak_mag - surf_peak_mag *
        #                      ap[1]) / surf_peak_mag < ratio_thresh

        ratio_check = True

        # print(dist_check, ratio_check)

        if dist_check and ratio_check:
            check = True
            break

        i += 1

    return check, i


def find_transition_point(turb_intens, surf_prom, surf_loc, surf_std):
    """Calculates the depth value at which the model switches from the surface gaussian to exponential decay. 

    Args:
        turb_intens (float): A value on the surface curve below the surface at which to calculate the corresponding depth, in photons.
        surf_prom (float): Peak prominence of the surface return, in photons.
        surf_loc (float): Location of the surface return, in meters.
        surf_std (float): Standard deviation of the surface return gaussian model.

    Returns:
        float: Depth corresponding to input photon count value, in meters.
    """
    transition_value = turb_intens

    # debugging aid
    if np.log(transition_value / (surf_prom)) * (-2*surf_std**2) < 0:
        print(transition_value, surf_prom, surf_std)

    transition_point = surf_loc \
        + np.sqrt(np.log(transition_value / (surf_prom))
                  * (-2*surf_std**2))

    return transition_point


def noise(depth_bins, surf_prom, surf_loc, surf_std,
          decay_param, trans_mult,
          noise_above, noise_below,
          bathy_prom, bathy_loc, bathy_std):
    """Model for noise in a pseudowaveform. Consists of an above surface noise rate (air) and a below surface noise rate (water column). 

    Air noise rate is from the top of the waveform (minimum depth) to the top of the surface return (modeled as 3 standard deviations above the surface peak location). The subsurface noise rate begins at the transition point between the surface gaussian model and the turbidity exponential components, and continues to the bottom of the waveform (max depth). Data is linearly interpolated between these values (i.e. across the surface return).

    Args:
        depth_bins (array of float): Centers of depth histogram bins, in meters.
        surf_prom (float): Peak prominence of the surface return, in photons.
        surf_loc (float): Location of the surface return, in meters.
        surf_std (float): Standard deviation of the surface return gaussian model.
        decay_param (float): Decay parameter of the turbidity exponential model component.
        trans_mult (float): Distance of turbidity below surface peak, in multiples of STD.
        noise_above (float): Air noise rate, in photons/bin.
        noise_below (float): Subsurface noise rate, in photons/bin.
        bathy_prom (_type_): Not used, provided for consistency of input across model components.
        bathy_loc (_type_): Not used, provided for consistency of input across model components.
        bathy_std (_type_): Not used, provided for consistency of input across model components.

    Returns:
        array: Noise model output, matching shape of input depth array.
    """

    # transition_point = find_transition_point(
    #     turb_intens, surf_prom, surf_loc, surf_std)

    transition_point = surf_loc + trans_mult * surf_std

    # add base noise rates above and below the water surface
    # ramp between two constant values on either side of the gaussian
    bound_l = surf_loc - 3*surf_std
    bound_r = transition_point

    return np.interp(depth_bins, [bound_l, bound_r], [noise_above, noise_below])


def bathy(depth_bins, surf_prom, surf_loc, surf_std,
          decay_param, trans_mult,
          noise_above, noise_below,
          bathy_prom, bathy_loc, bathy_std):
    """Gaussian model for the seafloor return. Currently intended to be added to the noise model.

    Args:
        depth_bins (array of float): Centers of depth histogram bins, in meters.
        surf_prom (float): Not used, provided for consistency of input across model components.
        surf_loc (float): Not used, provided for consistency of input across model components.
        surf_std (float): Not used, provided for consistency of input across model components.
        decay_param (float): Not used, provided for consistency of input across model components.
        trans_mult (float): Not used, provided for consistency of input across model components.
        noise_above (float): Not used, provided for consistency of input across model components.
        noise_below (float): Not used, provided for consistency of input across model components.
        bathy_prom (float): Peak prominence of the seafloor return, in photons.
        bathy_loc (float): Peak location of the seafloor return, in photons.
        bathy_std (float): Standard deviation of the seafloor return, in photons.

    Returns:
        array: Seafloor model output, matching shape of input depth array.
    """

    return (bathy_prom * (np.sqrt(2 * np.pi) * bathy_std)) \
        * norm.pdf(depth_bins, bathy_loc, bathy_std)


def surface(depth_bins, surf_prom, surf_loc, surf_std,
            decay_param, trans_mult,
            noise_above, noise_below,
            bathy_prom, bathy_loc, bathy_std):
    """Gaussian model for the water surface return. Returns values from the top of the water surface, to the start of the column turbidity model below surface, as defined by turbidity/decay model inputs. Currently intended to be added with other model components (noise, turbidity).

    Args:
        depth_bins (array of float): Centers of depth histogram bins, in meters.
        surf_prom (float): Peak prominence of the surface return, in photons.
        surf_loc (float): Location of the surface return, in meters.
        surf_std (float): Standard deviation of the surface return gaussian model.
        decay_param (float): Decay parameter of the turbidity exponential model component.
        trans_mult (float): Distance of turbidity below surface peak, in multiples of STD.
        noise_above (float): Not used, provided for consistency of input across model components.
        noise_below (float): Not used, provided for consistency of input across model components.
        bathy_prom (float): Not used, provided for consistency of input across model components.
        bathy_loc (float): Not used, provided for consistency of input across model components.
        bathy_std (float): Not used, provided for consistency of input across model components.


    Returns:
        array: Surface model output, matching shape of input depth array.
    """

    # TRANSITION point from gaussian surface to exponential model
    # the max intensity of the exponential curve should match the value of the gaussian model at the transition
    # transition_point = find_transition_point(
    #     turb_intens, surf_prom, surf_loc, surf_std)

    transition_point = surf_loc + surf_std * trans_mult

    # SURFACE GAUSSIAN peak values
    y_out = np.zeros_like(depth_bins)

    y_out[depth_bins < transition_point] = (surf_prom * (np.sqrt(2 * np.pi) * surf_std)) \
        * norm.pdf(depth_bins[depth_bins < transition_point], surf_loc, surf_std)

    return y_out


def turbidity(depth_bins, surf_prom, surf_loc, surf_std,
              decay_param, trans_mult,
              noise_above, noise_below,
              bathy_prom, bathy_loc, bathy_std):
    """Exponential decay model for water column / turbidity. Returns values from the bottom of the water surface to the max depth of the waveform. Currently intended to be added with other model components (noise, surface).

    Args:
        depth_bins (array of float): Centers of depth histogram bins, in meters.
        surf_prom (float): Peak prominence of the surface return, in photons.
        surf_loc (float): Location of the surface return, in meters.
        surf_std (float): Standard deviation of the surface return gaussian model.
        decay_param (float): Decay parameter of the turbidity exponential model component.
        trans_mult (float): Distance of turbidity below surface peak, in multiples of STD.
        noise_above (float): Not used, provided for consistency of input across model components.
        noise_below (float): Not used, provided for consistency of input across model components.
        bathy_prom (float): Not used, provided for consistency of input across model components.
        bathy_loc (float): Not used, provided for consistency of input across model components.
        bathy_std (float): Not used, provided for consistency of input across model components.


    Returns:
        array: Turbidity model output, matching shape of input depth array.
    """

    # depth_bins is the array of elevations

    # TRANSITION point from gaussian surface to exponential model
    # the max intensity of the exponential curve should match the value of the gaussian model at the transition
    # transition_point = find_transition_point(
    #     turb_intens, surf_prom, surf_loc, surf_std)

    transition_point = surf_loc + surf_std * trans_mult  # depth

    turb_intens = surf_prom * np.sqrt(np.pi * 2 * surf_std**2) * norm.pdf(transition_point,
                                                                          loc=surf_loc,
                                                                          scale=surf_std)  # actual value on the normal dist

    # calculating starting intensity of decay from surface gaussian and transition

    # EXPONENTIAL decay values
    # model depth values are measured from the surface peak location
    # exponential curve begins where the surface peak value matches the input turbid intensity
    # decay parameter is mapped to depth in m of the bins supplied by counts
    y_out = np.zeros_like(depth_bins)

    z_depth = depth_bins[depth_bins >= transition_point] - surf_loc

    # why do I have a division here?
    y_out[depth_bins >= transition_point] = \
        (turb_intens / np.exp(decay_param * (transition_point - surf_loc))) \
        * np.exp(decay_param*z_depth)

    # y_out[depth_bins >= transition_point] = \
    #     (turb_intens) \
    #     * np.exp(decay_param*z_depth)

    return y_out


def histogram_model(depth_bins, surf_prom, surf_loc, surf_std,
                    decay_param, trans_mult,
                    noise_above, noise_below,
                    bathy_prom, bathy_loc, bathy_std):
    """Combines all the histogram model components into one function. 

    Used for optimization with scipy's curve_fit within the Waveform.fit() method.

    Args:
    Args:
        depth_bins (array of float): Centers of depth histogram bins, in meters.
        surf_prom (float): Peak prominence of the surface return, in photons.
        surf_loc (float): Location of the surface return, in meters.
        surf_std (float): Standard deviation of the surface return gaussian model.
        decay_param (float): Decay parameter of the turbidity exponential model component.
        trans_mult (float): Distance of turbidity below surface peak, in multiples of STD.
        noise_above (float): Air noise rate, in photons/bin.
        noise_below (float): Subsurface noise rate, in photons/bin.
        bathy_prom (float): Peak prominence of the seafloor return, in photons.
        bathy_loc (float): Peak location of the seafloor return, in photons.
        bathy_std (float): Standard deviation of the seafloor return, in photons.

    Returns:
        array: Combined model output matching shape of input depth array.
    """

    # curve_fit does not appreciate keyword arguments

    y_noise = noise(depth_bins, surf_prom, surf_loc, surf_std,
                    decay_param, trans_mult,
                    noise_above, noise_below,
                    bathy_prom, bathy_loc, bathy_std)

    y_surface = surface(depth_bins, surf_prom, surf_loc, surf_std,
                        decay_param, trans_mult,
                        noise_above, noise_below,
                        bathy_prom, bathy_loc, bathy_std)

    y_bathy = bathy(depth_bins, surf_prom, surf_loc, surf_std,
                    decay_param, trans_mult,
                    noise_above, noise_below,
                    bathy_prom, bathy_loc, bathy_std)

    y_turbidity = turbidity(depth_bins, surf_prom, surf_loc, surf_std,
                            decay_param, trans_mult,
                            noise_above, noise_below,
                            bathy_prom, bathy_loc, bathy_std)

    return y_noise + y_surface + y_turbidity + y_bathy


def get_peak_info(hist, depth, verbose=False):
    """Evaluates input histogram to find peaks and associated peak statistics (calculated by peak_widths(), peak_prominences(), find_peaks() from scipy.signal). Pads input to return peaks at the start or end of the input array. Returns dataframe sorted by prominence.

    Args:
        hist (array): Histogram of photons by depth.
        depth (array): Centers of depth bins used to histogram photon returns.
        verbose (bool, optional): Option to print output and warnings. Defaults to False.

    Returns:
        Pandas DataFrame: DataFrame of peaks and peak statistics. 

        Columns include:
            'i':                    Integer index of the peak location in the input histogram.

            'heights':              Peak heights, in photons. I.e. return intensity.

            'depth':                Depth of the peak, in meters.

            'prominences' :         Peak prominences.

            'left_bases':           Integer location of left peak base (higher elev/lower depth). Peak bases are defined as the nearest minimum value on the interval found by extending a horizontal line from the current peak to either the edge of the window or to an intersection with a higher peak (same height peaks are ignored).

            'right_bases':          Integer location of right peak base (higher elev/lower depth). 

            'left_z':               Depth location of left peak base (higher elev/lower depth). 

            'right_z':              Depth location of right peak base (higher elev/lower depth). 

            'fwhm':                 Approx full-width at half maximum (at 60% relative height).

            'width_heights_hm':     Height of the contour lines at which fwhm was calulated.

            'left_ips_hm':          Interpolated positions of left intersection point of a horizontal line at half maximum.

            'right_ips_hm':         Interpolated positions of right intersection point of a horizontal line at half maximum.

            'widths_full':          Peak width at at its lowest contour line.

            'width_heights_full':   Height of contour lines at base of peak.

            'left_ips_full':        Interpolated positions of left intersection point of a horizontal line at the base of the peak.

            'right_ips_full':       Interpolated positions of right intersection point of a horizontal line at the base of the peak.

            'sigma_est_i':          Peak standard deviation, in terms of integer indices.

            'sigma_est_left_i':     Leftward (higher elev, shallower depth) peak standard deviation, in terms of integer indices.

            'sigma_est_right_i':    Rightward (lower elev, deeper depth) peak standard deviation, in terms of integer indices.

            'sigma_est':            Peak standard deviation, in meters.

            'prom_scaling_i':       Normal distribution scaling parameter to match peak prominence, for a distribution mapped to integer inputs.

            'mag_scaling_i':        Normal distribution scaling parameter to match absolute peak height, for a distribution mapped to integer inputs.

            'prom_scaling':         Normal distribution scaling parameter to match peak prominence, for a distribution with depth inputs.

            'mag_scaling':          Normal distribution scaling parameter to match absolute peak height, for a distribution with depth inputs.
    """

    depth_bin_size = np.unique(np.diff(depth))[0]

    # padding elevation mapping for peak finding at edges
    depth_padded = depth
    depth_padded = np.insert(depth_padded,
                             0,
                             depth[0] - depth_bin_size)  # use zbin for actual fcn

    depth_padded = np.insert(depth_padded,
                             len(depth_padded),
                             depth[-1] + depth_bin_size)  # use zbin for actual fcn

    dist_req_between_peaks = 0.49999  # m

    if dist_req_between_peaks/depth_bin_size < 1:
        warn_msg = '''get_peak_info(): Vertical bin resolution is greater than the req. min. distance 
        between peak. Setting req. min. distance = depth_bin_size. Results may not be as expected.
        '''
        if verbose:
            warnings.warn(warn_msg)
        dist_req_between_peaks = depth_bin_size

    # note: scipy doesnt identify peaks at the start or end of the array
    # so zeros are inserted on either end of the histogram and output indexes adjusted after

    # distance = distance required between peaks - use approx 0.5 m, accepts floats >=1
    # prominence = required peak prominence
    pk_i, pk_dict = find_peaks(np.pad(hist, 1),
                               distance=dist_req_between_peaks/depth_bin_size,
                               prominence=0.1)

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
    pk_df.sort_values(by='prominences', inplace=True, ascending=False)

    # is height or prominence the best way to sort peaks?

    return pk_df


def estimate_model_params(hist, depth, peaks=None, verbose=False):
    """Calculates parameters for noise, surface, bathy, and turbidity model components from statistical approximations. Also outputs parameter bounds to use for curve fitting and a quality flag (in progress).

    Quality flag interpretation:
        -2 Photons with no distinct peaks.
        -1 Not enough data (less than 1m worth of bins).
        0  Not set
        1  Surface peak exists, but no subsurface peaks found.
        2  Surface peak and weak bathy peak identified (prominence < noise_below * thresh).
        3  Surface peak and strong bathy peak identified (prominence > noise_below * thresh).

    Args:
        hist (array): Histogram of photons by depth.
        depth (array): Centers of depth bins used to histogram photon returns.
        peaks (DataFrame): Peak info statistics. If None, will calculate from get_peak_info().
        verbose (bool, optional): Option to output warnings/print statements. Defaults to False.

    Returns:
        params_out (dict): Model parameters. See histogram_model() docstring for more detail.
        bounds (DataFrame): Two-column dataframe consisting of upper and lower bounds on each model parameter. Required for curve fitting to ensure reasonable results and convergent solution. 
        quality_flag (int): Flag summarizing parameter confidence.
    """
    if isinstance(peaks, pd.DataFrame):
        pass
    else:
        peaks = get_peak_info(hist, depth)

    pk_df = peaks
    zero_val = 1e-31
    quality_flag = False
    params_out = {'surf_prom': np.nan,  # surface peak magnitude * consider using prom (count)
                  'surf_loc': np.nan,  # surface peak location (m)
                  'surf_std': np.nan,  # surface peak deviation (m)
                  'decay_param': np.nan,  # decay param, important to start with a low value
                  # starting depth of turbid decay model element (std multiples)
                  'trans_mult': np.nan,
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
                   # starting depth of turbid decay model element (std multiples)
                   'trans_mult': zero_val,
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
                   # # starting depth of turbid decay model element (std multiples)
                   'trans_mult': 2*zero_val,
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

    # first, check that the histogram has at least 0.5m with actual data
    # also takes care of the 'no peaks case'
    if np.sum(hist != 0) <= 0.5/depth_bin_size:
        quality_flag = -1

        bounds = pd.DataFrame([lower_bound, upper_bound], index=[
            'lower', 'upper']).T
        params_out = pd.Series(params_out, name='initial')
        bathy_quality_ratio = -1
        return params_out, bounds, quality_flag, bathy_quality_ratio

    # next, check for clear primary/surface peak within bins with data
    # if np.max(hist)/np.median(hist[np.nonzero(hist)]) < 3:

    #     # this is problematic when the only signal is a strong peak, median will be high, removing

    #     # use nonzero because binning window may be larger than noise block
    #     # dont bother fitting, its probably noise only
    #     quality_flag = -2

    #     bounds = pd.DataFrame([lower_bound, upper_bound], index=[
    #         'lower', 'upper']).T
    #     params_out = pd.Series(params_out, name='initial')
    #     return params_out, bounds, quality_flag

    # need a way to avoid chunks of random noise
    # require there be 1 peak at least X% higher than the next peak 
    # kind of like the two peak check later on

    if pk_df.shape[0] > 2:
        # two similar intensity returns can be surface/bathy
        # but 3 similar intensity returns are more likely to be noise than anything else
        similar_secondary = ((pk_df.iloc[1].prominences) / pk_df.iloc[0].prominences) > 0.6
        similar_tertiary = ((pk_df.iloc[2].prominences) / pk_df.iloc[0].prominences) > 0.6

        if similar_secondary and similar_tertiary:
            # more likely to be noise than bathy
            # weak signal all around
            quality_flag = -3
            bounds = pd.DataFrame([lower_bound, upper_bound], index=[
                'lower', 'upper']).T
            params_out = pd.Series(params_out, name='initial')
            bathy_quality_ratio = -1
            return params_out, bounds, quality_flag, bathy_quality_ratio


    if pk_df.shape[0] == 0:
        # no peaks, exit
        quality_flag = -2
        bounds = pd.DataFrame([lower_bound, upper_bound], index=[
            'lower', 'upper']).T
        params_out = pd.Series(params_out, name='initial')
        bathy_quality_ratio = -1
        return params_out, bounds, quality_flag, bathy_quality_ratio

    # ########################## ESTIMATING SURFACE PEAK PARAMETERS #############################

    # Surface return - largest peak
    pk_df.sort_values(by='prominences', inplace=True, ascending=False)

    # if the top two peaks are within 20% of each other
    # go with the higher elev one/the one closest to 0m
    # mitigating super shallow, bright reefs where seabed is actually brighter

    
    if pk_df.shape[0] > 1:

        # check if second peak is greater than 60% the height of the primary
        two_tall_peaks = ((pk_df.iloc[1].prominences) / pk_df.iloc[0].prominences) > 0.6

        if two_tall_peaks:

            # use the peak above the other
            # using the one nearest to 0 might have issues with tides
            pks2 = pk_df.iloc[[0, 1]]
            peak_at_0 = np.abs(pks2.depth).argmin()
            peak_above = pks2.depth.argmin()
            
            if (peak_at_0 != peak_above):
                if verbose:
                    print('Huh, you found a chunk you with 2 surfaces that dont make sense! Check that out!')

            surf_pk = pk_df.iloc[peak_above]

        else:
            surf_pk = pk_df.iloc[0]
    else:
        surf_pk = pk_df.iloc[0]



    # but this is a rough estimate of the surface peak assuming perfect gaussian on peak
    # surface peak is too important to leave this rough estimate
    # surf peak can have a turbid subsurface tail, too

    # dogleg detection - where does the surface peak end and become subsurface noise/signal?
    # basically mean value theorem applied to the subsurface histogram slope
    # how to know when the histogram starts looking like possible turbidity
    # we assume the surface should dissipate after say X meters depth, but will sooner in actuality
    # therefore, the actual numerical slope must cross the threshold required to dissipate after X meters at some point
    # where X is a function of surface peak width

    # what if the histogram has a rounded surface peak, or shallow bathy peaks?
    # rounded surface peak would be from pos to neg (excluded)
    # shallow bathy peaks would be the 2nd or 3rd slope threshold crossing
    # some sketches with the curve and mean slope help show the sense behind this

    # we'll use this to improve our estimate of the surface gaussian early on
    # dog leg detection start
    dissipation_range = 3  # m #surf_pk.sigma_est * 6
    z_bin_approx = np.diff(depth)[0]
    slope_thresh = -surf_pk.heights / (dissipation_range/z_bin_approx)
    diffed_subsurf = np.diff(hist[np.int64(surf_pk.i):])

    # detection of slope decreasing in severity, crossing the thresh
    sign_ = np.sign(diffed_subsurf - slope_thresh)
    # ie where sign changes (from negative to positive only)
    sign_changes_i = np.where(
        (sign_[:-1] != sign_[1:]) & (sign_[:-1] < 0))[0] + 1

    if len(sign_changes_i) == 0:
        # poorly conditioned surface peak
        # tbh no idea why this would even hit if we made it this far
        no_sign_change = True
        print('Check for poorly conditioned surface peak.')
        quality_flag = -2

        bounds = pd.DataFrame([lower_bound, upper_bound], index=[
            'lower', 'upper']).T
        params_out = pd.Series(params_out, name='initial')
        bathy_quality_ratio = -1
        return params_out, bounds, quality_flag, bathy_quality_ratio

    else:
        # calculate dogleg corner details
        transition_i = np.int64(surf_pk.i) + \
            sign_changes_i[0]  # full hist index
        transition_depth = depth[transition_i]
        transition_mult = (transition_depth -
                           surf_pk.depth) / surf_pk.sigma_est
    # end dog leg detection

    # just fit the gaussian here, why not improve your guess
    surf_range_i = np.arange((np.int64(surf_pk.i) - sign_changes_i[0] - 1),
                             (np.int64(surf_pk.i) + sign_changes_i[0] + 1))

    p_init = [surf_pk.mag_scaling, surf_pk.depth, surf_pk.sigma_est]

    [_surf_mag_scale, _surf_loc, _surf_sigma] = p_init

    try:
        [_surf_mag_scale, _surf_loc, _surf_sigma], _ = curve_fit(gauss,
                                                                 depth[surf_range_i],
                                                                 hist[surf_range_i],
                                                                 p0=p_init,
                                                                 bounds=([zero_val, -np.inf, zero_val],
                                                                         [3*surf_pk.mag_scaling, np.inf, np.inf]))
    except:
        pass

    params_out['surf_loc'] = _surf_loc
    params_out['surf_std'] = _surf_sigma
    params_out['surf_prom'] = _surf_mag_scale / \
        np.sqrt(2 * np.pi * _surf_sigma**2)

    # params_out['surf_loc'] = surf_pk.depth
    # params_out['surf_std'] = surf_pk.sigma_est
    # params_out['surf_prom'] = surf_pk.prominences

    # enable appropriate fitting bounds for surface model
    lower_bound['surf_loc'] = -np.inf
    lower_bound['surf_std'] = zero_val
    lower_bound['surf_prom'] = zero_val

    upper_bound['surf_loc'] = np.inf
    upper_bound['surf_std'] = np.inf
    upper_bound['surf_prom'] = 2*hist.max()

    # estimate noise rates above and below the surface peak - UPDATE THIS ITS NOT GREAT

    # dont use ips, it will run to the edge of an array with a big peak
    # using std estimate of surface peak instead
    surface_peak_left_edge_i = np.int64(
        np.floor(surf_pk.i - 2.5 * surf_pk.sigma_est_left_i))

    # using more detailed estimate of peak edge
    surface_peak_right_edge_i = transition_i
    # surface_peak_right_edge_i = np.int64(
    #     np.ceil(surf_pk.i + 2.5 * surf_pk.sigma_est_right_i))

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
        # leave bathy peak fitting params set to 0

        # set bathy peak values to 0
        params_out['bathy_loc'] = zero_val
        params_out['bathy_std'] = zero_val
        params_out['bathy_prom'] = zero_val

        # may still be subsurface noise
        # commenting out for now, will assume subsurface noise picked up by decay

        # params_out['noise_below'] = np.median(
        #     hist[int(surface_peak_right_edge_i):]) + zero_val
        
        # note the parameter is still fittable
        params_out['noise_below'] = zero_val

        lower_bound['noise_below'] = zero_val
        upper_bound['noise_below'] = hist.max()

        # may still have turbidity without subsurface peaks
        # e.g. in the case of a perfectly smooth turbid descent, or with coarsely binned data
        # continuing to turbidity estimation...

    # If some peak below the primary peak, quick checks for indexing, add as needed
    else:  # df.shape always non negative

        # this section can be improved

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

        # Prominence limited by the surface peak height
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
            # see where this line is commented out above for details
        
            # params_out['noise_below'] = np.median(
            #     subsurface_wout_peak) + zero_val  # eps to avoid 0 in fit
            
            params_out['noise_below'] = zero_val
            
            lower_bound['noise_below'] = zero_val
            upper_bound['noise_below'] = hist.max()

    # ###################### ESTIMATE TURBIDITY EXPONENTIAL PARAMETERS ############################

    # Using water column photon data to replace an exponential somewhere the bottom of the water surface
    # Then mapping it back to physically meaningful parameters with some
    # back and forth between gaussians and exponentials and fun indexing

    # - transition mult - how many multiples of surf_peak sigma below the surface to start turb model
    # - decay parameter - exponential decay parameter in terms of DEPTH BELOW THE SURFACE PEAK

    # briefly consider two possibilities - a clear bathy peak and a less clear bathy peak
    # for a very clear bathy peak we should remove it before estimating turbidity
    # for a less clear bathy peak it shouldnt have as much of an effect and so we may leave it in
    # and we cant confidently remove it anyways

    # # if high confidence bathy, get a deconvolved subsurface histogra hist_t
    # high_conf_mult_peaks = False
    # high_conf_single_peak = False
    # if (bathy_pk_df.shape[0] > 1):

    #     if (bathy_pk.prominences > (3 * bathy_pk_df.iloc[1].prominences)):
    #         # bathy peak is more than thrice as prominent as the next highest peak
    #         high_conf_mult_peaks = True

    # elif (bathy_pk_df.shape[0] == 1):

    #     if (bathy_pk.prominences > ((1/3) * params_out['surf_prom'])):
    #         # bathy peak is more than a third as tall as the main surface peak
    #         high_conf_single_peak = True

    # if high_conf_mult_peaks or high_conf_single_peak:

    if (bathy_pk_df.shape[0] > 0):
        # remove bathy peak from depth consideration

        # # adding +-1 to extend window just to be safe
        bathy_range_i = np.arange(np.int64(bathy_pk.i - np.floor(3 * bathy_pk.sigma_est_left_i) - 1),
                                  min(np.int64(bathy_pk.i + np.ceil(3 * bathy_pk.sigma_est_right_i) + 1), len(hist)))
            # min check above to make sure we dont go beyond the length of the histogram data
        # get indexer for non bathy data below the surface
        not_bathy_subsurface = np.full((len(hist), ), True)
        # remove surface and higher
        not_bathy_subsurface[:transition_i] = False
        not_bathy_subsurface[bathy_range_i] = False

        hist_t = hist[not_bathy_subsurface]
        depth_t = depth[not_bathy_subsurface]

    else:
        hist_t = hist[transition_i:]
        depth_t = depth[transition_i:]

    # hist/depth _t represent turbidity specific depth and histogram data
    # starts at the transition point and MAY have depth bins missing (removed bathy peak)

    # picking back up from transition point detection used in surface peak fitting
    # set up parameters and quality checks up to this point
    no_sign_change = False
    not_enough_subsurface = False
    too_far_subsurface = False

    lower_bound['trans_mult'] = 0.5
    upper_bound['trans_mult'] = 10 + zero_val
    lower_bound['decay_param'] = -1000  # -np.inf
    upper_bound['decay_param'] = -zero_val

    if no_sign_change == False:
        # transition point finding went ok earlier
        # continue with some quality checks
        # non_zero_column_bins = np.argwhere(hist[transition_i:] > 0).flatten()
        non_zero_column_bins = np.argwhere(hist_t > 0).flatten()

        if (len(non_zero_column_bins) < 3):
            # not enough data to actually evaluate 'turbidity'
            not_enough_subsurface = True

        if transition_mult > 10:
            # out of bounds for reasonable value
            too_far_subsurface = True

    if no_sign_change or not_enough_subsurface or too_far_subsurface:
        # null case - smooth transition from peak to nothing
        # or just too few bins to reasonably calculate turbidity
        # or first rise in photons is too far below the surface to be turb.
        params_out['trans_mult'] = 10  # get far below the surface gauss
        params_out['decay_param'] = -zero_val  # send to zero asap

    else:
        # continue estimating turbidity decay parameter
        # deepest_nonzero_bin_edge = np.argwhere(hist > 0).flatten()[-1] + 1
        deepest_nonzero_bin_edge = np.argwhere(hist_t > 0).flatten()[-1] + 1
        # extra +1 due to range behavior, not actual data
        # wc = water column
        # wc_i = np.arange(transition_i, deepest_nonzero_bin_edge + 1)
        wc_i = np.arange(0, deepest_nonzero_bin_edge)

        # getting the decay parameter *in terms of DEPTH* to preserve any physical mapping
        # technically more of an inverted height than a depth
        # wc_z = depth[wc_i]
        # wc_hist_ = hist[wc_i]
        wc_z = depth_t[wc_i]
        wc_hist_ = hist_t[wc_i]
        wc_depth_ = wc_z - params_out['surf_loc']  # true depth below surface

        # ignoring zero bins so log doesnt explode
        wc_hist = wc_hist_[wc_hist_ > 0]
        wc_depth = wc_depth_[wc_hist_ > 0]

        # remember these data might be missing a chunk where a bathy peak was when indexing

        # decay modeled in terms of depth
        # weighting to account for deviation in large hist values getting scaled
        # and to prioritize shallow data
        # inverse dist assuming max depth of 50m * log scaling weights
        weights = np.sqrt(wc_hist) * (1 - wc_depth/50)
        m, b = np.polyfit(wc_depth, np.log(wc_hist), 1, w=weights)

        # would typically use sqrt(y) for weights, but we know all values are pos
        # and we really really care about those values right near the surface

        turb_intens = np.exp(b)
        decay_param = m

        if (decay_param > -zero_val) or (decay_param < -1000) \
                or (turb_intens > surf_pk.heights) or (turb_intens < 0):
            # poorly conditioned decay component, use null values
            params_out['trans_mult'] = 10
            params_out['decay_param'] = -zero_val

        else:
            # we need to recalculate our transition point now that we have a turbidity model
            # where does our turbidity model intersect with our surface gaussian?

            # reconstructing higher res versions of these models and min(diff) them

            surf_col_z = np.arange(
                params_out['surf_loc'], depth[(np.argwhere(hist > 0).flatten()[-1])], 1e-3)

            surf_pk_model_z = params_out['surf_prom'] * (np.sqrt(2 * np.pi*params_out['surf_std']**2)) * \
                norm.pdf(surf_col_z,
                         params_out['surf_loc'], params_out['surf_std'])

            turbid_model_z = turb_intens * np.exp(decay_param * surf_col_z)

            model_diff = surf_pk_model_z - turbid_model_z

            # first sign change is the intersection we want
            sign_ = np.sign(model_diff)
            # ie where sign changes (from negative to positive only)
            sign_changes_i = np.where((sign_[:-1] != sign_[1:]))[0] + 1

            if (len(sign_changes_i) == 0) or \
                    ((surf_col_z[sign_changes_i[0]] - params_out['surf_loc']) > (10 * params_out['surf_std'])):

                # either no crossing between the two models,
                # or that crossing is way off, and should be ignored
                params_out['trans_mult'] = 10
                params_out['decay_param'] = -zero_val

            else:
                transition_z_new = surf_col_z[sign_changes_i[0]]
                transition_mult_new = (
                    transition_z_new - params_out['surf_loc']) / params_out['surf_std']

                # hooray! we have somewhat accurately estimated turbidity
                params_out['trans_mult'] = transition_mult_new
                params_out['decay_param'] = decay_param

    ##############################    REFINING BATHY PEAK     ################################
    # now that we have a more solid estimate of the base of the bathy peak (turbidity/decay)
    # lets improve our estimate of the bathy model, if there is one

    # if theres a bathy peak
    if bathy_pk_df.shape[0] > 0:

        # if theres turbidity modeled remove that base from consideration
        valid_turb = (params_out['trans_mult'] != 10) & (
            params_out['decay_param'] != -zero_val)

        if valid_turb:
            turb_intens_new = turbid_model_z[sign_changes_i[0]]
            bathy_base = turb_intens_new * \
                np.exp(decay_param * depth[bathy_range_i])

        else:
            bathy_base = np.zeros_like(depth[bathy_range_i])

        bathy_hist_prom = hist[bathy_range_i] - bathy_base

        # isolate the histogram values and fit

        # check that the bathy peak is well defined enough to try a fit process
        # requires more than 1 non zero bin
        # maybe other requirements im missing
        p_init_b = [bathy_pk.prom_scaling,
                    bathy_pk.depth, bathy_pk.sigma_est]
        [_bathy_prom_scale, _bathy_loc, _bathy_sigma] = p_init_b
        try:
            [_bathy_prom_scale, _bathy_loc, _bathy_sigma], _ = curve_fit(f=gauss,
                                                                         xdata=depth[bathy_range_i],
                                                                         ydata=bathy_hist_prom,
                                                                         p0=p_init_b,
                                                                         bounds=([zero_val, params_out['surf_loc'], zero_val],
                                                                                 [3*_surf_mag_scale, depth.max(), np.inf]))

        except:
            # curve fit can fail under a lot of common cases like 1 bin peaks etc
            # just catch them all instead of accounting for each for now
            [_bathy_prom_scale, _bathy_loc, _bathy_sigma] = p_init_b

        params_out['bathy_loc'] = _bathy_loc
        params_out['bathy_std'] = _bathy_sigma
        params_out['bathy_prom'] = _bathy_prom_scale / \
            np.sqrt(2 * np.pi * _bathy_sigma**2)

    ############################ CONFIDENCE OF BATHY PEAK ############################
    # One way we can measure the confidence of our bathy peak is how it compares
    # to the variations about the rest of the subsurface model
    # eg check if the peak is on the order of the rest of the subsurface variations

    # careful, using all subsurface data might skew things when we really just care
    # about the region with valid data

    # actually trying something a little simpler
    # quality_flag = 2 : only 1 isolated subsurface peak
    # quality_flag = 3 : multiple subsurface peaks -> check bathy_conf

    # bathy_conf = ratio of bathy peak prominence to next most prominent peak
    # still undefined if only 1 bathy peak - do something else for these

    # if theres a bathy peak
    if bathy_pk_df.shape[0] > 1:
        # trying a more naive approach
        # consider any peaks further than 1m in either direction
            # to help prevent cases of spread bathy signal falling in separate bins

        non_bathy_pks = bathy_pk_df.iloc[1:]
        comparison_peaks = non_bathy_pks.loc[np.abs(non_bathy_pks.depth - bathy_pk.depth) > 1, :]

        if comparison_peaks.shape[0] == 0:
            # no subsurface peaks 1m+ away from probably bathy peak
            quality_flag = 2 # good surface, effectively 1 bathy
            bathy_quality_ratio = -1

        else:
           quality_flag = 3 # good surface, multiple bathy, check quality ratio
           bathy_quality_ratio = bathy_pk.prominences / comparison_peaks.prominences.iloc[0]

    elif bathy_pk_df.shape[0] == 1:
        quality_flag = 2 # good surface, 1 bathy
        bathy_quality_ratio = -1 # do something here

    else:
        # recall if the surface peak is badly defined we wouldnt get to this point
        quality_flag = 1 # good surface, no bathy
        bathy_quality_ratio = -1

    # old approach for smoothed subsurface noise check
    # # get subsurface model residuals
    # # only up to deepest bin with data, though, so empty window doesnt skew results
    # test_index = np.full((len(hist),), False)
    # # initialize
    # test_index[transition_i:(np.argwhere(hist > 0).flatten()[-1])] = True
    # # use subsurf
    # # test_index[bathy_range_i] = False
    # # remove bathy peak section
    # test_index = np.argwhere(test_index).flatten()
    # # reindex to integers

    # # require test section to have at least a few valid bins
    # if len(test_index) < 3:
    #     bathy_quality_ratio = -1

    # else:
    #     # estimate subsurface error
    #     # subsurface_model = histogram_model(depth_bins=depth[test_index], **params_out)
    #     # use smoothed data instead

    #     subsurface_fitted = gaussian_filter1d(
    #         hist[test_index], 1/z_bin_approx)

    #     residuals = subsurface_fitted - hist[test_index]

    #     mae = np.mean(np.abs(residuals))

    #     bathy_quality_ratio = params_out['bathy_prom'] / mae

    bounds = pd.DataFrame([lower_bound, upper_bound],
                          index=['lower', 'upper']).T

    return params_out, bounds, quality_flag, bathy_quality_ratio


class Waveform:
    """Class for managing psuedowaveform data. Will approximate model parameters upon initialization.

    Attributes:
        data : array of histogram data input

        depth_bin_edges : histogram bin edges input 

        depths : center depths of histogram bins

        peaks : dataframe of info relating to peaks of input histogram

        bounds : dataframe with upper and lower bounds for fitting process

        quality_flag : approx of confidence 
            (-2:no peaks, -1:not enough data, 0:not set, 1:surface only, 2:strong surface and weak bathy peaks, 3:strong surface and bathy peaks)

        model : Dataframe of histogram model components and output. Includes columns for fitted  model output, which will be nan-filled until the fit() method is called.

        model_interp : Dataframe of histogram model components and output, interpolated to approximately 0.01m depth to help with visualization and plotting. Includes columns for fitted model output, which will be nan-filled until the fit() method is called.

        fitted : Has this waveform been fitted and corresponding dataframe columns filled? Boolean, default false.

    Methods:
        fit() : Non-linear least squares optimization to fit refine model to original data.

        show() : Plot histogram and model data.

    """

    def __init__(self, histogram, depth_bin_edges,
                 fit=False, sat_check=False, verbose=False):

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

        # has this been waveform model been fitted?
        self.fitted = False

        # get peak data about this waveform
        self.peaks = get_peak_info(self.data, self.depth, verbose=verbose)

        # ################ Afterpulsing check / removal #################

        # code can be simplified better but I am on a deadline and tired
        # note that each known AP is only allowed to be removed once

        # initialize dataframe of afterpulse peaks
        self.sat_peaks = pd.DataFrame(0, index=[], columns=self.peaks.columns)

        # save the original hist in case you need to modify it (AP, etc)
        self.data_original = histogram

        # initial sat flag (# of saturated peaks)
        self.sat_flag = 0

        # if peaks identified and contains saturation flagged photons
        if sat_check and (self.peaks.shape[0] > 0):

            main_peak = self.peaks.iloc[0]
            bathy_pk_df = self.peaks.loc[self.peaks.depth > main_peak.depth]

            # # any peaks matching saturation parameters get skipped and saved for later analysis (turbidity)
            sat_idx = []
            icheck_in = []  # list of afterpulses already been removed
            data_no_ap = self.data.copy()
            # only checking top 6 non-surface peaks (or however many fewer than that)
            for i in range(0, min(bathy_pk_df.shape[0], 11)):
                # test peak
                pkt = bathy_pk_df.iloc[i]
                check, icheck = is_peak_ap(pkt.heights, pkt.depth,
                                           main_peak.heights, main_peak.depth)
                # print(check, pkt.heights, pkt.depth, main_peak.heights, main_peak.depth)
                if check and (icheck not in icheck_in):
                    # remove peak from histogram
                    left_i = np.int64(
                        np.floor(pkt.i - 2.5 * pkt.sigma_est_left_i))
                    right_i = np.int64(
                        np.ceil(pkt.i + 2.5 * pkt.sigma_est_right_i))
                    data_no_ap[left_i:right_i+1] = np.nan

                    # save peak info
                    sat_idx.append(pkt.name)
                    icheck_in.append(icheck)

            # interpolate across nan values where sat peaks used to be
            data_no_ap = pd.Series(data_no_ap).interpolate(
                method='linear').values

            # one more pass for checking peaks
            # helpful in case an afterpulsing peak is missed due to the
            # peak distance requirement in the peak finder,
            # but leftover as a clear afterpulse

            peaks2 = get_peak_info(data_no_ap, self.depth, verbose=verbose)
            main_peak2 = peaks2.iloc[0]
            bathy_pk_df2 = peaks2.loc[peaks2.depth > main_peak2.depth]

            sat_idx2 = []
            data_no_ap2 = data_no_ap.copy()
            for i in range(0, min(bathy_pk_df2.shape[0], 11)):
                pkt = bathy_pk_df2.iloc[i]
                check, icheck = is_peak_ap(pkt.heights, pkt.depth,
                                           main_peak2.heights, main_peak2.depth)
                # print(check, pkt.heights, pkt.depth, main_peak.heights, main_peak.depth)
                if check and (icheck not in icheck_in):
                    # remove peak from histogram
                    left_i = np.int64(
                        np.floor(pkt.i - 2.5 * pkt.sigma_est_left_i))
                    right_i = np.int64(
                        np.ceil(pkt.i + 2.5 * pkt.sigma_est_right_i))
                    data_no_ap2[left_i:right_i+1] = np.nan

                    # save peak info
                    sat_idx2.append(pkt.name)

            # interpolate across nan values where sat peaks used to be
            data_no_ap2 = pd.Series(data_no_ap2).interpolate(
                method='linear').values

            # combine lists of saturation peaks
            sat_comb = pd.concat(
                [bathy_pk_df.loc[sat_idx, :], bathy_pk_df2.loc[sat_idx2, :]], ignore_index=True, axis=0)
            self.sat_peaks = sat_comb

            # sat flag is proportional to the number of peaks that match sat conditions
            self.sat_flag = sat_comb.shape[0]

            # save off original data and update histogram/peaks
            self.data = data_no_ap2

            self.peaks = peaks2.drop(sat_idx2)

        ###########################################################

        # get hist model parameters, fitting bounds, and qualitative quality flag
        params_est, self.bounds, self.quality_flag, self.bathy_quality_ratio = estimate_model_params(
            self.data, self.depth, self.peaks, verbose=verbose)

        params_est = pd.Series(params_est, name='initial')
        params_fit = pd.Series(np.nan, index=params_est.index, name='fitted')

        # set up model parameters alongside a future column for fitted parameters
        self.params = pd.DataFrame([params_est, params_fit]).T

        ###### Model Output ######
        # organize dataframe (_f : fitted model)
        self.model = pd.DataFrame(np.nan, index=np.arange(len(self.depth)),
                                  columns=['input', 'depth', 'output', 'noise', 'surface', 'bathy', 'turbidity', 'output_f', 'noise_f', 'surface_f', 'bathy_f', 'turbidity_f'])

        # statistical model results
        self.model.input = self.data
        self.model.depth = self.depth
        self.model.output = histogram_model(self.depth, **self.params.initial)

        # model components
        self.model.noise = noise(self.depth, **self.params.initial)
        self.model.surface = surface(self.depth, **self.params.initial)
        self.model.bathy = bathy(self.depth, **self.params.initial)
        self.model.turbidity = turbidity(self.depth, **self.params.initial)

        ###### Interpolated Model Output ######
        # interpolate to higher resolution for plotting and fitting
        # gives approx 0.01m vertical resolution, but not exact (depending on orig data)
        depth_interp = np.linspace(self.depth.min(),
                                   self.depth.max(),
                                   np.int64(np.abs(self.depth.max()-self.depth.min()) / 0.01))

        self.model_interp = pd.DataFrame(np.nan, index=np.arange(len(depth_interp)),
                                         columns=['input', 'depth', 'output', 'noise', 'surface', 'bathy', 'turbidity', 'output_f', 'noise_f', 'surface_f', 'bathy_f', 'turbidity_f'])

        # statistical model results
        self.model_interp.depth = depth_interp
        self.model_interp.input = np.interp(
            self.model_interp.depth, self.depth, self.data)
        self.model_interp.output = histogram_model(
            self.model_interp.depth, **self.params.initial)

        # model components
        self.model_interp.noise = noise(
            self.model_interp.depth, **self.params.initial)
        self.model_interp.surface = surface(
            self.model_interp.depth, **self.params.initial)
        self.model_interp.bathy = bathy(
            self.model_interp.depth, **self.params.initial)
        self.model_interp.turbidity = turbidity(
            self.model_interp.depth, **self.params.initial)

        if fit == True:
            _ = self.fit()

        # placeholder for the segments of profile data
        self.profile = None
        self.sat_check = sat_check

    def __str__(self):
        '''Human readable summary'''
        s = f"""----- PSEUDOWAVEFORM -----
TOTAL PHOTONS: {np.sum(self.data)}
Depth Range : [{min(self.bin_edges):.2f}m, {max(self.bin_edges):.2f}m]
Depth Bin Count : {len(self.bin_edges) - 1}
Peak Count : {self.peaks.shape[0]}
Overall Quality Flag : {self.quality_flag}
Bathymetry Confidence : {self.bathy_quality_ratio}
Fitted? : {self.fitted}

Initial Parameter Estimates:
    Surface Peak Location : {self.params.initial.surf_loc:.2f}m
    Bathy Peak Location : {self.params.initial.bathy_loc:.2f}m
    Surface/Bathy Peak Ratio : {(self.params.initial.surf_prom / self.params.initial.bathy_prom):.2f}
        """
        return s

    def _deconvolve_atlas(self):
        # deconvolves the atlas response at whatever the depth resolution is
        pass

    def fit(self, xtol=1e-6, ftol=1e-6):
        """Attempts to fit the initially estimated histogram model to the original data. Returns dict of new model parameters, but also saves fitted model data to Waveform.params, Wavefrom.model, and Waveform.model_interp attributes for further analysis.

        Args:
            xtol (float, optional): Relative error desired in the approximate solution (passes to scipy least squares). Defaults to 1e-6.
            ftol (float, optional): Relative error desired in the sum of squares (passes to scipy least squares). . Defaults to 1e-6.

        Returns:
            dict: Dictionary of refined model parameters
        """
        if self.quality_flag <= 0:
            # bad window or error setting quality flag, dont fit
            return

        params_fit, _ = curve_fit(histogram_model,
                                  self.model_interp.depth,
                                  self.model_interp.input,
                                  p0=self.params.initial,
                                  bounds=(self.bounds.lower,
                                          self.bounds.upper),
                                  xtol=xtol, ftol=ftol)  # , maxfev=5000)

        # save fitted model output
        self.model.output_f = histogram_model(self.model.depth, *params_fit)
        self.model.noise_f = noise(self.model.depth, *params_fit)
        self.model.bathy_f = bathy(self.model.depth, *params_fit)
        self.model.surface_f = surface(self.model.depth, *params_fit)
        self.model.turbidity_f = turbidity(self.model.depth, *params_fit)

        # save fitted model output at interpolated resolution
        self.model_interp.output_f = histogram_model(
            self.model_interp.depth, *params_fit)
        self.model_interp.noise_f = noise(self.model_interp.depth, *params_fit)
        self.model_interp.bathy_f = bathy(self.model_interp.depth, *params_fit)
        self.model_interp.surface_f = surface(
            self.model_interp.depth, *params_fit)
        self.model_interp.turbidity_f = turbidity(
            self.model_interp.depth, *params_fit)

        self.params.fitted = params_fit
        self.fitted = True

        # evaluate error and store result
        # not done yet

        return params_fit

    def add_profile(self, profile):
        self.profile = profile

    def show(self, ylim=None, logplot=True, hide_fitted=False):
        """Visualize histogram and model data in a plot.

        Args:
            hide_fitted (bool, optional): Whether or not to hide the fitted model from the plot if it's already been calculated. Defaults to False.

        Returns:
            f (pyplot figure): Matplolib Figure with data.
        """

        # plotting just statistical model
        if (self.fitted == False) or (hide_fitted == True):
            f = plt.figure(figsize=[4.8, 6.4])
            ax1 = f.gca()

        # plot fitted model too
        else:
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=[9.6, 6.4])

            ax2.plot(self.model_interp.noise_f + self.model_interp.surface_f + self.model_interp.turbidity_f +
                     self.model_interp.bathy_f, self.model_interp.depth, 'b.', label='Seafloor_F')

            ax2.plot(self.model_interp.noise_f + self.model_interp.surface_f,
                     self.model_interp.depth, 'r.', label='Surface_F')

            ax2.plot(self.model_interp.noise_f + self.model_interp.turbidity_f,
                     self.model_interp.depth, 'g.', label='Turbidity_F')

            ax2.plot(self.model_interp.noise_f,
                     self.model_interp.depth, marker='.', linestyle='', color='grey', label='Noise_F')

            # plotting model input/output
            ax2.plot(self.model.input, self.model.depth,
                     'kx-', linewidth=2, label='Input Histogram')

            ax2.plot(self.model.output_f, self.model.depth, marker='s', linestyle='-',
                     color='darkorange', linewidth=2, alpha=0.75, label='Fitted Histogram Model')

            ax2.legend(loc='lower right')
            ax2.grid('major')
            # ax2.invert_yaxis()
            # ax2.set_ylabel('Depth (m)')
            ax2.set_xlabel('Photons (count)')
            ax2.set_title('Fitted Model')

            if logplot:
                ax2.set_xscale('log')
                ax2.set_xlim([1e-1, ax2.get_xlim()[1]])

        # plotting model components (interpolated for clarity)
        ax1.plot(self.model_interp.noise + self.model_interp.surface + self.model_interp.turbidity +
                 self.model_interp.bathy, self.model_interp.depth, 'b.', label='Seafloor')

        surface_mask = self.model_interp.surface > 0
        ax1.plot(self.model_interp.noise[surface_mask] + self.model_interp.surface[surface_mask],
                 self.model_interp.depth[surface_mask], 'r.', label='Surface')

        turbidity_mask = self.model_interp.turbidity > 0

        ax1.plot(self.model_interp.noise[turbidity_mask] + self.model_interp.turbidity[turbidity_mask],
                 self.model_interp.depth[turbidity_mask], 'g.', label='Turbidity')

        ax1.plot(self.model_interp.noise,
                 self.model_interp.depth, marker='.', linestyle='', color='grey', label='Noise')

        # plotting model input/output
        # extra index prevents mess when plotting log
        ax1.plot(self.model.input[self.model.input > 0], self.model.depth[self.model.input > 0],
                 'kx-', linewidth=2, label='Input Histogram')

        ax1.plot(self.model.output[self.model.output > 0], self.model.depth[self.model.output > 0], marker='.', linestyle='-',
                 color='darkorange', linewidth=1, alpha=0.9, label='Model Estimate')

        ax1.legend(loc='lower right')
        ax1.grid('major')
        ax1.invert_yaxis()

        ax1.set_ylabel('(-) Elevation EGM08 (m)')
        ax1.set_xlabel('Photons (count)')
        ax1.set_title('Model Estimate')

        if logplot:
            ax1.set_xscale('log')
            ax1.set_xlim([1e-1, ax1.get_xlim()[1]])

        if ~isinstance(ylim, type(None)):
            ax1.set_ylim(ylim)

        plt.tight_layout()

        return f


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from icesat2 import Profile
    import os
    from geo_aoi import GeoAOI

    # Demo Sample Waveform
    # # increasing in depth, decreasing in elevation
    # hist = np.array([1, 2, 10, 25, 8, 5, 5, 4, 8, 6, 5, 4, 4, 2, 1])
    # # depth bins must be increasing L-R
    # depth_bin_edges = np.linspace(-5, 15, len(hist)+1)
    # w = Waveform(hist, depth_bin_edges)
    # print(w)
    # w.fit()
    # # w.show()

    # sample aoi from GBR
    aoi = GeoAOI.from_points(
        x=[149.4, 149.6, 149.6, 149.4], 
        y=[-19.8, -19.8, -19.797, -19.797])

    p = Profile.load_sample(aoi=aoi)

    # sample aoi for NC
    # aoi = GeoAOI.from_points(
    #     x=[-76.7, -74.2, -74.2, -76.7], y=[34.4, 34.4, 34.55, 34.55])

    # base = '/Users/jonathan/Documents/Research/PAPER_Density_Heuristic_IS2_Bathy/nc_is2_2019'
    # f = 'ATL03_20191206003613_10710502_005_01.h5'
    # p = Profile.from_h5(os.path.join(base, f), 'gt1l', aoi=aoi)

    # organizing data as if it were a chunk of photons histogrammed
    df = p.data

    # test range of photons
    # df = p.data.loc[(p.data.lat > 35.43) & (p.data.lat < 35.5), :]
    # df = p.data.loc[(p.data.lat > -19.8) & (p.data.lat < -19.795), :]

    # pseudowaveform of data
    zres = 0.1
    bounds = [-10, 50]
    z_bin_edges = np.linspace(bounds[0], bounds[1], np.int64(
        np.abs(bounds[1] - bounds[0])/zres)+1)
    hist, _ = np.histogram(-(df.height - df.geoid_z), z_bin_edges)
    depths = z_bin_edges[:-1] + zres

    # # interpolating zero bin values
    # idx = np.where(hist != 0)
    # hist_new = np.interp(depths, depths[idx], hist[idx])
    # dont like this assumption, reverting back
    hist_new = hist

    w = Waveform(hist_new, z_bin_edges)
    w.add_profile(p)
    # only run these if there are any sat photons in chunk

    test_peak_mag = w.params.initial.bathy_prom  # should use magnitude instead
    test_peak_loc = w.params.initial.bathy_loc
    surf_peak_mag = w.params.initial.surf_prom
    surf_peak_loc = w.params.initial.surf_loc

    print('')
