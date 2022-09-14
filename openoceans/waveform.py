from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd

from scipy.signal import medfilt, peak_widths, peak_prominences, find_peaks
from scipy.stats import exponnorm, norm, skewnorm, expon
from scipy.optimize import curve_fit

# to do
# - convert turbid intensity back to a transition mult to prevent fitting errors
# - account for skewed bathymetry peak
# - try out slope based peak finding
# - atlas response deconvolution
# - add error calculations to the waveform


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
          decay_param, turb_intens,
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
        turb_intens (float): Max turbidity (transition between surface/column model), in photons.
        noise_above (float): Air noise rate, in photons/bin.
        noise_below (float): Subsurface noise rate, in photons/bin.
        bathy_prom (_type_): Not used, provided for consistency of input across model components.
        bathy_loc (_type_): Not used, provided for consistency of input across model components.
        bathy_std (_type_): Not used, provided for consistency of input across model components.

    Returns:
        array: Noise model output, matching shape of input depth array.
    """

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
    """Gaussian model for the seafloor return. Currently intended to be added to the noise model.

    Args:
        depth_bins (array of float): Centers of depth histogram bins, in meters.
        surf_prom (float): Not used, provided for consistency of input across model components.
        surf_loc (float): Not used, provided for consistency of input across model components.
        surf_std (float): Not used, provided for consistency of input across model components.
        decay_param (float): Not used, provided for consistency of input across model components.
        turb_intens (float): Not used, provided for consistency of input across model components.
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
            decay_param, turb_intens,
            noise_above, noise_below,
            bathy_prom, bathy_loc, bathy_std):
    """Gaussian model for the water surface return. Returns values from the top of the water surface, to the start of the column turbidity model below surface, as defined by turbidity/decay model inputs. Currently intended to be added with other model components (noise, turbidity).

    Args:
        depth_bins (array of float): Centers of depth histogram bins, in meters.
        surf_prom (float): Peak prominence of the surface return, in photons.
        surf_loc (float): Location of the surface return, in meters.
        surf_std (float): Standard deviation of the surface return gaussian model.
        decay_param (float): Decay parameter of the turbidity exponential model component.
        turb_intens (float): Max turbidity (transition between surface/column model), in photons.
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
    """Exponential decay model for water column / turbidity. Returns values from the bottom of the water surface to the max depth of the waveform. Currently intended to be added with other model components (noise, surface).

    Args:
        depth_bins (array of float): Centers of depth histogram bins, in meters.
        surf_prom (float): Peak prominence of the surface return, in photons.
        surf_loc (float): Location of the surface return, in meters.
        surf_std (float): Standard deviation of the surface return gaussian model.
        decay_param (float): Decay parameter of the turbidity exponential model component.
        turb_intens (float): Max turbidity (transition between surface/column model), in photons.
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


def histogram_model(depth_bins, surf_prom, surf_loc, surf_std,
                    decay_param, turb_intens,
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
        turb_intens (float): Max turbidity (transition between surface/column model), in photons.
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
                    decay_param, turb_intens,
                    noise_above, noise_below,
                    bathy_prom, bathy_loc, bathy_std)

    y_surface = surface(depth_bins, surf_prom, surf_loc, surf_std,
                        decay_param, turb_intens,
                        noise_above, noise_below,
                        bathy_prom, bathy_loc, bathy_std)

    y_bathy = bathy(depth_bins, surf_prom, surf_loc, surf_std,
                    decay_param, turb_intens,
                    noise_above, noise_below,
                    bathy_prom, bathy_loc, bathy_std)

    y_turbidity = turbidity(depth_bins, surf_prom, surf_loc, surf_std,
                            decay_param, turb_intens,
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

    # is prominence truly the best way to sort peaks?

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
    if ~isinstance(peaks, pd.DataFrame):
        peaks = get_peak_info(hist, depth)

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

    return params_out, bounds, quality_flag


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

    def __init__(self, histogram, depth_bin_edges, fit=False, verbose=False):

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

        # get hist model parameters, fitting bounds, and qualitative quality flag
        params_est, self.bounds, self.quality_flag = estimate_model_params(
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

        if fit==True:
            _ = self.fit()

        # placeholder for the segments of profile data
        self.profile = None
        self.sat_ph_any = None

    def __str__(self):
        '''Human readable summary'''
        s = f"""----- PSEUDOWAVEFORM -----
TOTAL PHOTONS: {np.sum(self.data)}
Depth Range : [{min(self.bin_edges):.2f}m, {max(self.bin_edges):.2f}m]
Depth Bin Count : {len(self.bin_edges) - 1}
Peak Count : {self.peaks.shape[0]}
Parameter Quality Flag : {self.quality_flag}
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

    def set_sat_ph_any(self, flag):
        'true if there are any quality_ph flagged photons in this window'
        self.sat_ph_any = flag

    def fit(self, xtol=1e-6, ftol=1e-6):
        """Attempts to fit the initially estimated histogram model to the original data. Returns dict of new model parameters, but also saves fitted model data to Waveform.params, Wavefrom.model, and Waveform.model_interp attributes for further analysis.

        Args:
            xtol (float, optional): Relative error desired in the approximate solution (passes to scipy least squares). Defaults to 1e-6.
            ftol (float, optional): Relative error desired in the sum of squares (passes to scipy least squares). . Defaults to 1e-6.

        Returns:
            dict: Dictionary of refined model parameters
        """
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
        # to do

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

            ax2.legend()
            ax2.grid('major')
            # ax2.invert_yaxis()
            # ax2.set_ylabel('Depth (m)')
            ax2.set_xlabel('Photons (count)')
            ax2.set_title('Fitted Model')

        # plotting model components (interpolated for clarity)
        ax1.plot(self.model_interp.noise + self.model_interp.surface + self.model_interp.turbidity +
                 self.model_interp.bathy, self.model_interp.depth, 'b.', label='Seafloor')

        ax1.plot(self.model_interp.noise + self.model_interp.surface,
                 self.model_interp.depth, 'r.', label='Surface')

        ax1.plot(self.model_interp.noise + self.model_interp.turbidity,
                 self.model_interp.depth, 'g.', label='Turbidity')

        ax1.plot(self.model_interp.noise,
                 self.model_interp.depth, marker='.', linestyle='', color='grey', label='Noise')

        # plotting model input/output
        # extra index prevents mess when plotting log
        ax1.plot(self.model.input[self.model.input > 0], self.model.depth[self.model.input > 0],
                 'kx-', linewidth=2, label='Input Histogram')

        ax1.plot(self.model.output[self.model.output > 0], self.model.depth[self.model.output > 0], marker='s', linestyle='-',
                 color='darkorange', linewidth=2, alpha=0.75, label='Model Estimate')

        ax1.legend()
        ax1.grid('major')
        ax1.invert_yaxis()
        
        ax1.set_ylabel('Depth (m)')
        ax1.set_xlabel('Photons (count)')
        ax1.set_title('Initial Model')

        if logplot:
            ax1.set_xscale('log')
            ax1.set_xlim([1e-1, ax1.get_xlim()[1]])

        if ~isinstance(ylim, type(None)):
            ax1.set_ylim(ylim)

        plt.tight_layout()

        return f


if __name__ == "__main__":
    import numpy as np

    # increasing in depth, decreasing in elevation
    hist = np.array([1, 2, 10, 25, 8, 5, 5, 4, 8, 2, 2, 1])

    # depth bins must be increasing L-R
    depth_bin_edges = np.linspace(-5, 15, len(hist)+1)

    w = Waveform(hist, depth_bin_edges)
    print(w)
    w.fit()
    # w.show()

    print('')
