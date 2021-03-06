from __future__ import print_function

import argparse
import cartopy.crs as ccrs
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from cartopy.util import add_cyclic_point

from climate_indices.scand import (calculate_daily_region_anomalies,
                                   calculate_seasonal_eof,
                                   calculate_scand_pc_index)


DEFAULT_TIME_FIELD = 'time'
DEFAULT_LAT_FIELD = 'lat'
DEFAULT_LON_FIELD = 'lon'
DEFAULT_HGT_FIELD = 'hgt'


DEFAULT_START_YEAR = 1950
DEFAULT_END_YEAR = 2000


VALID_SEASONS = ['DJF', 'MAM', 'JJA', 'SON']

FIGURE_SIZE = (7, 5)
CMAP = plt.cm.RdBu_r
CENTRAL_LATITUDE = 90.0
CENTRAL_LONGITUDE = -80.0


def read_tibaldi_data(datafile):
    data = np.genfromtxt(datafile, delimiter=',')

    times = np.array([datetime.datetime(
        int(data[i, 0]), int(data[i, 1]), int(data[i, 2]))
        for i in range(data.shape[0])])
    index = data[:, -1]

    return times, index


def plot_seasonal_eofs(seasonal_eofs, evr, mode=2,
                       output_file=None, show_plot=True,
                       lat_field=DEFAULT_LAT_FIELD,
                       lon_field=DEFAULT_LON_FIELD,
                       hgt_field=DEFAULT_HGT_FIELD,
                       n_levels=7):
    lat = seasonal_eofs[lat_field].values
    lon = seasonal_eofs[lon_field].values
    cyclic_lon = np.full(np.size(lon) + 1, 360.)
    cyclic_lon[:-1] = lon

    lon_grid, lat_grid = np.meshgrid(cyclic_lon, lat)

    eof_vals = add_cyclic_point(np.squeeze(seasonal_eofs.values))

    amin = np.abs(np.min(eof_vals))
    amax = np.abs(np.max(eof_vals))
    max_contour = amax if amax > amin else amin
    levels = np.linspace(-max_contour, max_contour, n_levels)

    fig = plt.figure(figsize=FIGURE_SIZE)
    proj = ccrs.Orthographic(central_latitude=CENTRAL_LATITUDE,
                             central_longitude=CENTRAL_LONGITUDE)
    ax = fig.add_subplot(111, projection=proj)

    ax.coastlines()
    ax.set_global()

    ax.set_title(r'Mode {:d} ({:.2f}%)'.format(mode + 1, 100.0 * evr))

    cs = ax.contourf(lon_grid, lat_grid, eof_vals,
                     contour_levels=levels, cmap=CMAP,
                     transform=ccrs.PlateCarree())

    fig.colorbar(cs)

    if output_file is not None and output_file:
        plt.savefig(output_file)

    if show_plot:
        plt.show()


def get_time_masks(x_times, y_times):
    all_times = np.unique(np.concatenate([x_times, y_times]))
    x_mask = np.zeros(x_times.shape, dtype=bool)
    y_mask = np.zeros(y_times.shape, dtype=bool)

    for t in all_times:
        if t in x_times and t in y_times:
            x_idx = np.nonzero(x_times == t)[0][0]
            x_mask[x_idx] = True
            y_idx = np.nonzero(y_times == t)[0][0]
            y_mask[y_idx] = True

    return x_mask, y_mask


def get_correlation_coeff(x_times, x_vals, y_times, y_vals):
    x_mask, y_mask = get_time_masks(x_times, y_times)
    x_data = x_vals[x_mask]
    y_data = y_vals[y_mask]

    data = np.vstack([x_data, y_data])
    corr_coeff = np.corrcoef(data, rowvar=True)[0, 1]

    return corr_coeff


def plot_scand_index(index, tibaldi_times=None, tibaldi_index=None,
                     output_file=None, show_plots=True,
                     time_field=DEFAULT_TIME_FIELD, n_years=10):
    years = index[time_field].dt.year.values
    months = index[time_field].dt.month.values
    days = index[time_field].dt.day.values
    times = np.array(
        [datetime.datetime(int(years[i]), int(months[i]), int(days[i]))
         for i in range(years.shape[0])])
    index_vals = np.squeeze(index.values)

    last_year = int(np.max(years))
    first_year = int(last_year - n_years + 1)
    valid_years = np.arange(first_year, last_year + 1)
    mask = np.zeros(times.shape, dtype=bool)
    for i, t in enumerate(times):
        mask[i] = t.year in valid_years

    valid_times = times[mask]
    valid_index = index_vals[mask]

    fig = plt.figure(figsize=FIGURE_SIZE)

    ax = fig.add_subplot(111)

    if tibaldi_times is not None and tibaldi_index is not None:
        corr_coeff = get_correlation_coeff(
            times, index_vals, tibaldi_times, tibaldi_index)
        mask = np.zeros(tibaldi_times.shape, dtype=bool)
        for i, t in enumerate(tibaldi_times):
            mask[i] = t.year in valid_years
        normalization = tibaldi_index.std()
        ax.plot(
            tibaldi_times[mask],
            tibaldi_index[mask] / normalization, 'r--',
            label='Normalized Tibaldi index (r = {:.2f})'.format(corr_coeff))

    ax.plot(valid_times, valid_index, 'b-', label='SCAND index')

    ax.grid(ls='--', color='gray', alpha=0.7)

    years = mdates.YearLocator(2)
    months = mdates.MonthLocator()
    yearsFmt = mdates.DateFormatter('%Y')

    ax.set_xlabel('Date')
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    ax.set_ylabel('Index')

    ax.tick_params(axis='x', labelsize=8, labelrotation=45)
    ax.tick_params(axis='y', labelsize=8)

    ax.legend()

    if output_file:
        plt.savefig(output_file)

    if show_plots:
        plt.show()

    plt.close()


def write_index_values(index, output_file=None,
                       time_field=DEFAULT_TIME_FIELD):
    fields = [('year', '%d'),
              ('month', '%d'),
              ('day', '%d'),
              ('index', '%14.8e')]

    header = ','.join([f[0] for f in fields])
    fmt = ','.join([f[1] for f in fields])

    n_fields = len(fields)
    n_samples = index.shape[0]

    data = np.empty((n_samples, n_fields), dtype=index.values.dtype)
    data[:, 0] = index[time_field].dt.year.values
    data[:, 1] = index[time_field].dt.month.values
    data[:, 2] = index[time_field].dt.day.values
    data[:, 3] = np.squeeze(index.values)

    if output_file is None or not output_file:
        print('# ' + header)
        for i in range(n_samples):
            line = '  {:d},{:d},{:d},{:14.8e}'.format(
                int(data[i, 0]), int(data[i, 1]), int(data[i, 2]),
                data[i, 3])
            print(line)
    else:
        np.savetxt(output_file, data, header=header, fmt=fmt)


def parse_cmd_line_args():
    parser = argparse.ArgumentParser(
        description='Plot daily SCAND index')

    parser.add_argument(
        'datafile', help='datafile containing 500 hPa geopotential heights')
    parser.add_argument(
        '--time-field', dest='time_field', default=DEFAULT_TIME_FIELD,
        help='name of variable corresponding to time in input datafile')
    parser.add_argument(
        '--lat-field', dest='lat_field', default=DEFAULT_LAT_FIELD,
        help='name of variable corresponding to latitude in input datafile')
    parser.add_argument(
        '--lon-field', dest='lon_field', default=DEFAULT_LON_FIELD,
        help='name of variable corresponding to longitude in input datafile')
    parser.add_argument(
        '--hgt-field', dest='hgt_field', default=DEFAULT_HGT_FIELD,
        help='name of variable corresponding to height in input datafile')
    parser.add_argument(
        '--start-year', dest='start_year', type=int,
        default=DEFAULT_START_YEAR,
        help='start year of reference period')
    parser.add_argument(
        '--end-year', dest='end_year', type=int,
        default=DEFAULT_END_YEAR,
        help='end year of reference period')
    parser.add_argument(
        '--season', dest='season', choices=VALID_SEASONS,
        default='DJF', help='season to compute EOFs in')
    parser.add_argument(
        '--standardize', dest='standardize', action='store_true',
        help='calculate EOFs for standardized anomalies')
    parser.add_argument(
        '--index-output-file', dest='index_output_file',
        default='', help='name of file to write index to')
    parser.add_argument(
        '--index-plot-output-file', dest='index_plot_output_file',
        default='', help='name of file to write index plot to')
    parser.add_argument(
        '--eof-output-file', dest='eof_output_file',
        default='', help='name of file to write EOF to')
    parser.add_argument(
        '--eof-plot-output-file', dest='eof_plot_output_file',
        default='', help='name of file to write EOF plots to')
    parser.add_argument(
        '--no-show-plots', dest='no_show_plots', action='store_true',
        help='do not show plots')
    parser.add_argument(
        '--tibaldi-datafile', dest='tibaldi_datafile',
        default='', help='datafile containing daily Tibaldi index values')

    return parser.parse_args()


def main():
    args = parse_cmd_line_args()

    if args.tibaldi_datafile:
        tibaldi_times, tibaldi_index = read_tibaldi_data(
            args.tibaldi_datafile)
    else:
        tibaldi_times = None
        tibaldi_index = None

    with xr.open_dataset(args.datafile) as ds:
        hgt_data = ds[args.hgt_field]

        ref_data = hgt_data.where(
            (hgt_data[args.time_field].dt.year >= args.start_year) &
            (hgt_data[args.time_field].dt.year <= args.end_year),
            drop=True)

        ref_anom_data, clim = calculate_daily_region_anomalies(ref_data)
        if args.standardize:
            ref_anom_data = ref_anom_data / ref_anom_data.std(args.time_field)

        seasonal_eofs = calculate_seasonal_eof(
            ref_anom_data, season=args.season, time_field=args.time_field,
            lat_field=args.lat_field, hgt_field=args.hgt_field)

        if args.eof_output_file:
            seasonal_eofs['eofs'].to_netcdf(args.eof_output_file)

        if not args.no_show_plots or args.eof_plot_output_file:
            plot_seasonal_eofs(
                seasonal_eofs['eofs'],
                seasonal_eofs['explained_variance_ratio'],
                output_file=args.eof_plot_output_file,
                show_plot=(not args.no_show_plots))

        anom_data, _ = calculate_daily_region_anomalies(
            hgt_data, climatology=clim)
        if args.standardize:
            anom_data = anom_data / ref_anom_data.std(args.time_field)

        index = calculate_scand_pc_index(
            anom_data, seasonal_eofs['eofs'],
            clim_start_year=args.start_year,
            clim_end_year=args.end_year, time_field=args.time_field,
            lat_field=args.lat_field)

        write_index_values(index, output_file=args.index_output_file,
                           time_field=args.time_field)

        if not args.no_show_plots or args.index_plot_output_file:
            plot_scand_index(index, tibaldi_times=tibaldi_times,
                             tibaldi_index=tibaldi_index,
                             output_file=args.index_plot_output_file,
                             show_plots=(not args.no_show_plots),
                             time_field=args.time_field)


if __name__ == '__main__':
    main()
