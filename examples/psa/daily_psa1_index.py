from __future__ import print_function

import argparse
import cartopy.crs as ccrs
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from cartopy.util import add_cyclic_point

from climate_indices.psa import (calculate_daily_region_anomalies,
                                 calculate_monthly_region_anomalies,
                                 calculate_seasonal_eof,
                                 calculate_psa1_real_pc_index)


DEFAULT_TIME_FIELD = 'time'
DEFAULT_LAT_FIELD = 'lat'
DEFAULT_LON_FIELD = 'lon'
DEFAULT_HGT_FIELD = 'hgt'


DEFAULT_START_YEAR = 1979
DEFAULT_END_YEAR = 2000


FIGURE_SIZE = (5, 5)
CMAP = plt.cm.RdBu_r
CENTRAL_LATITUDE = -90.0
CENTRAL_LONGITUDE = 0.0


def plot_annual_eofs(seasonal_eofs, evr, mode=2,
                     output_file=None, show_plot=True,
                     lat_field=DEFAULT_LAT_FIELD,
                     lon_field=DEFAULT_LON_FIELD,
                     hgt_field=DEFAULT_HGT_FIELD,
                     n_levels=20):
    lat = seasonal_eofs[lat_field].values
    lon = seasonal_eofs[lon_field].values
    cyclic_lon = np.full(np.size(lon) + 1, 360.)
    cyclic_lon[:-1] = lon

    lon_grid, lat_grid = np.meshgrid(cyclic_lon, lat)

    eof_vals = add_cyclic_point(np.squeeze(seasonal_eofs.values))

    amin = np.min(np.abs(eof_vals))
    amax = np.max(np.abs(eof_vals))
    max_contour = amax if amax > amin else amin

    fig = plt.figure(figsize=FIGURE_SIZE)
    proj = ccrs.Orthographic(central_latitude=CENTRAL_LATITUDE,
                             central_longitude=CENTRAL_LONGITUDE)
    ax = fig.add_subplot(111, projection=proj)

    ax.coastlines()
    ax.set_global()

    cs = ax.contourf(lon_grid, lat_grid, eof_vals, n_levels,
                     vmin=-max_contour, vmax=max_contour,
                     cmap=CMAP,
                     transform=ccrs.PlateCarree())

    ax.set_title(r'Mode {:d} ({:.2f}%)'.format(mode + 1, 100.0 * evr))

    fig.colorbar(cs)

    if output_file is not None and output_file:
        plt.savefig(output_file)

    if show_plot:
        plt.show()


def plot_psa1_index(index, output_file=None, show_plots=True,
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

    ax.plot(valid_times, valid_index, 'b-', label='PSA1 index')

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
        description='Plot daily PSA1 index')

    parser.add_argument(
        'datafile', help='datafile containing geopotential heights')
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

    return parser.parse_args()


def main():
    args = parse_cmd_line_args()

    with xr.open_dataset(args.datafile) as ds:
        hgt_data = ds[args.hgt_field]

        ref_data = hgt_data.where(
            (hgt_data[args.time_field].dt.year >= args.start_year) &
            (hgt_data[args.time_field].dt.year <= args.end_year),
            drop=True)

        ref_anom_data, clim = calculate_daily_region_anomalies(ref_data)
        if args.standardize:
            ref_anom_data = ref_anom_data / ref_anom_data.std(args.time_field)

        annual_eofs = calculate_seasonal_eof(
            ref_anom_data, time_field=args.time_field,
            lat_field=args.lat_field, hgt_field=args.hgt_field)

        if args.eof_output_file:
            annual_eofs['eofs'].to_netcdf(args.eof_output_file)

        if not args.no_show_plots or args.eof_plot_output_file:
            plot_annual_eofs(
                annual_eofs['eofs'],
                annual_eofs['explained_variance_ratio'],
                output_file=args.eof_plot_output_file,
                show_plot=(not args.no_show_plots))

        ref_daily_anom_data, daily_clim = calculate_daily_region_anomalies(
            ref_data)
        daily_anom_data, _ = calculate_daily_region_anomalies(
            hgt_data, climatology=daily_clim)
        if args.standardize:
            daily_anom_data = (daily_anom_data /
                               ref_anom_data.std(args.time_field))

        index = calculate_psa1_real_pc_index(
            daily_anom_data, annual_eofs['eofs'],
            time_field=args.time_field,
            lat_field=args.lat_field)

        write_index_values(index, output_file=args.index_output_file,
                           time_field=args.time_field)

        if not args.no_show_plots or args.index_plot_output_file:
            plot_psa1_index(index,
                            output_file=args.index_plot_output_file,
                            show_plots=(not args.no_show_plots),
                            time_field=args.time_field)


if __name__ == '__main__':
    main()
