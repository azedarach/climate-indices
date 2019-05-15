from __future__ import print_function

import argparse
import cartopy.crs as ccrs
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from climate_indices.ao import (calculate_daily_region_anomalies,
                                calculate_monthly_region_anomalies,
                                calculate_annual_eof, calculate_ao_pc_index)


DEFAULT_TIME_FIELD = 'time'
DEFAULT_LAT_FIELD = 'lat'
DEFAULT_LON_FIELD = 'lon'
DEFAULT_HGT_FIELD = 'hgt'


DEFAULT_START_YEAR = 1979
DEFAULT_END_YEAR = 2000


FIGURE_SIZE = (5, 5)
CMAP = plt.cm.RdBu_r
CENTRAL_LATITUDE = 90.0
CENTRAL_LONGITUDE = -80.0


def read_cpc_data(datafile):
    data = np.genfromtxt(datafile)

    year = np.asarray(data[:, 0], dtype='i8')
    month = np.asarray(data[:, 1], dtype='i8')
    day = np.asarray(data[:, 2], dtype='i8')
    index = data[:, 3]

    n_samples = data.shape[0]

    time = np.array([datetime.datetime(year[i], month[i], day[i])
                     for i in range(n_samples)])

    return time, index


def plot_eofs(eofs, mode=0, output_file=None, show_plot=True):
    fig = plt.figure(figsize=FIGURE_SIZE)
    proj = ccrs.Orthographic(central_latitude=CENTRAL_LATITUDE,
                             central_longitude=CENTRAL_LONGITUDE)
    ax = fig.add_subplot(111, projection=proj)

    ax.coastlines()
    ax.set_global()

    eof_data = eofs[{'mode': mode}].squeeze()
    eof_data.plot.contourf(ax=ax, transform=ccrs.PlateCarree())

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


def plot_ao_index(index, cpc_times=None, cpc_index=None,
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

    ax.plot(valid_times, valid_index, 'b-', label='AO index')

    if cpc_times is not None and cpc_index is not None:
        corr_coeff = get_correlation_coeff(
            times, index_vals, cpc_times, cpc_index)
        mask = np.zeros(cpc_times.shape, dtype=bool)
        for i, t in enumerate(cpc_times):
            mask[i] = t.year in valid_years
        ax.plot(
            cpc_times[mask], cpc_index[mask], 'r--',
            label='CPC AO index (r = {:.2f})'.format(corr_coeff))

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
        description='Plot daily AO index')

    parser.add_argument(
        'datafile', help='datafile containing 1000 hPa geopotential heights')
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
        '--cpc-datafile', dest='cpc_datafile',
        default='', help='datafile containing daily CPC index values')

    return parser.parse_args()


def main():
    args = parse_cmd_line_args()

    if args.cpc_datafile:
        cpc_times, cpc_index = read_cpc_data(args.cpc_datafile)
    else:
        cpc_times = None
        cpc_index = None

    with xr.open_dataset(args.datafile) as ds:
        hgt_data = ds[args.hgt_field]

        ref_data = hgt_data.where(
            (hgt_data[args.time_field].dt.year >= args.start_year) &
            (hgt_data[args.time_field].dt.year <= args.end_year),
            drop=True)

        ref_anom_data, clim = calculate_monthly_region_anomalies(ref_data)

        annual_eofs = calculate_annual_eof(
            ref_anom_data, time_field=args.time_field,
            lat_field=args.lat_field, hgt_field=args.hgt_field)

        if args.eof_output_file:
            annual_eofs['eofs'].to_netcdf(args.eof_output_file)

        if not args.no_show_plots or args.eof_plot_output_file:
            plot_eofs(
                annual_eofs['eofs'], output_file=args.eof_plot_output_file,
                show_plot=(not args.no_show_plots))

        ref_daily_anom_data, daily_clim = calculate_daily_region_anomalies(
            ref_data)
        daily_anom_data, _ = calculate_daily_region_anomalies(
            hgt_data, climatology=daily_clim)

        index = calculate_ao_pc_index(
            daily_anom_data, annual_eofs['eofs'],
            time_field=args.time_field,
            lat_field=args.lat_field,
            normalization=annual_eofs['pcs'].std(args.time_field, ddof=1))

        write_index_values(index, output_file=args.index_output_file,
                           time_field=args.time_field)

        if not args.no_show_plots or args.index_plot_output_file:
            plot_ao_index(index, cpc_times=cpc_times, cpc_index=cpc_index,
                          output_file=args.index_plot_output_file,
                          show_plots=(not args.no_show_plots),
                          time_field=args.time_field)


if __name__ == '__main__':
    main()
