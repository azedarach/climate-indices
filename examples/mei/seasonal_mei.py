from __future__ import print_function

import argparse
import cartopy.crs as ccrs
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from climate_indices.mei import (SEASON_DIM_NAME,
                                 calculate_monthly_anomalies,
                                 calculate_seasonal_eofs,
                                 calculate_seasonal_mei,
                                 fix_phases,
                                 get_seasonal_data, get_season_name,
                                 standardize_values)

DEFAULT_TIME_FIELD = 'time'
DEFAULT_LAT_FIELD = 'lat'
DEFAULT_LON_FIELD = 'lon'
DEFAULT_SLP_FIELD = 'PRMSL_GDS0_MSL'
DEFAULT_SST_FIELD = 'BRTMP_GDS0_SFC'
DEFAULT_UWND_FIELD = 'UGRD_GDS0_HTGL'
DEFAULT_VWND_FIELD = 'VGRD_GDS0_HTGL'
DEFAULT_OLR_FIELD = 'olr'

DEFAULT_VARIABLES = [
    DEFAULT_SLP_FIELD,
    DEFAULT_SST_FIELD,
    DEFAULT_UWND_FIELD,
    DEFAULT_VWND_FIELD,
    DEFAULT_OLR_FIELD]

DEFAULT_CLIM_START_YEAR = 1980
DEFAULT_CLIM_END_YEAR = 2018


FIGURE_SIZE = (8, 11)
N_COLS = 1
N_ROWS = 5
CMAP = plt.cm.RdBu_r


def read_esrl_data(datafile):
    data = np.genfromtxt(datafile, delimiter=',')

    year = np.asarray(data[:, 0], dtype='i8')
    month = np.asarray(data[:, 1], dtype='i8')
    day = np.asarray(data[:, 2], dtype='i8')
    index = data[:, 3]

    n_samples = data.shape[0]

    time = np.array([datetime.datetime(year[i], month[i], day[i])
                     for i in range(n_samples)])

    return time, index


def plot_mei_eofs(eofs_ds, lat_field=DEFAULT_LAT_FIELD,
                  lon_field=DEFAULT_LON_FIELD,
                  variables=DEFAULT_VARIABLES,
                  output_file=None, show_plots=True):
    n_vars = len(variables)
    seasons = eofs_ds[SEASON_DIM_NAME].values

    lon_grid, lat_grid = np.meshgrid(
        eofs_ds[lon_field].values, eofs_ds[lat_field].values)

    proj = ccrs.PlateCarree(central_longitude=180)

    if output_file is not None and output_file:
        pdf = PdfPages(output_file)
    else:
        pdf = None

    for s in seasons:
        current_ds = eofs_ds.where(
            (eofs_ds[SEASON_DIM_NAME] == s) &
            (eofs_ds['mode'] == 0), drop=True).squeeze()

        n_pages = int(np.ceil(n_vars / (N_COLS * N_ROWS)))
        var_idx = 0
        for pg in range(n_pages):
            fig = plt.figure(figsize=FIGURE_SIZE)
            gs = GridSpec(N_ROWS, N_COLS, figure=fig,
                          wspace=0, hspace=0.25,
                          left=0.15, right=0.85)
            for row_idx in range(N_ROWS):
                for col_idx in range(N_COLS):
                    if var_idx >= n_vars:
                        break

                    ax = fig.add_subplot(gs[row_idx, col_idx], projection=proj)
                    current_ds[variables[var_idx]].plot.contourf(
                        ax=ax, transform=ccrs.PlateCarree())

                    ax.coastlines()
                    ax.set_title('%s' % variables[var_idx])

                    var_idx += 1

            fig.suptitle('Season: %s' % get_season_name(s))

            if pdf is not None:
                pdf.savefig()

            if show_plots:
                plt.show()

            plt.close()

    if pdf is not None:
        pdf.close()


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


def plot_mei(index, esrl_times=None, esrl_index=None,
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

    ax.plot(valid_times, valid_index, 'b-', label='MEI')

    if esrl_times is not None and esrl_index is not None:
        corr_coeff = get_correlation_coeff(
            times, index_vals, esrl_times, esrl_index)
        mask = np.zeros(esrl_times.shape, dtype=bool)
        for i, t in enumerate(esrl_times):
            mask[i] = t.year in valid_years
        ax.plot(
            esrl_times[mask], esrl_index[mask], 'r--',
            label='ESRL MEI.v2 (r = {:.2f})'.format(corr_coeff))

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
        description='Calculate monthly MEI values')

    parser.add_argument(
        'datafile', help='netCDF file containing input data')
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
        '--slp-field', dest='slp_field', default=DEFAULT_SLP_FIELD,
        help='name of variable corresponding to SLP in input datafile')
    parser.add_argument(
        '--sst-field', dest='sst_field', default=DEFAULT_SST_FIELD,
        help='name of variable corresponding to SST in input datafile')
    parser.add_argument(
        '--uwnd-field', dest='uwnd_field', default=DEFAULT_UWND_FIELD,
        help='name of variable corresponding to u-wind in input datafile')
    parser.add_argument(
        '--vwnd-field', dest='vwnd_field', default=DEFAULT_VWND_FIELD,
        help='name of variable corresponding to v-wind in input datafile')
    parser.add_argument(
        '--olr-field', dest='olr_field', default=DEFAULT_OLR_FIELD,
        help='name of variable corresponding to OLR in input datafile')

    parser.add_argument(
        '--base-period-start-year', dest='start_year',
        type=int, default=DEFAULT_CLIM_START_YEAR,
        help='initial year in base period used to compute EOFs')
    parser.add_argument(
        '--base-period-end-year', dest='end_year',
        type=int, default=DEFAULT_CLIM_END_YEAR,
        help='last year in base period used to compute EOFs')

    parser.add_argument(
        '--index-output-file', dest='index_output_file',
        default='', help='name of file to write index to')
    parser.add_argument(
        '--index-plot-output-file', dest='index_plot_output_file',
        default='', help='name of file to write index plot to')
    parser.add_argument(
        '--anom-output-file', dest='anom_output_file',
        default='', help='name of netCDF file to write anomalies to')
    parser.add_argument(
        '--clim-output-file', dest='clim_output_file',
        default='', help='name of netCDF file to write climatology to')
    parser.add_argument(
        '--eofs-output-file', dest='eofs_output_file',
        default='', help='name of netCDF file to write EOFs to')
    parser.add_argument(
        '--plot-output-file', dest='plot_output_file',
        default='', help='name of file to write EOF plots to')

    parser.add_argument(
        '--no-show-plots', dest='no_show_plots', action='store_true',
        help='do not show plots')

    parser.add_argument(
        '--esrl-datafile', dest='esrl_datafile',
        default='', help='datafile containing seasonal ESRL MEI.v2 values')

    return parser.parse_args()


def main():
    args = parse_cmd_line_args()

    if args.esrl_datafile:
        esrl_times, esrl_index = read_esrl_data(args.esrl_datafile)
    else:
        esrl_times = None
        esrl_index = None

    with xr.open_dataset(args.datafile) as ds:

        ref_ds = ds.where(
            (ds[args.time_field].dt.year >= args.start_year) &
            (ds[args.time_field].dt.year <= args.end_year),
            drop=True)

        ref_anom_ds, clim_ds = calculate_monthly_anomalies(
            ref_ds, time_field=args.time_field)

        if args.anom_output_file:
            ref_anom_ds.to_netcdf(args.anom_output_file)

        if args.clim_output_file:
            clim_ds.to_netcdf(args.clim_output_file)

        ref_std_anom_ds = standardize_values(
            ref_anom_ds, time_field=args.time_field,
            clim_start_year=args.start_year,
            clim_end_year=args.end_year)

        ref_seasonal_anom = get_seasonal_data(
            ref_std_anom_ds, time_field=args.time_field)

        eofs_ds, pcs_da = calculate_seasonal_eofs(
            ref_seasonal_anom, time_field=args.time_field)

        eofs_ds, pcs_da = fix_phases(
            eofs_ds, pcs_da, time_field=args.time_field,
            lon_field=args.lon_field, sst_field=args.sst_field)

        if args.eofs_output_file:
            eofs_ds.to_netcdf(args.eofs_output_file)

        plot_mei_eofs(eofs_ds, output_file=args.plot_output_file,
                      show_plots=(not args.no_show_plots))

        anom_ds, _ = calculate_monthly_anomalies(
            ds, climatology=clim_ds, time_field=args.time_field)

        std_anom_ds = standardize_values(
            anom_ds, time_field=args.time_field,
            clim_start_year=args.start_year,
            clim_end_year=args.end_year)

        seasonal_anom = get_seasonal_data(
            std_anom_ds, time_field=args.time_field)

        index = calculate_seasonal_mei(
            seasonal_anom, eofs_ds, pcs_da, time_field=args.time_field)

        write_index_values(index, output_file=args.index_output_file,
                           time_field=args.time_field)

        if not args.no_show_plots or args.index_plot_output_file:
            plot_mei(index, esrl_times=esrl_times, esrl_index=esrl_index,
                     output_file=args.index_plot_output_file,
                     show_plots=(not args.no_show_plots),
                     time_field=args.time_field)


if __name__ == '__main__':
    main()
