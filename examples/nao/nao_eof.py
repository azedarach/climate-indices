import argparse
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr

from climate_indices.nao import (calculate_daily_region_anomalies,
                                 calculate_monthly_region_anomalies,
                                 calculate_seasonal_eof)


DEFAULT_TIME_FIELD = 'time'
DEFAULT_LAT_FIELD = 'lat'
DEFAULT_LON_FIELD = 'lon'
DEFAULT_HGT_FIELD = 'hgt'


DEFAULT_START_YEAR = 1950
DEFAULT_END_YEAR = 2000


FREQUENCY = ['daily', 'monthly']
VALID_SEASONS = ['DJF', 'MAM', 'JJA', 'SON']

FIGURE_SIZE = (7, 5)
CMAP = plt.cm.RdBu_r
CENTRAL_LATITUDE = 90.0
CENTRAL_LONGITUDE = -80.0


def plot_seasonal_eofs(seasonal_eofs, mode=0,
                       output_file=None, show_plot=True):
    fig = plt.figure(figsize=FIGURE_SIZE)
    proj = ccrs.Orthographic(central_latitude=CENTRAL_LATITUDE,
                             central_longitude=CENTRAL_LONGITUDE)
    ax = fig.add_subplot(111, projection=proj)

    ax.coastlines()
    ax.set_global()

    eof_data = seasonal_eofs[{'mode': mode}].squeeze()
    print(eof_data)
    eof_data.plot.contourf(ax=ax, transform=ccrs.PlateCarree())

    if show_plot:
        plt.show()


def parse_cmd_line_args():
    parser = argparse.ArgumentParser(
        description='Plot seasonal EOF identified with NAO')

    parser.add_argument(
        'datafile', help='datafile containing geopotential heights')
    parser.add_argument(
        '--frequency', dest='frequency', choices=FREQUENCY,
        default='daily', help='frequency of sampling in data')
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
        '--plot-output-file', dest='plot_output_file',
        default='', help='name of file to write EOF plots to')
    parser.add_argument(
        '--no-show-plots', dest='no_show_plots', action='store_true',
        help='do not show plots')
    return parser.parse_args()


def main():
    args = parse_cmd_line_args()

    with xr.open_dataset(args.datafile) as ds:
        if args.frequency == 'daily':
            hgt_data = ds[args.hgt_field]
        else:
            hgt_data = ds[args.hgt_field].resample(
                {args.time_field: '1M'}).mean(args.time_field)

        ref_data = hgt_data.where(
            (hgt_data[args.time_field].dt.year >= args.start_year) &
            (hgt_data[args.time_field].dt.year <= args.end_year),
            drop=True)

        if args.frequency == 'daily':
            anom_data, _ = calculate_daily_region_anomalies(ref_data)
        else:
            anom_data, _ = calculate_monthly_region_anomalies(ref_data)

        if args.standardize:
            anom_data = anom_data / anom_data.std(args.time_field)

        seasonal_eofs = calculate_seasonal_eof(
            anom_data, season=args.season)

        plot_seasonal_eofs(
            seasonal_eofs['eofs'], output_file=args.plot_output_file,
            show_plot=(not args.no_show_plots))


if __name__ == '__main__':
    main()
