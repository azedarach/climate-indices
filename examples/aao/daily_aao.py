import argparse
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

from cartopy.util import add_cyclic_point

from climate_indices import calc_aao, calc_aao_index
from climate_indices.anomalies import daily_anomalies
from climate_indices.timeavg import (monthly_means, daily_means,
                                     multiyear_daily_means)


DEFAULT_TIME_FIELD = 'time'
DEFAULT_LAT_FIELD = 'lat'
DEFAULT_LON_FIELD = 'lon'
DEFAULT_HGT_FIELD = 'hgt'

DEFAULT_START_YEAR = 1979
DEFAULT_END_YEAR = 2000


FIGURE_SIZE = (5, 5)
CMAP = plt.cm.RdBu_r


def read_data(datafile, time=DEFAULT_TIME_FIELD, lat=DEFAULT_LAT_FIELD,
              lon=DEFAULT_LON_FIELD, hgt=DEFAULT_HGT_FIELD):
    time_data = None
    lat_data = None
    lon_data = None
    hgt_data = None
    with nc.Dataset(datafile, 'r') as ncin:
        time_vals = ncin[time][:]
        time_data = nc.num2date(time_vals, ncin[time].units,
                                calendar=ncin[time].calendar)
        lat_data = ncin[lat][:]
        lon_data = ncin[lon][:]
        hgt_data = np.squeeze(ncin[hgt][:])

    return time_data, lat_data, lon_data, hgt_data


def get_base_period_mask(dt, start_year, end_year):
    years = np.array([t.year for t in dt], dtype='i8')
    return np.logical_and(years >= start_year, years <= end_year)


def plot_pattern(lat, lon, aao_pattern, output_file=None, n_contours=12):
    cyclic_lon = np.full(np.size(lon) + 1, 360.)
    cyclic_lon[:-1] = lon
    cyclic_data = add_cyclic_point(aao_pattern)

    vmin = np.min(cyclic_data)
    vmax = np.max(cyclic_data)
    clevs = np.linspace(vmin, vmax, n_contours)

    lon_grid, lat_grid = np.meshgrid(cyclic_lon, lat)

    plt.figure(figsize=FIGURE_SIZE)

    proj = ccrs.Orthographic(central_longitude=0, central_latitude=-90)

    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.coastlines(resolution='110m')
    ax.gridlines(linestyle='--', alpha=0.7)

    cs = ax.contourf(lon_grid, lat_grid, cyclic_data, levels=clevs, cmap=CMAP,
                     transform=ccrs.PlateCarree())

    plt.colorbar(cs)

    plt.show()

    if output_file is not None and output_file:
        plt.savefig(output_file, bbox_inches='tight')

    plt.close()


def write_aao_index(time, index, output_file=None):
    if output_file is None or not output_file:
        return

    fields = [('year', '%d'),
              ('month', '%d'),
              ('day', '%d'),
              ('index', '%14.8e')]

    header = ','.join([f[0] for f in fields])
    fmt = ','.join([f[1] for f in fields])

    n_fields = len(fields)
    n_samples = np.size(time)

    data = np.empty((n_samples, n_fields))
    years = np.array([t.year for t in time], dtype='i8')
    months = np.array([t.month for t in time], dtype='i8')
    days = np.array([t.day for t in time], dtype='i8')

    data[:, 0] = years
    data[:, 1] = months
    data[:, 2] = days
    data[:, 3] = index

    np.savetxt(output_file, data, header=header, fmt=fmt)


def parse_cmd_line_args():
    parser = argparse.ArgumentParser(
        description='Calculate AAO pattern and daily index from daily data')

    parser.add_argument(
        'datafile',
        help='netCDF file containing 700 hPa heights')
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
        '--base-period-start-year', dest='start_year',
        type=int, default=DEFAULT_START_YEAR,
        help='initial year in base period used to compute pattern')
    parser.add_argument(
        '--base-period-end-year', dest='end_year',
        type=int, default=DEFAULT_END_YEAR,
        help='last year in base period used to compute pattern')
    parser.add_argument(
        '--index-output-file', dest='index_output_file',
        default='', help='name of file to write index data in CSV format')
    parser.add_argument(
        '--plot-output-file', dest='plot_output_file',
        default='', help='name of file to write pattern plot to')

    return parser.parse_args()


def main():
    args = parse_cmd_line_args()

    time, lat, lon, z700 = read_data(
        args.datafile, time=args.time_field, lat=args.lat_field,
        lon=args.lon_field, hgt=args.hgt_field)

    daily_times, daily_z700 = daily_means(time, z700)
    monthly_times, monthly_z700 = monthly_means(daily_times, daily_z700)

    aao_pcs, aao_eofs, aao_lat, aao_lon = calc_aao(
        monthly_times, lat, lon, monthly_z700,
        start_year=args.start_year,
        end_year=args.end_year)

    plot_pattern(aao_lat, aao_lon, aao_eofs, output_file=args.plot_output_file)

    daily_base_period_mask = get_base_period_mask(
        daily_times, args.start_year, args.end_year)

    daily_clim = multiyear_daily_means(
        daily_times[daily_base_period_mask],
        daily_z700[daily_base_period_mask])

    daily_z700_anom = daily_anomalies(
        daily_times, daily_z700, daily_clim)

    normalization = aao_pcs.std(ddof=1)

    daily_aao_index_times, daily_aao_index = calc_aao_index(
        daily_times, lat, lon, daily_z700_anom,
        aao_eofs, normalization=normalization)

    if args.index_output_file:
        write_aao_index(
            daily_aao_index_times, daily_aao_index, args.index_output_file)


if __name__ == '__main__':
    main()
