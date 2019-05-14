import argparse
import numpy as np
import xarray as xr

from climate_indices.blocking import (BOM_BLOCKING_INDEX_FIELD_NAME,
                                      calc_bom_blocking)

DEFAULT_TIME_FIELD = 'time'
DEFAULT_LAT_FIELD = 'lat'
DEFAULT_LON_FIELD = 'lon'
DEFAULT_UWND_FIELD = 'uwnd'

DEFAULT_WINDOW_LENGTH = 0


def calc_zonal_mean_index(idx_data, lon_field=DEFAULT_LON_FIELD):
    return idx_data.mean(lon_field)


def write_zonal_mean_index(zonal_idx_data, output_file,
                           time_field=DEFAULT_TIME_FIELD,
                           index_field=BOM_BLOCKING_INDEX_FIELD_NAME):
    n_samples = zonal_idx_data[time_field].values.shape[0]

    fields = [('year', '%d'),
              ('month', '%d'),
              ('day', '%d'),
              ('zonal_mean_index', '%14.8e')]

    n_fields = len(fields)
    header = ','.join([f[0] for f in fields])
    fmt = ','.join([f[1] for f in fields])

    data = np.zeros((n_samples, n_fields))
    data[:, 0] = zonal_idx_data[time_field].dt.year.values
    data[:, 1] = zonal_idx_data[time_field].dt.month.values
    data[:, 2] = zonal_idx_data[time_field].dt.day.values
    data[:, 3] = np.squeeze(zonal_idx_data[index_field].values)

    np.savetxt(output_file, data, header=header, fmt=fmt)


def parse_cmd_line_args():
    parser = argparse.ArgumentParser(
        description='Calculate Wright BOM blocking index')

    parser.add_argument('datafile', help='datafile containing u-wind data')
    parser.add_argument(
        '--time-field', dest='time_field', default=DEFAULT_TIME_FIELD,
        help='name of time dimension in input datafile')
    parser.add_argument(
        '--lat-field', dest='lat_field', default=DEFAULT_LAT_FIELD,
        help='name of latitude dimension in input datafile')
    parser.add_argument(
        '--lon-field', dest='lon_field', default=DEFAULT_LON_FIELD,
        help='name of longitude dimension in input datafile')
    parser.add_argument(
        '--uwnd-field', dest='uwnd_field', default=DEFAULT_UWND_FIELD,
        help='name of u-wind dimension in input datafile')
    parser.add_argument(
        '--window-lenght', dest='window_length', type=int,
        default=DEFAULT_WINDOW_LENGTH,
        help='length of window over which to perform running mean')
    parser.add_argument(
        '--index-output-nc', dest='index_output_nc', default='',
        help='name of output file to write index values to')
    parser.add_argument(
        '--zonal-mean-index-output', dest='zonal_mean_output_file',
        default='',
        help='name of output file to write zonal mean index values to')

    return parser.parse_args()


def main():
    args = parse_cmd_line_args()

    if args.window_length < 0:
        raise ValueError('window length must be non-negative')

    with xr.open_dataset(args.datafile) as ds:
        uwnd_data = ds[args.uwnd_field]
        idx_data = calc_bom_blocking(
            uwnd_data, time_field=args.time_field,
            lat_field=args.lat_field,
            window_length=args.window_length)

        if args.index_output_nc:
            idx_data.to_netcdf(args.index_output_nc)

        if args.zonal_mean_output_file:
            zonal_mean_index = calc_zonal_mean_index(
                idx_data, lon_field=args.lon_field)

            write_zonal_mean_index(zonal_mean_index,
                                   args.zonal_mean_output_file,
                                   time_field=args.time_field)


if __name__ == '__main__':
    main()
