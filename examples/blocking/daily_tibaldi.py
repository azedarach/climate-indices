import argparse
import numpy as np
import xarray as xr

from climate_indices.blocking import calc_tibaldi, tibaldi_index_1d


MIN_LONGITUDE = 330
MAX_LONGITUDE = 30


DEFAULT_TIME_FIELD = 'time'
DEFAULT_LAT_FIELD = 'lat'
DEFAULT_LON_FIELD = 'lon'
DEFAULT_HGT_FIELD = 'hgt'

DEFAULT_START_YEAR = 1950
DEFAULT_END_YEAR = 2000

DEFAULT_WINDOW_LENGTH = 0


def get_blocking_sector(ds, lon_field=DEFAULT_LON_FIELD):
    return ds.where(((ds[lon_field] >= MIN_LONGITUDE) |
                     (ds[lon_field] <= MAX_LONGITUDE)), drop=True)


def write_tibaldi_index(time, index, output_file=None):
    if output_file is None or not output_file:
        return

    fields = [('year', '%d'),
              ('month', '%d'),
              ('day', '%d'),
              ('index', '%14.8e')]

    header = ','.join([f[0] for f in fields])
    fmt = ','.join([f[1] for f in fields])

    n_samples = time.shape[0]
    n_fields = len(fields)

    data = np.empty((n_samples, n_fields), dtype=index.dtype)
    data[:, 0] = np.array([t.year for t in time], dtype='i8')
    data[:, 1] = np.array([t.month for t in time], dtype='i8')
    data[:, 2] = np.array([t.day for t in time], dtype='i8')
    data[:, 3] = index

    np.savetxt(output_file, data, header=header, fmt=fmt)


def parse_cmd_line_args():
    parser = argparse.ArgumentParser(
        description='Calculate daily Tibaldi index')

    parser.add_argument(
        'datafile', help='netCDF file containing geopotential heights')
    parser.add_argument(
        '--time-field', dest='time_field', default=DEFAULT_TIME_FIELD,
        help='name of variable containing time in input datafile')
    parser.add_argument(
        '--lat-field', dest='lat_field', default=DEFAULT_LAT_FIELD,
        help='name of variable containing latitude in input datafile')
    parser.add_argument(
        '--lon-field', dest='lon_field', default=DEFAULT_LON_FIELD,
        help='name of variable containing longitude in input datafile')
    parser.add_argument(
        '--hgt-field', dest='hgt_field', default=DEFAULT_HGT_FIELD,
        help='name of variable containing height data in input datafile')
    parser.add_argument(
        '--southern-hemisphere', dest='southern_hemisphere',
        action='store_true', help='calculate index for Southern Hemisphere')
    parser.add_argument(
        '--window-length', dest='window_length', type=int,
        default=DEFAULT_WINDOW_LENGTH,
        help='length of rolling window to average over')
    parser.add_argument('--start-year', dest='start_year', type=int,
                        default=DEFAULT_START_YEAR,
                        help='start year of reference period')
    parser.add_argument('--end-year', dest='end_year', type=int,
                        default=DEFAULT_END_YEAR,
                        help='end year of reference period')
    parser.add_argument(
        '--2d-index-output-file', dest='index_2d_output_file',
        default='', help='name of file to write two-dimensional index to')
    parser.add_argument(
        '--index-output-file', dest='index_output_file',
        default='', help='name of file to write one-dimensional index to')

    return parser.parse_args()


def main():
    args = parse_cmd_line_args()

    if args.window_length < 0:
        raise ValueError('rolling window length must be non-negative')

    with xr.open_dataset(args.datafile) as ds:
        sector_ds = get_blocking_sector(ds, lon_field=args.lon_field)
        index_ds = calc_tibaldi(sector_ds, time_field=args.time_field,
                                lat_field=args.lat_field,
                                lon_field=args.lon_field,
                                hgt_field=args.hgt_field,
                                southern_hemisphere=args.southern_hemisphere,
                                window_length=args.window_length)

        if args.index_2d_output_file:
            index_ds.to_netcdf(args.index_2d_output_file)

        index_time, index_vals = tibaldi_index_1d(
            index_ds, time_field=args.time_field,
            lon_field=args.lon_field,
            southern_hemisphere=args.southern_hemisphere,
            clim_start_year=args.start_year,
            clim_end_year=args.end_year)

        if args.index_output_file:
            write_tibaldi_index(index_time, index_vals, args.index_output_file)


if __name__ == '__main__':
    main()
