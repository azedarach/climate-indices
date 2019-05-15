import argparse
import numpy as np
import xarray as xr
import xesmf as xe


DEFAULT_TIME_FIELD = 'initial_time0_hours'
DEFAULT_LAT_FIELD = 'g0_lat_1'
DEFAULT_LON_FIELD = 'g0_lon_2'


MIN_LATITUDE = -30.0
MAX_LATITUDE = 30.0
MIN_LONGITUDE = 100.0
MAX_LONGITUDE = 290.0
GRID_RESOLUTION = 2.5


REGRIDDING_CHOICES = [
    'bilinear', 'conservative', 'nearest_s2d',
    'nearest_d2s', 'patch']


def get_grid_values():
    lon_values = [MIN_LONGITUDE]
    current_lon = MIN_LONGITUDE
    while current_lon < MAX_LONGITUDE:
        current_lon += GRID_RESOLUTION
        lon_values.append(current_lon)

    lat_values = [MIN_LATITUDE]
    current_lat = MIN_LATITUDE
    while current_lat < MAX_LATITUDE:
        current_lat += GRID_RESOLUTION
        lat_values.append(current_lat)

    return np.array(lon_values), np.array(lat_values)


def parse_cmd_line_args():
    parser = argparse.ArgumentParser(
        description='Regrid data onto 2.5 degree grid')

    parser.add_argument('input_file', help='input datafile')
    parser.add_argument('output_file', help='output datafile')
    parser.add_argument('variable', help='variable to regrid')
    parser.add_argument('--lat-field', dest='lat_field',
                        default=DEFAULT_LAT_FIELD,
                        help='name of field corresponding to latitude')
    parser.add_argument('--lon-field', dest='lon_field',
                        default=DEFAULT_LON_FIELD,
                        help='name of field corresponding to longitude')
    parser.add_argument('--method', dest='method',
                        choices=REGRIDDING_CHOICES,
                        default='bilinear',
                        help='regridding method')

    return parser.parse_args()


def main():
    args = parse_cmd_line_args()

    lon_values, lat_values = get_grid_values()

    grid_ds = xr.Dataset(
        {'lat': (['lat'], lat_values),
         'lon': (['lon'], lon_values)})
    print(grid_ds)

    with xr.open_dataset(args.input_file) as ds:
        ds = ds.rename({args.lat_field: 'lat', args.lon_field: 'lon'})
        dr = ds[args.variable]
        regridder = xe.Regridder(ds, grid_ds, args.method)
        dr_out = regridder(dr)
        regridder.clean_weight_file()

        ds_out = xr.Dataset({args.variable: dr_out})
        ds_out = ds_out.rename({'lat': args.lat_field, 'lon': args.lon_field})
        ds_out.to_netcdf(args.output_file)


if __name__ == '__main__':
    main()
