import argparse
import datetime
import numpy as np
import regionmask
import xarray as xr
import xesmf as xe


DEFAULT_SLP_FIELD = 'PRMSL_GDS0_MSL'
DEFAULT_SST_FIELD = 'BRTMP_GDS0_SFC'
DEFAULT_UWND_FIELD = 'UGRD_GDS0_HTGL'
DEFAULT_VWND_FIELD = 'VGRD_GDS0_HTGL'
DEFAULT_OLR_FIELD = 'olr'

DEFAULT_SLP_TIME_FIELD = 'initial_time0_hours'
DEFAULT_SLP_LAT_FIELD = 'g0_lat_1'
DEFAULT_SLP_LON_FIELD = 'g0_lon_2'
DEFAULT_SST_TIME_FIELD = 'initial_time0_hours'
DEFAULT_SST_LAT_FIELD = 'g0_lat_2'
DEFAULT_SST_LON_FIELD = 'g0_lon_3'
DEFAULT_UWND_TIME_FIELD = 'initial_time0_hours'
DEFAULT_UWND_LAT_FIELD = 'g0_lat_1'
DEFAULT_UWND_LON_FIELD = 'g0_lon_2'
DEFAULT_VWND_TIME_FIELD = 'initial_time0_hours'
DEFAULT_VWND_LAT_FIELD = 'g0_lat_1'
DEFAULT_VWND_LON_FIELD = 'g0_lon_2'
DEFAULT_OLR_TIME_FIELD = 'time'
DEFAULT_OLR_LAT_FIELD = 'lat'
DEFAULT_OLR_LON_FIELD = 'lon'


MIN_LATITUDE = -30.0
MAX_LATITUDE = 30.0
MIN_LONGITUDE = 100.0
MAX_LONGITUDE = 290.0
GRID_RESOLUTION = 2.5

REF_POINTS = [(283.0, 7.0), (248.0, 29.0)]

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


def regrid_dataset(ds, lat='lat', lon='lon', var='slp', method='bilinear'):
    lon_values, lat_values = get_grid_values()
    grid_ds = xr.Dataset(
        {'lat': (['lat'], lat_values),
         'lon': (['lon'], lon_values)})

    if lat != 'lat':
        ds = ds.rename({lat: 'lat'})

    if lon != 'lon':
        ds = ds.rename({lon: 'lon'})

    dr = ds[var]

    regridder = xe.Regridder(ds, grid_ds, method)

    regridded_dr = regridder(dr)

    regridder.clean_weight_file()

    regridded_ds = xr.Dataset({var: regridded_dr})

    return regridded_ds


def get_land_mask(lat, lon):
    land = regionmask.defined_regions.natural_earth.land_110
    return xr.Dataset({'region': land.mask(lon, lat=lat, wrap_lon=True)})


def get_atlantic_mask(lat, lon):
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    x0, y0 = REF_POINTS[0]
    x1, y1 = REF_POINTS[1]
    b = x1 * (y1 - y0) - y1 * (x1 - x0)

    mask = lon_grid * (y1 - y0) - lat_grid * (x1 - x0) > b
    region = np.full(lon_grid.shape, np.NaN)
    region[mask] = 0.0

    ds = xr.Dataset({'region': (['lat', 'lon'], region)},
                    coords={'lat': (['lat'], lat),
                            'lon': (['lon'], lon)})

    return ds


def get_region_mask(lat, lon):
    land_mask = get_land_mask(lat, lon)
    atl_mask = get_atlantic_mask(lat, lon)

    full_mask = np.logical_or(~np.isnan(land_mask.region),
                              ~np.isnan(atl_mask.region))

    full_region = np.full(full_mask.shape, np.NaN)
    full_region[full_mask.values] = 0.0

    ds = xr.Dataset({'region': (['lat', 'lon'], full_region)},
                    coords={'lat': (['lat'], lat),
                            'lon': (['lon'], lon)})

    return ds


def apply_mei_region_mask(ds):
    region_mask = get_region_mask(ds.lat, ds.lon)
    masked_ds = ds.where(region_mask.region != 0)
    return masked_ds


def set_record_dates(ds):
    n_samples = ds['time'].shape[0]
    new_times = np.empty(ds['time'].shape, dtype=ds['time'].dtype)
    for i in range(n_samples):
        year = ds['time'].dt.year[i]
        month = ds['time'].dt.month[i]
        new_times[i] = datetime.datetime(year, month, 1)

    ds['time'] = new_times

    return ds


def get_input_data(datafiles, time='time', lat='lat', lon='lon', var='slp',
                   regrid=True, method='bilinear', apply_mask=True):
    with xr.open_mfdataset(datafiles) as ds:
        if time != 'time':
            ds = ds.rename({time: 'time'})
        if lat != 'lat':
            ds = ds.rename({lat: 'lat'})
        if lon != 'lon':
            ds = ds.rename({lon: 'lon'})

        if regrid:
            ds = regrid_dataset(ds, lat='lat', lon='lon', var=var,
                                method=method)

        if apply_mask:
            ds = apply_mei_region_mask(ds)

        ds = set_record_dates(ds)

        ds.load()

        return ds


def parse_cmd_line_args():
    parser = argparse.ArgumentParser(
        description='Create combined input datafile for MEI calculation')

    parser.add_argument('output_file', help='name for combined output file')
    parser.add_argument(
        '--slp-file', dest='slp_file', action='append',
        help='name of input file containing monthly SLP values')
    parser.add_argument(
        '--sst-file', dest='sst_file', action='append',
        help='name of input file containing monthly SST values')
    parser.add_argument(
        '--uwnd-file', dest='uwnd_file', action='append',
        help='name of input file containing monthly u-wind values')
    parser.add_argument(
        '--vwnd-file', dest='vwnd_file', action='append',
        help='name of input file containing monthly v-wind values')
    parser.add_argument(
        '--olr-file', dest='olr_file', action='append',
        help='name of input file containing monthly OLR values')

    parser.add_argument(
        '--slp-field', dest='slp_field', default=DEFAULT_SLP_FIELD,
        help='name of SLP field in input datafiles')
    parser.add_argument(
        '--slp-time-field', dest='slp_time_field',
        default=DEFAULT_SLP_TIME_FIELD,
        help='name of time dimension in SLP files')
    parser.add_argument(
        '--slp-lat-field', dest='slp_lat_field',
        default=DEFAULT_SLP_LAT_FIELD,
        help='name of latitude dimension in SLP files')
    parser.add_argument(
        '--slp-lon-field', dest='slp_lon_field',
        default=DEFAULT_SLP_LON_FIELD,
        help='name of longitude dimension in SLP files')

    parser.add_argument(
        '--sst-field', dest='sst_field', default=DEFAULT_SST_FIELD,
        help='name of SST field in input datafiles')
    parser.add_argument(
        '--sst-time-field', dest='sst_time_field',
        default=DEFAULT_SST_TIME_FIELD,
        help='name of time dimension in SST files')
    parser.add_argument(
        '--sst-lat-field', dest='sst_lat_field',
        default=DEFAULT_SST_LAT_FIELD,
        help='name of latitude dimension in SST files')
    parser.add_argument(
        '--sst-lon-field', dest='sst_lon_field',
        default=DEFAULT_SST_LON_FIELD,
        help='name of longitude dimension in SST files')

    parser.add_argument(
        '--uwnd-field', dest='uwnd_field', default=DEFAULT_UWND_FIELD,
        help='name of u-wind field in input datafiles')
    parser.add_argument(
        '--uwnd-time-field', dest='uwnd_time_field',
        default=DEFAULT_UWND_TIME_FIELD,
        help='name of time dimension in u-wind files')
    parser.add_argument(
        '--uwnd-lat-field', dest='uwnd_lat_field',
        default=DEFAULT_UWND_LAT_FIELD,
        help='name of latitude dimension in u-wind files')
    parser.add_argument(
        '--uwnd-lon-field', dest='uwnd_lon_field',
        default=DEFAULT_UWND_LON_FIELD,
        help='name of longitude dimension in u-wind files')

    parser.add_argument(
        '--vwnd-field', dest='vwnd_field', default=DEFAULT_VWND_FIELD,
        help='name of v-wind field in input datafiles')
    parser.add_argument(
        '--vwnd-time-field', dest='vwnd_time_field',
        default=DEFAULT_VWND_TIME_FIELD,
        help='name of time dimension in v-wind files')
    parser.add_argument(
        '--vwnd-lat-field', dest='vwnd_lat_field',
        default=DEFAULT_VWND_LAT_FIELD,
        help='name of latitude dimension in v-wind files')
    parser.add_argument(
        '--vwnd-lon-field', dest='vwnd_lon_field',
        default=DEFAULT_VWND_LON_FIELD,
        help='name of longitude dimension in v-wind files')

    parser.add_argument(
        '--olr-field', dest='olr_field', default=DEFAULT_OLR_FIELD,
        help='name of OLR field in input datafiles')
    parser.add_argument(
        '--olr-time-field', dest='olr_time_field',
        default=DEFAULT_OLR_TIME_FIELD,
        help='name of time dimension in OLR files')
    parser.add_argument(
        '--olr-lat-field', dest='olr_lat_field',
        default=DEFAULT_OLR_LAT_FIELD,
        help='name of latitude dimension in OLR files')
    parser.add_argument(
        '--olr-lon-field', dest='olr_lon_field',
        default=DEFAULT_OLR_LON_FIELD,
        help='name of longitude dimension in OLR files')

    return parser.parse_args()


def main():
    args = parse_cmd_line_args()

    if not args.slp_file:
        raise RuntimeError('no SLP datafiles given')

    if not args.sst_file:
        raise RuntimeError('no SST datafiles given')

    if not args.uwnd_file:
        raise RuntimeError('no u-wind datafiles given')

    if not args.vwnd_file:
        raise RuntimeError('no v-wind datafiles given')

    if not args.olr_file:
        raise RuntimeError('no OLR datafiles given')

    slp_ds = get_input_data(args.slp_file, time=args.slp_time_field,
                            lat=args.slp_lat_field,
                            lon=args.slp_lon_field, var=args.slp_field)
    sst_ds = get_input_data(args.sst_file, time=args.sst_time_field,
                            lat=args.sst_lat_field,
                            lon=args.sst_lon_field, var=args.sst_field)
    uwnd_ds = get_input_data(args.uwnd_file, time=args.uwnd_time_field,
                             lat=args.uwnd_lat_field,
                             lon=args.uwnd_lon_field, var=args.uwnd_field)
    vwnd_ds = get_input_data(args.vwnd_file, time=args.vwnd_time_field,
                             lat=args.vwnd_lat_field,
                             lon=args.vwnd_lon_field, var=args.vwnd_field)
    olr_ds = get_input_data(args.olr_file, time=args.olr_time_field,
                            lat=args.olr_lat_field,
                            lon=args.olr_lon_field, var=args.olr_field)

    combined_ds = xr.merge([slp_ds, sst_ds, uwnd_ds, vwnd_ds, olr_ds],
                           join='inner')

    combined_ds.to_netcdf(args.output_file)


if __name__ == '__main__':
    main()
