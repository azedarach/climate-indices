import argparse
import numpy as np
import regionmask
import xarray as xr


MIN_LATITUDE = -30.0
MAX_LATITUDE = 30.0
MIN_LONGITUDE = 100.0
MAX_LONGITUDE = 290.0
GRID_RESOLUTION = 2.5

REF_POINTS = [(283.0, 7.0), (248.0, 29.0)]


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


def parse_cmd_line_args():
    parser = argparse.ArgumentParser(
        description='Generate region mask for MEI calculation')

    parser.add_argument('output_file', help='output file to save mask as')

    return parser.parse_args()


def main():
    args = parse_cmd_line_args()

    lon, lat = get_grid_values()
    land_mask = get_region_mask(lat, lon)

    land_mask.to_netcdf(args.output_file)


if __name__ == '__main__':
    main()
