import argparse
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from cartopy.util import add_cyclic_point
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from climate_indices.nhtele import (calculate_seasonal_eofs, cluster_pcs,
                                    calculate_composites)


DEFAULT_TIME_FIELD = 'time'
DEFAULT_LAT_FIELD = 'lat'
DEFAULT_LON_FIELD = 'lon'
DEFAULT_HGT_FIELD = 'hgt'

VALID_SEASONS = ['ALL', 'DJF', 'MAM', 'JJA', 'SON']
DEFAULT_SEASON = 'DJF'
DEFAULT_START_YEAR = 1950
DEFAULT_END_YEAR = 2000

DEFAULT_N_EOFS = 6
DEFAULT_N_CLUSTERS = 4
DEFAULT_N_INIT = 20

EOF_DIM_NAME = 'mode'
CLUSTER_DIM_NAME = 'cluster'

FIGURE_SIZE = (8, 11)
CMAP = plt.cm.RdBu_r
N_ROWS = 5
N_COLS = 2


def calculate_anomalies(hgt_data, climatology=None,
                        time_field=DEFAULT_TIME_FIELD):
    if climatology is None:
        climatology = hgt_data.groupby(
            hgt_data[time_field].dt.dayofyear).mean(time_field)

    anom = hgt_data.groupby(
        hgt_data[time_field].dt.dayofyear) - climatology

    return anom, climatology


def project_data(anom_data, composite):
    n_samples = anom_data.shape[0]
    n_features = np.product(anom_data.shape[1:])

    flat_data = np.reshape(anom_data, (n_samples, n_features))
    flat_composites = np.reshape(composite, (1, n_features))

    a = np.dot(flat_composites, flat_composites.T)
    b = np.dot(flat_composites, flat_data.T)

    sol = np.linalg.lstsq(a, b, rcond=None)[0]

    return sol.T


def calculate_indices(anom_data, composites,
                      time_field=DEFAULT_TIME_FIELD):
    n_samples = anom_data.sizes[time_field]
    n_clusters = composites.sizes[CLUSTER_DIM_NAME]

    indices = np.empty((n_samples, n_clusters))

    for i in range(n_clusters):
        composite_data = composites.isel({CLUSTER_DIM_NAME: i})
        projection = project_data(anom_data.values,
                                  composite_data.values)
        indices[:, i] = np.squeeze(projection)

    indices_dims = (time_field, CLUSTER_DIM_NAME)

    indices_coords = {time_field: anom_data[time_field].values,
                      CLUSTER_DIM_NAME: np.arange(n_clusters)}

    indices_da = xr.DataArray(indices, dims=indices_dims,
                              coords=indices_coords)

    return indices_da


def write_indices(indices, output_file=None,
                  time_field=DEFAULT_TIME_FIELD):
    if output_file is None or not output_file:
        return

    years = indices[time_field].dt.year.values
    months = indices[time_field].dt.month.values
    days = indices[time_field].dt.day.values
    index_data = indices.values

    fields = [('year', '%d'),
              ('month', '%d'),
              ('day', '%d')]

    n_samples, n_indices = index_data.shape
    for i in range(n_indices):
        fields += [('index_{:d}'.format(i), '%14.8e')]

    n_fields = len(fields)
    header = ','.join([f[0] for f in fields])
    fmt = ','.join([f[1] for f in fields])

    data = np.empty((n_samples, n_fields))
    data[:, 0] = years
    data[:, 1] = months
    data[:, 2] = days
    for i in range(n_indices):
        data[:, i + 3] = index_data[:, i]

    np.savetxt(output_file, data, header=header, fmt=fmt)


def plot_eofs(eofs_da, explained_variances, output_file=None,
              show_plot=True, lat_field=DEFAULT_LAT_FIELD,
              lon_field=DEFAULT_LON_FIELD, wrap_lon=True):
    lat = eofs_da[lat_field]
    lon = eofs_da[lon_field]
    eofs_data = eofs_da.values

    if wrap_lon:
        if lon.min() >= 0:
            cyclic_lon = np.full(np.size(lon) + 1, 360.)
            cyclic_lon[:-1] = lon
            lon = cyclic_lon
        else:
            cyclic_lon = np.full(np.size(lon) + 1, -180.)
            cyclic_lon[:-1] = lon
            lon = cyclic_lon
        eofs_data = add_cyclic_point(eofs_data)

    lon_grid, lat_grid = np.meshgrid(lon, lat)

    n_eofs = eofs_da.sizes[EOF_DIM_NAME]

    if output_file is not None and output_file:
        pdf = PdfPages(output_file)
    else:
        pdf = None

    n_pages = int(np.ceil(n_eofs / (N_COLS * N_ROWS)))
    proj = ccrs.Orthographic(central_longitude=0, central_latitude=90.0)

    eof_index = 0
    for pg in range(n_pages):
        fig = plt.figure(figsize=FIGURE_SIZE)
        gs = GridSpec(N_ROWS, N_COLS, figure=fig,
                      wspace=0, hspace=0.25,
                      left=0.15, right=0.85)

        for row_idx in range(N_ROWS):
            for col_idx in range(N_COLS):
                if eof_index >= n_eofs:
                    break

                eof = np.squeeze(eofs_data[eof_index])

                ax = fig.add_subplot(gs[row_idx, col_idx], projection=proj)

                ax.coastlines()
                ax.set_global()

                cs = ax.contourf(lon_grid, lat_grid, eof, cmap=CMAP,
                                 transform=ccrs.PlateCarree())

                ax.set_aspect('equal')

                cb = fig.colorbar(cs)

                ax.set_title(r'EOF {:d} ({:.2f}%)'.format(
                    eof_index + 1, 100 * explained_variances[eof_index]))

                eof_index += 1

        if pdf is not None:
            pdf.savefig()

        if show_plot:
            plt.show()

        plt.close()

    if pdf is not None:
        pdf.close()


def plot_composites(composites_da, output_file=None,
                    show_plot=True, lat_field=DEFAULT_LAT_FIELD,
                    lon_field=DEFAULT_LON_FIELD, wrap_lon=True):
    lat = composites_da[lat_field]
    lon = composites_da[lon_field]
    composites_data = composites_da.values

    if wrap_lon:
        if lon.min() >= 0:
            cyclic_lon = np.full(np.size(lon) + 1, 360.)
            cyclic_lon[:-1] = lon
            lon = cyclic_lon
        else:
            cyclic_lon = np.full(np.size(lon) + 1, -180.)
            cyclic_lon[:-1] = lon
            lon = cyclic_lon
        print(lon)
        composites_data = add_cyclic_point(composites_data)

    lon_grid, lat_grid = np.meshgrid(lon, lat)

    n_clusters = composites_da.sizes[CLUSTER_DIM_NAME]

    if output_file is not None and output_file:
        pdf = PdfPages(output_file)
    else:
        pdf = None

    n_pages = int(np.ceil(n_clusters / (N_COLS * N_ROWS)))
    proj = ccrs.Orthographic(central_longitude=0, central_latitude=90.0)

    composite_index = 0
    for pg in range(n_pages):
        fig = plt.figure(figsize=FIGURE_SIZE)
        gs = GridSpec(N_ROWS, N_COLS, figure=fig,
                      wspace=0, hspace=0.25,
                      left=0.15, right=0.85)

        for row_idx in range(N_ROWS):
            for col_idx in range(N_COLS):
                if composite_index >= n_clusters:
                    break

                composite = np.squeeze(composites_data[composite_index])

                ax = fig.add_subplot(gs[row_idx, col_idx], projection=proj)

                ax.coastlines()
                ax.set_global()

                cs = ax.contourf(lon_grid, lat_grid, composite, cmap=CMAP,
                                 transform=ccrs.PlateCarree())

                ax.set_aspect('equal')

                cb = fig.colorbar(cs)

                ax.set_title(r'Cluster {:d}'.format(
                    composite_index + 1))

                composite_index += 1

        if pdf is not None:
            pdf.savefig()

        if show_plot:
            plt.show()

        plt.close()

    if pdf is not None:
        pdf.close()


def parse_cmd_line_args():
    parser = argparse.ArgumentParser(
        description='Plot composites from k-means affiliations')

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
        '--season', dest='season', choices=VALID_SEASONS,
        default=DEFAULT_SEASON, help='season to perform analysis in')
    parser.add_argument(
        '--n-eofs', dest='n_eofs', type=int,
        default=DEFAULT_N_EOFS, help='number of EOFs')
    parser.add_argument(
        '--n-clusters', dest='n_clusters', type=int,
        default=DEFAULT_N_CLUSTERS, help='number of clusters')
    parser.add_argument(
        '--n-init', dest='n_init', type=int,
        default=DEFAULT_N_INIT, help='number of initializations')
    parser.add_argument(
        '--eof-output-file', dest='eof_output_file',
        default='', help='name of file to write EOF to')
    parser.add_argument(
        '--pcs-output-file', dest='pcs_output_file',
        default='', help='name of file to write PCs to')
    parser.add_argument(
        '--eof-plot-output-file', dest='eof_plot_output_file',
        default='', help='name of file to write EOF plots to')
    parser.add_argument(
        '--composite-output-file', dest='composite_output_file',
        default='', help='name of file to write composites to')
    parser.add_argument(
        '--composite-plot-output-file', dest='composite_plot_output_file',
        default='', help='name of file to write composite plots to')
    parser.add_argument(
        '--no-show-plots', dest='no_show_plots', action='store_true',
        help='do not show plots')
    parser.add_argument(
        '--index-output-file', dest='index_output_file',
        default='', help='file to write indices to')

    return parser.parse_args()


def main():
    args = parse_cmd_line_args()

    with xr.open_dataset(args.datafile) as ds:
        hgt_data = ds[args.hgt_field]

        ref_data = hgt_data.where(
            (hgt_data[args.time_field].dt.year >= args.start_year) &
            (hgt_data[args.time_field].dt.year <= args.end_year),
            drop=True)

        filtered_data = ref_data.rolling(
            {args.time_field: 10}).mean().dropna(args.time_field)

        anom_data, clim_data = calculate_anomalies(
            filtered_data, time_field=args.time_field)

        lat_bounds = np.array([20.0, 90.0])
        lon_bounds = np.array([[0, 30], [270, 360]])

        eofs_results = calculate_seasonal_eofs(
            anom_data, season=args.season,
            lat_bounds=lat_bounds, lon_bounds=lon_bounds,
            n_eofs=args.n_eofs, time_field=args.time_field,
            lat_field=args.lat_field, lon_field=args.lon_field,
            var_field=args.hgt_field)

        if args.eof_output_file:
            eofs_results['eofs'].to_netcdf(args.eof_output_file)

        if args.pcs_output_file:
            eofs_results['pcs'].to_netcdf(args.pcs_output_file)

        if not args.no_show_plots or args.eof_plot_output_file:
            plot_eofs(eofs_results['eofs'],
                      eofs_results['explained_variance_ratio'],
                      output_file=args.eof_plot_output_file,
                      show_plot=(not args.no_show_plots),
                      lat_field=args.lat_field,
                      lon_field=args.lon_field,
                      wrap_lon=False)

        cluster_results = cluster_pcs(
            eofs_results['pcs'], time_field=args.time_field,
            n_clusters=args.n_clusters, n_init=args.n_init)

        composites = calculate_composites(
            anom_data, cluster_results['labels'],
            season=args.season,
            time_field=args.time_field)

        if args.composite_output_file:
            composites.to_netcdf(args.composite_output_file)

        if not args.no_show_plots or args.composite_plot_output_file:
            plot_composites(
                composites, output_file=args.composite_plot_output_file,
                show_plot=(not args.no_show_plots), lat_field=args.lat_field,
                lon_field=args.lon_field, wrap_lon=True)

        daily_anom_data, _ = calculate_anomalies(
            ref_data, time_field=args.time_field)
        indices = calculate_indices(
            daily_anom_data, composites, time_field=args.time_field)

        if args.index_output_file:
            write_indices(indices, args.index_output_file,
                          time_field=args.time_field)


if __name__ == '__main__':
    main()
