from glob import glob
import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pdb import set_trace
import pyproj
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds
from re import sub
from shapely.geometry import box
## import json
## from tqdm import tqdm
## import glob
## from pathlib import Path
## from datetime import datetime
## import multiprocessing

from aadata import loader
from aautils import geometry

def read_tif(tif_file, bbox, sample_distance_km):
    with rasterio.open(tif_file) as src:
        print(f"Processing: {tif_file}")
        print(f"Original CRS: {src.crs}")

        # Transform the bounding box to the raster's CRS
        bbox_transformed = transform_bounds("EPSG:4326", src.crs, *bbox)

        # Create a GeoDataFrame from the bounding box
        gdf = gpd.GeoDataFrame({'geometry': [box(*bbox_transformed)]}, crs=src.crs)

        # Clip the raster
        clipped, clip_transform = mask(src, gdf.geometry, crop=True)

        # Create a transformer to convert from raster CRS to EPSG:4326
        transformer = pyproj.Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

        # Get the clipped data
        tifdata = clipped[0]  # Get the first (and only) band

        # Calculate sampling rate
        resolution = src.res[0]  # assuming square pixels
        sample_rate = 1 #max(1, int(sample_distance_km * 1000 / resolution))

        # Optimize loop processing
        rows, cols = tifdata.shape
        row_indices, col_indices = np.mgrid[0:rows:sample_rate, 0:cols:sample_rate]
        valid_mask = ~np.isnan(tifdata[row_indices, col_indices])
        
        # Get coordinates for valid points
        x_coords, y_coords = rasterio.transform.xy(clip_transform, 
                                                   row_indices[valid_mask], 
                                                   col_indices[valid_mask])
        lon, lat = transformer.transform(x_coords, y_coords)
        
        # Get valid abundance data
        valid_data = tifdata[row_indices[valid_mask], col_indices[valid_mask]]
        df = pd.DataFrame({'lon': lon, 'lat': lat, 'value': valid_data})
        xx = geometry.lonlat_to_glw(df.lon, df.lat)
        df['x'] = [l[0] for l in xx]
        df['y'] = [l[1] for l in xx]
        df.loc[df.value<0, 'value'] = 0
        df = df.groupby(['x', 'y'], as_index=False)['value'].max()

    return df


if __name__ == '__main__':
    ## xx = pd.read_parquet('bird_prevalence.parquet')
    ## set_trace()
    dfl = []
    for f in glob('../birds_prevalence/Month_*tif'):
        month = sub('Month_', '', os.path.basename(f))
        month = int(sub('.tif', '', month))
        # states = loader.load('usa_state_shapes', contiguous_us=True)
        df = read_tif(f, bbox=(-170, 20, -50, 75), sample_distance_km=3)
        df['month'] = month 
        dfl.append(df)

    df = pd.concat(dfl)
    df.to_parquet('bird_h5_prevalence.parquet')

