import eumdac
import datetime
import shutil
import fnmatch
import rasterio
import os
import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as ResampleEnum
from affine import Affine
from pyproj import CRS, Transformer
from tqdm import tqdm
import concurrent.futures
import logging
from threading import Semaphore

matplotlib.use('TkAgg')

### Logging Setup ###
logging.basicConfig(
    filename="download_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


CONSUMER_KEY = "WKG3zga7URJWbu0QKFPnaZXvpyUa"
CONSUMER_SECRET = "pmhjLEAKjFFsbahQ7_KInpKkDUAa"

DATASET_COLLECTION = "EO:EUM:DAT:MSG:MSG15-RSS"
DATASET_ID = "EO:EUM:DAT:MSG:HRSEVIRI"
BBOX = (5.99, 16.15, 47.28, 55.03)  
DATA_FOLDER = "./satelite_np_germany_2024"

START_TIME = datetime.datetime(2023, 11, 1, 0, 0)
END_TIME = datetime.datetime(2023, 11, 30, 23, 59)


def authenticate(key, secret):
    credentials = (key, secret)
    token = eumdac.AccessToken(credentials, validity=86400, cache=False)
    store = eumdac.DataStore(token)
    logging.info(f"Token expires at {token.expiration}")
    return store, token

def search_products(datastore, collection, id, start, end):
    selected_collection = datastore.get_collection(collection)
    products = selected_collection.search(dtstart=start, dtend=end)
    logging.info(f"Found Datasets: {products.total_results} datasets for the given time range")
    return products

def get_product(datastore, product, collection):
    return datastore.get_product(product_id=str(product), collection_id=collection)

def customization(datatailor, product):
    chain = eumdac.tailor_models.Chain(
        product='HRSEVIRI_RSS',
        format='geotiff',
        filter={"bands": ['channel_7', 'channel_9']},
        roi='germany',
    )
    return datatailor.new_customisation(product, chain)

def wait_for_customisation(customisation, product, sleep_time=30):
    while True:
        status = customisation.status
        if "DONE" in status:
            return True
        elif status in ["ERROR", "FAILED", "DELETED", "KILLED", "INACTIVE"]:
            logging.warning(f"Customisation {customisation._id} failed for {product._id} with status {status}.")
            return False
        time.sleep(sleep_time)

def download(customisation, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    zip_file = fnmatch.filter(customisation.outputs, '*.tif')[0]
    original_path = os.path.join(output_folder, os.path.basename(zip_file))
    with customisation.stream_output(zip_file) as stream, open(original_path, 'wb') as f:
        shutil.copyfileobj(stream, f)
    return original_path

def get_lcc_transform_and_shape(bbox_wgs84, resolution=1000):
    wgs84 = CRS.from_epsg(4326)
    lcc_crs = CRS.from_proj4(
        "+proj=lcc +lat_1=48 +lat_2=54 +lat_0=51 +lon_0=10 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
    transformer = Transformer.from_crs(wgs84, lcc_crs, always_xy=True)
    min_lon, max_lon, min_lat, max_lat = bbox_wgs84
    x_min, y_min = transformer.transform(min_lon, min_lat)
    x_max, y_max = transformer.transform(max_lon, max_lat)
    fixed_width = 501
    fixed_height = 301
    transform = Affine.translation(x_min, y_max) * Affine.scale(
        (x_max - x_min) / fixed_width, -(y_max - y_min) / fixed_height)
    return transform, fixed_width, fixed_height, lcc_crs

def rescale_brightness(cropped_data):
    c1 = 1.19104e-5
    c2 = 1.43877
    center_wavelengths = [1e4 / x for x in [6.2, 7.3, 8.7, 10.8]]
    rescaled = np.empty_like(cropped_data)
    for i, v in enumerate(center_wavelengths):
        rescaled[i] = c2 * v / np.log(1 + c1 * v**3 / cropped_data[i])
    return rescaled

def save_geotiff(output_path, data, meta):
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=meta['height'],
        width=meta['width'],
        count=meta['count'],
        dtype=meta['dtype'],
        crs=meta['crs'],
        transform=meta['transform']
    ) as dst:
        for i in range(meta['count']):
            dst.write(data[i], i + 1)

def normalize(data, method="min-max"):
    normalized = np.empty_like(data, dtype=np.float32)
    for i in range(data.shape[0]):
        band = data[i]
        if method == "min-max":
            band_min = np.nanmin(band)
            band_max = np.nanmax(band)
            normalized[i] = (band - band_min) / (band_max - band_min) if band_max > band_min else band
        elif method == "z-score":
            mean = np.nanmean(band)
            std = np.nanstd(band)
            normalized[i] = (band - mean) / std if std > 0 else band
    return normalized

def plot_band(tif_path, band_index=1, save_folder="./plots"):
    os.makedirs(save_folder, exist_ok=True)
    with rasterio.open(tif_path) as src:
        data = src.read(band_index)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    plt.imshow(data, cmap='gray_r', extent=extent, origin='upper')
    plt.colorbar(label='Brightness Temperature')
    plt.title(f"{os.path.basename(tif_path)} - Band {band_index}")
    plt.savefig(os.path.join(save_folder, os.path.basename(tif_path).replace(".tif", f"_band{band_index}.png")))
    plt.close()

def safe_delete_customisation(c, retries=3, delay=5):
    for attempt in range(retries):
        try:
            c.delete()
            return True
        except Exception as e:
            if "500 Server Error" in str(e):
                logging.warning(f"Delete failed for {c._id} (attempt {attempt+1}/{retries})")
                time.sleep(delay)
            else:
                logging.error(f"Unexpected error on delete for {c._id}: {e}")
                return False
    return False

def clean_up_all(datatailor):
    for c in datatailor.customisations:
        if c.status in ['DONE', 'FAILED', 'KILLED', 'DELETED', 'QUEUED', 'INACTIVE']:
            safe_delete_customisation(c)
    logging.info('Complete Datatailor cleaned!')

def safe_customization(datatailor, product, max_customisations=3, sleep_time=10):
    while True:
        active = sum(1 for c in datatailor.customisations if c.status in ['QUEUED', 'RUNNING'])
        if active < max_customisations:
            try:
                return customization(datatailor, product)
            except Exception as e:
                if "Runtime Error" in str(e) and "maximum number" in str(e):
                    logging.info(f"Retrying customization due to limit: {e}")
                    time.sleep(sleep_time)
                    continue
                raise
        logging.info(f"Waiting: {active} customisations active, limit is {max_customisations}")
        time.sleep(sleep_time)

def process_product(product, datastore, datatailor, collection, output_folder):
    try:
        single_product = get_product(datastore, product, collection)
        single_customisation = safe_customization(datatailor, single_product)
        if wait_for_customisation(single_customisation, single_product):
            tif_path = download(single_customisation, output_folder)
            try:
                base = os.path.basename(tif_path)
                try:
                    start_dt_part = base.split("T")[1].split("Z")[0]
                    date_part = base.split("_")[2][:8]
                    hhmm = start_dt_part[:4]
                    final_name = f"HRSEVIRI_RSS_{date_part}_{hhmm}_CH7_CH9.npz"
                except Exception:
                    final_name = base.replace(".tif", ".npz")

                final_path = os.path.join(output_folder, final_name)

                if os.path.exists(final_path):
                    logging.info(f"Skipping {final_name} – already exists.")
                    return True

                with rasterio.open(tif_path) as src:
                    data = src.read()
                    profile = src.profile
                    transform = tuple(src.transform.to_gdal())
                    crs = src.crs.to_wkt() if src.crs else None
                    bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
                    tags = src.tags()

                np.savez_compressed(
                    final_path,
                    data=data.astype(np.float32),
                    crs=crs,
                    transform=transform,
                    nodata=profile.get("nodata"),
                    dtype=str(profile["dtype"]),
                    width=profile["width"],
                    height=profile["height"],
                    count=profile["count"],
                    bounds=np.array(bounds),
                    filename=final_name,
                    tags=tags,
                )
                logging.info(f"Saved: {final_name}")
                return True
            finally:
                if os.path.exists(tif_path):
                    os.remove(tif_path)
        return False
    except Exception as e:
        if "Not found (404)" in str(e):
            logging.warning(f"Skipping product {product}: File not found (404)")
            return False
        logging.error(f"Failed to process product {product}: {e}")
        return False

def parallel_download_and_process(datastore, datatailor, collection, products, output_folder, max_workers=15, max_customisations=3):
    os.makedirs(output_folder, exist_ok=True)
    remaining = list(products)
    attempt = 0
    while remaining:
        logging.info(f"Attempt {attempt + 1}: Processing {len(remaining)} products")
        successful = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_product = {
                executor.submit(process_product, p, datastore, datatailor, collection, output_folder): p
                for p in remaining
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_product), total=len(remaining), desc="Processing Products"):
                product = future_to_product[future]
                if future.result():
                    successful.append(product)
        remaining = [p for p in remaining if p not in successful]
        attempt += 1
        logging.info(f"Attempt {attempt} completed: {len(successful)} successful, {len(remaining)} remaining")
        if attempt >= 3:
            logging.warning(f"Giving up after {attempt} attempts. {len(remaining)} products failed: {remaining}")
            break

def organize_npy_by_date(data_folder):
    for file in os.listdir(data_folder):
        if file.endswith(".npz"):
            try:
                parts = file.split("_")
                timestamp_part = parts[2]
                date_str = timestamp_part[:8]
                date_folder = os.path.join(data_folder, date_str)
                os.makedirs(date_folder, exist_ok=True)
                shutil.move(os.path.join(data_folder, file), os.path.join(date_folder, file))
                logging.info(f"Moved {file} to {date_folder}")
            except Exception as e:
                logging.error(f"Filename parsing failed for {file}: {e}")

def main():
    start_time = time.time()
    try:
        datastore, token = authenticate(CONSUMER_KEY, CONSUMER_SECRET)
        datatailor = eumdac.DataTailor(token)
        products = search_products(datastore, DATASET_COLLECTION, DATASET_ID, START_TIME, END_TIME)
        parallel_download_and_process(datastore, datatailor, DATASET_COLLECTION, products, DATA_FOLDER)
        organize_npy_by_date(DATA_FOLDER)
        clean_up_all(datatailor)
    except Exception as e:
        logging.error(f"Main process failed: {e}")
        raise
    finally:
        logging.info(f"Done. Runtime: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()