import os
import io
import re
import tarfile
import requests
import numpy as np
import wradlib as wrl
from tqdm import tqdm

products = {
    "YW": {
        "freq": "5_minutes",
        "file_ending": "tar",
        "pattern_extracted": "%Y/%y%m%d%H/raa01-yw2017.002_10000-%y%m%d%H%M-dwd---bin",
        "pattern_npy": "%y%m/%d/%y%m%d_%H%M.npy",
        "multiplication_factor": 12.0,
    },
    "RW": {
        "freq": "hourly",
        "file_ending": "tar.gz",
        "pattern_extracted": "%Y/%y%m%d%H/raa01-rw2017.002_10000-%y%m%d%H%M-dwd---bin",
        "pattern_npy": "%y%m/%d/%y%m%d_%H%M.npy",
        "multiplication_factor": 1.0,
    },
}


def download_dwd_5_minutes_monthly_tars(years, outdir, product="YW"):
    """Download RADOLAN monthly .tar files into per-year subfolders."""
    assert product in products, f"product must be one of {list(products)}"

    BASE_URL = f"https://opendata.dwd.de/climate_environment/CDC/grids_germany/{products[product]['freq']}/radolan/reproc"
    VERSION_DIR = "2017_002"
    VERSION_TAG = "2017.002"

    for year in years:
        year_dir = os.path.join(outdir, str(year))
        os.makedirs(year_dir, exist_ok=True)

        for month in range(1, 13):
            yyyymm = f"{year}{month:02d}"
            fname = f"{product}{VERSION_TAG}_{yyyymm}.{products[product]['file_ending']}"
            url = f"{BASE_URL}/{VERSION_DIR}/bin/{year}/{fname}"
            out_path = os.path.join(year_dir, fname)

            if os.path.exists(out_path):
                print(f"✔ Skipping existing {fname} in {year}")
                continue

            print(f"→ Downloading {fname} for {year} …")
            resp = requests.get(url, stream=True)
            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                print(f" Failed to fetch {fname}: {e}")
                continue

            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            print(f" Saved {fname} to {year_dir}")


def extract_dwd_tarballs(input_dir, year, output_dir):
    """Extract daily .bin files from the monthly .tar archives."""
    year_in = os.path.join(input_dir, f"{year}")
    if not os.path.isdir(year_in):
        print(f"Folder of year {year} not found in tar-folder")
        return

    for fname in sorted(os.listdir(year_in)):
        if not (fname.endswith(".tar") or fname.endswith(".tar.gz")):
            continue

        month_tar_path = os.path.join(year_in, fname)
        mode = "r" if fname.endswith(".tar") else "r:gz"
        with tarfile.open(month_tar_path, mode) as month_tar:
            with tqdm(total=len(month_tar.getmembers()),
                      desc=f"Processing year {year}, {fname}", unit="file") as pbar:
                for member in month_tar.getmembers():
                    name = os.path.basename(member.name)

                    # YW: outer .tar contains inner daily .tar.gz
                    if name.endswith(".tar.gz"):
                        buf = month_tar.extractfile(member)
                        if buf is None:
                            pbar.update(1)
                            continue
                        date_str = name.split("_")[1].replace(".tar.gz", "")
                        daily_out_dir = os.path.join(output_dir, f"{year}", date_str)
                        os.makedirs(daily_out_dir, exist_ok=True)
                        with tarfile.open(fileobj=io.BytesIO(buf.read()), mode="r:gz") as daily_tar:
                            daily_tar.extractall(path=daily_out_dir)
                        pbar.update(1)
                        continue

                    if name.endswith("bin"):
                        date_match = re.search(r"\d{8}", name)
                        date_str = date_match.group(0) if date_match else "unknown"
                        daily_out_dir = os.path.join(output_dir, f"{year}", date_str)
                        os.makedirs(daily_out_dir, exist_ok=True)
                        month_tar.extract(member, path=daily_out_dir)
                        pbar.update(1)
                        continue

                    pbar.update(1)


def dwd_bin2numpy(input_dir, output_dir, precision="float16",
                   remove_source=False, multiplication_factor=1.0):
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(r"raa01-(?:yw|rw)2017\.002_10000-(\d{10})-dwd---bin$", re.IGNORECASE)

    file_list = [(root, fname) for root, _, files in os.walk(input_dir) for fname in files]

    num_zero_arrays = 0
    with tqdm(total=len(file_list), desc="Processing files", unit="file") as pbar:
        for root, fname in file_list:
            m = pattern.match(fname)
            if not m:
                pbar.update(1)
                continue

            yymmddhhmm = m.group(1)
            hhmm = yymmddhhmm[-4:]
            date_str = yymmddhhmm[:6]
            year, month, day = date_str[:4], date_str[4:6], date_str[6:8]

            out_subdir = os.path.join(output_dir, year, month, day)
            os.makedirs(out_subdir, exist_ok=True)
            out_file = os.path.join(out_subdir, f"{date_str}_{hhmm}.npy")

            if os.path.exists(out_file):
                pbar.update(1)
                continue

            fpath = os.path.join(root, fname)
            data, meta = wrl.io.read_radolan_composite(fpath)

            arr = data.astype(np.float32).flatten()
            arr[meta["nodatamask"]] = np.nan
            arr[meta["cluttermask"]] = 0.0
            arr = arr.reshape((meta["nrow"], meta["ncol"]))

            if multiplication_factor != 1.0:
                arr *= multiplication_factor

            if np.isclose(np.nansum(arr), 0):
                num_zero_arrays += 1

            np.save(out_file, arr.astype(precision), allow_pickle=False)

            if remove_source:
                try:
                    os.remove(fpath)
                except Exception as e:
                    print(f"Warning: could not remove source file '{fpath}': {e}")

            pbar.update(1)

    print(f"All data processed. Zero-arrays: [{num_zero_arrays}/{len(file_list)}]"
          f"={num_zero_arrays/len(file_list)*100:.2f}%")


if __name__ == "__main__":
    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    product = "YW"  

    download_folder = f"data/{product}_tar"
    extraction_folder = f"data/{product}_extracted"
    numpy_folder = f"data/{product}_npy"

    factor = products[product]["multiplication_factor"] 

    for year in years:
        download_dwd_5_minutes_monthly_tars(years=[year], outdir=download_folder, product=product)
        extract_dwd_tarballs(input_dir=download_folder, year=year, output_dir=extraction_folder)
        dwd_bin2numpy(
            input_dir=f"{extraction_folder}/{year}",
            output_dir=numpy_folder + "/radar",
            multiplication_factor=factor,  
            remove_source=True,
        )