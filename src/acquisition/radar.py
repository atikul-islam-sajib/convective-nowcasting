import os
import requests
import tarfile
import io
import os
import numpy as np
import wradlib as wrl
import re
from tqdm import tqdm


def download_dwd_5_minutes_monthly_tars(years, outdir):
    BASE_URL = (
        "https://opendata.dwd.de/climate_environment/"
        "CDC/grids_germany/5_minutes/radolan/reproc"
    )
    VERSION_DIR = "2017_002"
    VERSION_TAG = "2017.002"

    for year in years:
        year_dir = os.path.join(outdir, str(year))
        os.makedirs(year_dir, exist_ok=True)

        for month in range(1, 13):
            yyyymm = f"{year}{month:02d}"
            fname = f"YW{VERSION_TAG}_{yyyymm}.tar"
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
                print(f"✗ Failed to fetch {fname}: {e}")
                continue

            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
            print(f"✔ Saved {fname} to {year_dir}")


def extract_dwd_tarballs(input_dir, output_dir):
    for year in sorted(os.listdir(input_dir)):
        year_in = os.path.join(input_dir, year)
        if not os.path.isdir(year_in):
            continue
        
        file_list = sorted(os.listdir(year_in))
        for fname in file_list:
            if not fname.endswith(".tar"):
                continue

            month_tar_path = os.path.join(year_in, fname)
            with tarfile.open(month_tar_path, "r") as month_tar:
                with tqdm(total=len(month_tar.getmembers()), desc=f"Processing year {year}, month {fname}", unit="file") as pbar:
                    
                    for member in month_tar.getmembers():
                        if not member.name.endswith(".tar.gz"):
                            pbar.update(1)
                            continue

                        buf = month_tar.extractfile(member)
                        if buf is None:
                            pbar.update(1)
                            continue

                        daily_name = os.path.basename(member.name)
                        date_str = daily_name.split("_")[1].replace(".tar.gz", "")
                        daily_out_dir = os.path.join(output_dir, year, date_str)
                        os.makedirs(daily_out_dir, exist_ok=True)

                        with tarfile.open(fileobj=io.BytesIO(buf.read()), mode="r:gz") as daily_tar:
                            daily_tar.extractall(path=daily_out_dir)
                            pbar.update(1)

            pbar.update(1)


def dwd_bin2numpy(input_dir, output_dir, precision='float16'):
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(r"raa01-yw2017\.002_10000-(\d{10})-dwd---bin$")

    file_list = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            file_list.append((root, fname))

    num_zero_arrays = 0
    with tqdm(total=len(file_list), desc="Processing files", unit="file") as pbar:
        for root, fname in file_list:
            m = pattern.match(fname)
            if not m:
                pbar.update(1)
                continue

            yymmddhhmm = m.group(1)      # yymmddHHMM
            hhmm = yymmddhhmm[-4:]       # HHMM
            date_str = yymmddhhmm[:6]    # yymmdd

            year = date_str[:4]          # yyyy
            month = date_str[4:6]        # mm
            day = date_str[6:8]          # dd

            out_subdir = os.path.join(output_dir, year, month, day)
            os.makedirs(out_subdir, exist_ok=True)
            out_file = os.path.join(out_subdir, f"{date_str}_{hhmm}.npy")

            if os.path.exists(out_file):
                pbar.update(1)
                continue

            fpath = os.path.join(root, fname)
            data, meta = wrl.io.read_radolan_composite(fpath)

            arr = data.astype(np.float32).flatten()
            arr[meta['nodatamask']] = np.nan
            arr[meta['cluttermask']] = 0.0
            arr = arr.reshape((meta['nrow'], meta['ncol']))

            if np.isclose(np.nansum(arr), 0):
                num_zero_arrays += 1

            np.save(out_file, arr.astype(precision), allow_pickle=False)

            pbar.update(1)

        print(f"All data processed. Found [{num_zero_arrays}/{len(file_list)}]={num_zero_arrays/len(file_list)*100:.2f}%")


if __name__=="__main__":
    download_dwd_5_minutes_monthly_tars(
        years=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024], outdir="data/radar_tar_germany"
    )
    extract_dwd_tarballs(
        input_dir="data/radar_tar_germany",
        output_dir="data/radar_extracted_germany"
    )
    dwd_bin2numpy(
        input_dir="data/radar_extracted_germany",
        output_dir="data/germany_fp16/radar"
    )