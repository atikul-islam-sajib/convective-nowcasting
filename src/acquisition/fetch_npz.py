import os
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_file(npz_path, output_root):
    try:
        filename = os.path.basename(npz_path)
        parts = filename.split("_")
        date_str = parts[2]       
        time_str = parts[3][:4]   
        timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M")
        full_time_str = timestamp.strftime("%H%M%S")  
    except Exception as e:
        return f"Skipping {filename} due to parse error: {e}"

    try:
        with np.load(npz_path, allow_pickle=True) as data:
            all_data = data['data']
            tags = data['tags'].item() if 'tags' in data else {}

            for i, ch in enumerate([7, 9]):
                ch_data = all_data[i]
                folder_path = os.path.join(
                    output_root,
                    timestamp.strftime("%Y"),
                    timestamp.strftime("%m"),
                    timestamp.strftime("%d")
                )
                os.makedirs(folder_path, exist_ok=True)

                filename_out = f"{full_time_str}_CH{ch}.npz"
                out_path = os.path.join(folder_path, filename_out)

                np.savez_compressed(
                    out_path,
                    data=ch_data.astype(np.float32),
                    crs=data['crs'],
                    transform=data['transform'],
                    nodata=data['nodata'] if 'nodata' in data else None,
                    dtype=data['dtype'],
                    width=data['width'],
                    height=data['height'],
                    bounds=data['bounds'],
                    filename=filename_out,
                    tags=tags
                )
        return f"Processed: {filename}"
    except Exception as e:
        return f"Failed to process {filename}: {e}"

def convert_january_month_parallel(january_folder, output_root, max_workers=8):
    all_npz_files = []

    for root, _, files in os.walk(january_folder):
        for f in files:
            if (
                f.endswith(".npz")
                and 'CH7_CH9' in f
                and not f.startswith("._")  
            ):
                all_npz_files.append(os.path.join(root, f))

    print(f"Found {len(all_npz_files)} .npz files for parallel processing...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, file, output_root) for file in all_npz_files]
        for future in as_completed(futures):
            print(future.result())

convert_january_month_parallel(
    january_folder="/home/sajib/Desktop/npz-again",
    output_root="/home/sajib/Desktop/npz-to-npy",
    max_workers=8
)


#ssh sajib@h9yrm62
#rsync -avz --progress /home/sajib/Desktop/satellite_europe_npz/ sajib@h9yrm62:/data/satellite_europe_npz/
#rsync -avz --progress /home/sajib/Desktop/satellite_europe_npz/2024/09/ sajib@h9yrm62:/data/satellite_europe_npz/2024/09
#rsync -avz --progress sajib@h9yrm62:/data/germany_npz_18_19_23/{2019,2023} /home/sajib/Desktop/     -> to download the data