import os
import shutil
import logging


SRC_FOLDER = "/home/sajib/Desktop/nowcasting_medewsa/redownloaded_missing_npy"
DST_FOLDER = "/home/sajib/Desktop/npz-again"

logging.basicConfig(
    filename="organize_missing_npz.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def organize_npy_by_date(src_folder, dst_folder):
    """
    Same logic as your original function, but moves files into a different folder.
    No changes to filename parsing or structure.
    """

    os.makedirs(dst_folder, exist_ok=True)

    for file in os.listdir(src_folder):
        if file.endswith(".npz"):
            try:
                parts = file.split("_")
                timestamp_part = parts[2]
                date_str = timestamp_part[:8]

                date_folder = os.path.join(dst_folder, date_str)
                os.makedirs(date_folder, exist_ok=True)

                shutil.move(
                    os.path.join(src_folder, file),
                    os.path.join(date_folder, file)
                )

                logging.info(f"Moved {file} to {date_folder}")

            except Exception as e:
                logging.error(f"Filename parsing failed for {file}: {e}")


if __name__ == "__main__":
    organize_npy_by_date(SRC_FOLDER, DST_FOLDER)