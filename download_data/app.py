import os
import subprocess
from pathlib import Path
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import threading
from tqdm import tqdm

# Load credentials from .env file
load_dotenv()
username = os.getenv("PHYSIONET_USERNAME")
password = os.getenv("PHYSIONET_PASSWORD")
if not username or not password:
    raise EnvironmentError(
        "Please set PHYSIONET_USERNAME and PHYSIONET_PASSWORD in your .env file"
    )


def list_patients(base_url, partition, username, password):
    """
    Uses wget to fetch the partition index and parses patient directory names.
    """
    index_url = f"{base_url}/files/{partition}/"
    cmd = [
        "wget",
        "-q",
        "-O",
        "-",
        f"--user={username}",
        f"--password={password}",
        index_url,
    ]
    html = subprocess.check_output(cmd, text=True)
    soup = BeautifulSoup(html, "html.parser")
    patients = [
        a["href"].rstrip("/")
        for a in soup.find_all("a", href=True)
        if a["href"].endswith("/") and a["href"].startswith("p")
    ]
    return sorted(patients)


def download_patient(base_url, partition, patient, username, password, output_dir):
    """
    Downloads a single patient's directory using wget, suppressing stdout and capturing errors.
    """
    patient_url = f"{base_url}/files/{partition}/{patient}/"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget",
        "-r",
        "-N",
        "-c",
        "-np",
        "-nH",
        "--cut-dirs=5",
        f"--user={username}",
        f"--password={password}",
        patient_url,
        "-P",
        str(output_dir),
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to download {patient_url}: {result.stderr.decode(errors='ignore')}"
        )


if __name__ == "__main__":
    # Configuration
    base_url = "https://physionet.org/files/mimic-cxr/2.1.0"
    partition = "p10"  # hardcode the pXX partition
    patient_start_offset = 0  # start from the first patient
    num_patients = 10  # up to N patients
    output_root = Path("./mimic-cxr-download")
    max_workers = 50  # concurrency limit

    # Fetch patient list via wget
    all_patients = list_patients(base_url, partition, username, password)
    selected = all_patients[patient_start_offset : patient_start_offset + num_patients]

    # Setup threading and progress bar
    semaphore = threading.Semaphore(max_workers)
    pbar = tqdm(total=len(selected), desc="Downloading patients", unit="patient")
    threads = []

    def worker(patient):
        try:
            download_patient(
                base_url, partition, patient, username, password, output_root
            )
        except Exception as e:
            print(f"Error downloading {patient}: {e}")
        finally:
            pbar.update(1)
            semaphore.release()

    # Launch threads
    for patient in selected:
        semaphore.acquire()
        t = threading.Thread(target=worker, args=(patient,))
        t.start()
        threads.append(t)

    # Wait for all to finish
    for t in threads:
        t.join()
    pbar.close()

    print(f"Downloaded {len(selected)} patient directories to {output_root}")
