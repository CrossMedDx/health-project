import os
import threading
import subprocess
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup
from dotenv import load_dotenv

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


def download_by_partition(
    base_url, partition, patient_start_offset, num_patients, output_root, max_workers
):
    """
    Downloads a specific number of patients from a partition.
    """
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


def download_by_study_ids(base_url, csv_file, output_root, max_workers):
    """
    Downloads specific study IDs based on a CSV file.
    """
    df = pd.read_csv(csv_file)
    semaphore = threading.Semaphore(max_workers)
    pbar = tqdm(total=len(df), desc="Downloading studies", unit="study")
    threads = []

    def worker(subject_id, study_id):
        try:
            # Download the s text file (report)
            report_url = (
                f"{base_url}/files/p{str(subject_id)[:2]}/p{subject_id}/s{study_id}.txt"
            )
            report_dir = output_root / "textData"
            report_dir.mkdir(parents=True, exist_ok=True)
            study_output_dir = report_dir / f"s{str(study_id)}"
            study_output_dir.mkdir(parents=True, exist_ok=True)
            report_output_path = study_output_dir /"report.txt"
            cmd_report = [
                "wget",
                "-N",
                "-c",
                f"--user={username}",
                f"--password={password}",
                report_url,
                "-O",
                str(report_output_path),
            ]
            result_report = subprocess.run(
                cmd_report,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result_report.returncode != 0:
                raise RuntimeError(
                    f"Failed to download {report_url}: {result_report.stderr.decode(errors='ignore')}."
                )

            # Download the images folder
            images_url = (
                f"{base_url}/files/p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/"
            )
            images_dir = output_root / "imageData"
            images_dir.mkdir(parents=True, exist_ok=True)
            study_output_dir = images_dir / f"s{str(study_id)}"
            study_output_dir.mkdir(parents=True, exist_ok=True)
            images_output_dir = study_output_dir /"report.txt"
            cmd_images = [
                "wget",
                "-r",
                "-N",
                "-c",
                "-np",
                "-nH",
                "--cut-dirs=7",
                f"--user={username}",
                f"--password={password}",
                images_url,
                "-P",
                str(images_output_dir),
            ]
            result_images = subprocess.run(
                cmd_images,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result_images.returncode != 0:
                raise RuntimeError(
                    f"Failed to download {images_url}: {result_images.stderr.decode(errors='ignore')}."
                )
        except Exception as e:
            print(f"Error downloading study {study_id} for subject {subject_id}: {e}")
        finally:
            pbar.update(1)
            semaphore.release()

    for _, row in df.iterrows():
        subject_id = row["subject_id"]
        study_id = row["study_id"]
        semaphore.acquire()
        t = threading.Thread(target=worker, args=(subject_id, study_id))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    pbar.close()

    print(f"Downloaded {len(df)} studies to {output_root}")


if __name__ == "__main__":
    # Configuration
    base_url = "https://physionet.org/files/mimic-cxr/2.1.0"
    output_root = Path("./mimic-cxr-download")
    max_workers = 50  # concurrency limit

    while True:
        try:
            print("\nSelect mode:")
            print("1. Download by partition")
            print("2. Download by study IDs")
            print("3. Exit")
            mode = int(input("Enter your choice (1/2/3): ").strip())

            if mode == 1:
                partition = input("Enter partition (e.g., p10): ").strip()
                patient_start_offset = int(
                    input("Enter patient start offset: ").strip()
                )
                num_patients = int(
                    input("Enter number of patients to download: ").strip()
                )
                download_by_partition(
                    base_url,
                    partition,
                    patient_start_offset,
                    num_patients,
                    output_root,
                    max_workers,
                )
                break
            elif mode == 2:
                csv_file = input("Enter path to CSV file with study IDs: ").strip()
                download_by_study_ids(base_url, csv_file, output_root, max_workers)
                break
            elif mode == 3:
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number (1, 2, or 3).")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
