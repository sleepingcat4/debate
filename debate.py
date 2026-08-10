import os
import json
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, hf_hub_download
import nemo.collections.asr as nemo_asr
import librosa
import soundfile as sf


SOURCE_REPO = "teamsleeping/sleeping-debate"
TRANSCRIBE_REPO = "teamsleeping/sleeping-debate-transcribe"

OUTPUT_FILE = "transcribe.jsonl"
FAILED_FILE = "failed.jsonl"

BATCH_SIZE = 20
DOWNLOAD_WORKERS = 20


def load_model():
    print("\n" + "=" * 80)
    print("LOADING NVIDIA PARAKEET TDT 0.6B V3")
    print("=" * 80)

    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v3"
    )

    print("Updating self-attention model of Fast-Conformer encoder")
    print("Attention context: left=256, right=256")

    asr_model.change_attention_model(
        self_attention_model="rel_pos_local_attn",
        att_context_size=[256, 256]
    )

    asr_model.eval()

    print("Model loaded successfully")
    print("=" * 80)

    return asr_model


def load_completed_files():
    completed = set()

    if not os.path.exists(OUTPUT_FILE):
        return completed

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

                filename = record.get("filename")

                if filename:
                    completed.add(filename)

            except json.JSONDecodeError:
                continue

    return completed


def get_wav_files():
    print("\nGetting WAV file list from Hugging Face...")

    api = HfApi()

    files = api.list_repo_files(
        repo_id=SOURCE_REPO,
        repo_type="dataset"
    )

    wav_files = [
        filename
        for filename in files
        if filename.lower().endswith(".wav")
    ]

    wav_files.sort()

    print(
        f"Found {len(wav_files):,} WAV files"
    )

    return wav_files


def download_file(filename):
    local_filename = os.path.basename(filename)

    local_path = os.path.join(
        os.getcwd(),
        local_filename
    )

    try:
        if os.path.exists(local_path):
            return {
                "filename": filename,
                "local_path": local_path,
                "success": True,
                "error": None
            }

        downloaded_path = hf_hub_download(
            repo_id=SOURCE_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=os.getcwd()
        )

        return {
            "filename": filename,
            "local_path": downloaded_path,
            "success": True,
            "error": None
        }

    except Exception as e:
        return {
            "filename": filename,
            "local_path": None,
            "success": False,
            "error": str(e)
        }


def download_batch(batch):
    print("\n" + "-" * 80)
    print(f"DOWNLOADING {len(batch)} FILES")
    print("-" * 80)

    downloaded = []
    failed = []

    with ThreadPoolExecutor(
        max_workers=DOWNLOAD_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                download_file,
                filename
            ): filename
            for filename in batch
        }

        for future in as_completed(futures):

            result = future.result()

            if result["success"]:

                downloaded.append(
                    (
                        result["filename"],
                        result["local_path"]
                    )
                )

                print(
                    f"[DOWNLOADED] "
                    f"{result['filename']}"
                )

            else:

                failed.append(result)

                print(
                    f"[DOWNLOAD FAILED] "
                    f"{result['filename']}"
                )

                print(
                    f"Error: "
                    f"{result['error']}"
                )

    print(
        f"\nDownloaded: {len(downloaded)}"
    )

    print(
        f"Download failures: {len(failed)}"
    )

    return downloaded, failed


def convert_to_mono(path):
    audio, sample_rate = librosa.load(
        path,
        sr=None,
        mono=True
    )

    sf.write(
        path,
        audio,
        sample_rate,
        subtype="PCM_16"
    )


def write_jsonl(handle, record):
    handle.write(
        json.dumps(
            record,
            ensure_ascii=False
        ) + "\n"
    )

    handle.flush()
    os.fsync(handle.fileno())


def write_failure(
    failed_handle,
    filename,
    stage,
    error
):
    record = {
        "filename": filename,
        "stage": stage,
        "error": str(error),
        "timestamp": time.time()
    }

    write_jsonl(
        failed_handle,
        record
    )


def extract_word_timestamps(timestamp_data):
    words = []

    if not timestamp_data:
        return words

    if isinstance(timestamp_data, dict):

        timestamp_data = timestamp_data.get(
            "word",
            []
        )

    for item in timestamp_data:

        if not isinstance(item, dict):
            continue

        word = item.get("word")
        start = item.get("start")
        end = item.get("end")

        if word is None:
            continue

        words.append(
            {
                "word": word,
                "start": start,
                "end": end
            }
        )

    return words


def transcribe_one(
    asr_model,
    filename,
    local_path,
    output_handle,
    failed_handle
):
    start_time = time.time()

    try:

        print("\n" + "-" * 80)
        print(
            f"[TRANSCRIBING] {filename}"
        )
        print("-" * 80)

        convert_to_mono(local_path)

        output = asr_model.transcribe(
            [local_path],
            timestamps=True
        )

        result = output[0]

        word_timestamps = extract_word_timestamps(
            result.timestamp
        )

        record = {
            "filename": filename,
            "transcript": result.text,
            "word_timestamps": word_timestamps
        }

        write_jsonl(
            output_handle,
            record
        )

        elapsed = time.time() - start_time

        print(
            f"[DONE] {filename}"
        )

        print(
            f"Time: {elapsed:.2f}s"
        )

        print(
            f"Words: {len(word_timestamps)}"
        )

        print(
            f"Transcript: {result.text[:200]}"
        )

        return True

    except Exception as e:

        print(
            f"[TRANSCRIPTION FAILED] "
            f"{filename}"
        )

        print(
            f"Error: {e}"
        )

        write_failure(
            failed_handle,
            filename,
            "transcription",
            e
        )

        return False

    finally:

        if os.path.exists(local_path):

            try:

                os.remove(local_path)

                print(
                    f"[DELETED] {filename}"
                )

            except Exception as e:

                print(
                    f"[DELETE FAILED] "
                    f"{filename}: {e}"
                )


def upload_results():
    print("\n" + "=" * 80)
    print("UPLOADING TRANSCRIPTIONS TO HUGGING FACE")
    print("=" * 80)

    command = [
        "hf",
        "upload",
        TRANSCRIBE_REPO,
        OUTPUT_FILE,
        "--repo-type=dataset"
    ]

    print(
        "Running:",
        " ".join(command)
    )

    try:

        subprocess.run(
            command,
            check=True
        )

        print(
            "\n[HF UPLOAD COMPLETE]"
        )

        return True

    except subprocess.CalledProcessError as e:

        print(
            f"\n[HF UPLOAD FAILED]"
        )

        print(
            f"Exit code: {e.returncode}"
        )

        return False

    except Exception as e:

        print(
            f"\n[HF UPLOAD FAILED] {e}"
        )

        return False


def process_dataset():

    start_time = time.time()

    print("\n" + "=" * 80)
    print("SLEEPING DEBATE TRANSCRIPTION")
    print("=" * 80)

    print(
        f"Source repository     : {SOURCE_REPO}"
    )

    print(
        f"Transcript repository : {TRANSCRIBE_REPO}"
    )

    print(
        f"Download batch        : {BATCH_SIZE}"
    )

    print(
        f"Download workers      : {DOWNLOAD_WORKERS}"
    )

    print(
        f"Transcript file       : {OUTPUT_FILE}"
    )

    print(
        f"Failed file           : {FAILED_FILE}"
    )

    print("=" * 80)

    completed = load_completed_files()

    print(
        f"\nAlready completed: "
        f"{len(completed):,}"
    )

    wav_files = get_wav_files()

    pending = [
        filename
        for filename in wav_files
        if filename not in completed
    ]

    print(
        f"Pending: "
        f"{len(pending):,}"
    )

    print(
        f"Skipped: "
        f"{len(wav_files) - len(pending):,}"
    )

    if not pending:

        print(
            "\nNothing to process."
        )

        return

    asr_model = load_model()

    total_files = len(wav_files)

    completed_count = len(completed)
    failed_count = 0

    total_batches = (
        len(pending)
        + BATCH_SIZE
        - 1
    ) // BATCH_SIZE

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8"
    ) as output_handle, open(
        FAILED_FILE,
        "a",
        encoding="utf-8"
    ) as failed_handle:

        for batch_start in range(
            0,
            len(pending),
            BATCH_SIZE
        ):

            batch = pending[
                batch_start:
                batch_start + BATCH_SIZE
            ]

            batch_number = (
                batch_start
                // BATCH_SIZE
            ) + 1

            print("\n\n" + "=" * 80)

            print(
                f"BATCH "
                f"{batch_number}/{total_batches}"
            )

            print(
                f"Completed : "
                f"{completed_count:,}"
            )

            print(
                f"Pending   : "
                f"{len(pending) - batch_start:,}"
            )

            print(
                f"Failed    : "
                f"{failed_count:,}"
            )

            print("=" * 80)

            downloaded, download_failed = (
                download_batch(batch)
            )

            for failure in download_failed:

                write_failure(
                    failed_handle,
                    failure["filename"],
                    "download",
                    failure["error"]
                )

                failed_count += 1

            for filename, local_path in downloaded:

                success = transcribe_one(
                    asr_model,
                    filename,
                    local_path,
                    output_handle,
                    failed_handle
                )

                if success:
                    completed_count += 1
                else:
                    failed_count += 1

                elapsed = (
                    time.time()
                    - start_time
                )

                remaining = (
                    total_files
                    - completed_count
                    - failed_count
                )

                print("\n" + "-" * 80)

                print(
                    f"Completed : "
                    f"{completed_count:,}"
                )

                print(
                    f"Pending   : "
                    f"{max(0, remaining):,}"
                )

                print(
                    f"Failed    : "
                    f"{failed_count:,}"
                )

                print(
                    f"Runtime   : "
                    f"{elapsed / 3600:.2f} hours"
                )

                print("-" * 80)

            print("\n" + "=" * 80)
            print(
                f"BATCH {batch_number} COMPLETE"
            )
            print("=" * 80)

            upload_results()

    elapsed = (
        time.time()
        - start_time
    )

    print("\n" + "=" * 80)
    print("TRANSCRIPTION COMPLETE")
    print("=" * 80)

    print(
        f"Completed : "
        f"{completed_count:,}"
    )

    print(
        f"Failed    : "
        f"{failed_count:,}"
    )

    print(
        f"Runtime   : "
        f"{elapsed / 3600:.2f} hours"
    )

    print(
        f"Local transcript : "
        f"{os.path.abspath(OUTPUT_FILE)}"
    )

    print(
        f"Local failures   : "
        f"{os.path.abspath(FAILED_FILE)}"
    )

    print("=" * 80)


if __name__ == "__main__":
    process_dataset()
