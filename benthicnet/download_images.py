#!/usr/bin/env python

"""
Downloading BenthicNet images from CSV file.
"""

import datetime
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
import requests
import tqdm

import benthicnet.io
from benthicnet import __meta__

# Rate limiting for pangaea.de: max 180 requests per 30 seconds
PANGAEA_MIN_INTERVAL = 0.167
PANGAEA_COOLDOWN = 30
MAX_RETRY_ATTEMPTS = 5
RETRYABLE_STATUS_CODES = {429, 500, 503}


def _calculate_wait_time(response, attempt, url):
    """
    Calculate how long to wait before retrying a request.

    Parameters
    ----------
    response : requests.Response
        The HTTP response object.
    attempt : int
        Current retry attempt number (0-indexed).
    url : str
        The URL being requested.

    Returns
    -------
    int
        Number of seconds to wait before retrying.
    """
    retry_after = response.headers.get("Retry-After", "")
    if retry_after:
        try:
            return int(retry_after)
        except ValueError:
            return 30

    if response.status_code == 429:
        return PANGAEA_COOLDOWN

    if response.status_code == 503 and "pangaea.de/" in url:
        return PANGAEA_COOLDOWN

    # Exponential backoff for other server errors
    return 2**attempt


def _download_with_retry(session, url, verbose=1, innerpad=""):
    """
    Download a URL with retry logic for rate limiting and server errors.

    Parameters
    ----------
    session : requests.Session
        The requests session to use for connection pooling.
    url : str
        The URL to download.
    verbose : int, optional
        Verbosity level. Default is ``1``.
    innerpad : str, optional
        Padding for log messages. Default is ``""``.

    Returns
    -------
    requests.Response or None
        The Response object, or None if request failed due to an exception.
    """
    last_request_time = 0.0
    response = None

    for attempt in range(MAX_RETRY_ATTEMPTS):
        # Rate limiting for pangaea.de
        if "pangaea.de/" in url:
            elapsed = time.time() - last_request_time
            if elapsed < PANGAEA_MIN_INTERVAL:
                time.sleep(PANGAEA_MIN_INTERVAL - elapsed)

        try:
            response = session.get(url, stream=True)
            last_request_time = time.time()
        except requests.exceptions.RequestException as err:
            print(f"Error while handling: {url}")
            print(err)
            return None

        if response.status_code not in RETRYABLE_STATUS_CODES:
            return response

        wait_time = _calculate_wait_time(response, attempt, url)

        retry_after = response.headers.get("Retry-After", "")
        if verbose >= 1 and retry_after:
            print(f"{innerpad}Server asks us to retry after: {retry_after}")

        if verbose >= 1:
            print(
                f"{innerpad}Retrying in {wait_time} seconds "
                f"(HTTP Status {response.status_code}): {url}"
            )
        time.sleep(wait_time)

    return response


def _save_image_to_temp(response, url, temp_dir, verbose=1, innerpad=""):
    """
    Save response content to a temporary file.

    Parameters
    ----------
    response : requests.Response
        The HTTP response with image content.
    url : str
        The original URL (for filename extraction).
    temp_dir : str
        Path to temporary directory.
    verbose : int, optional
        Verbosity level. Default is ``1``.
    innerpad : str, optional
        Padding for log messages. Default is ``""``.

    Returns
    -------
    str
        Path to the temporary file.
    """
    if verbose >= 3:
        print(f"{innerpad}Downloading {url}")

    basename = os.path.basename(url.rstrip("/"))
    temp_path = os.path.join(temp_dir, basename)

    with open(temp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1048576):
            f.write(chunk)

    if verbose >= 4:
        print(f"{innerpad}  Wrote to {temp_path}")

    return temp_path


def _validate_image(file_path, url):
    """
    Validate that a file is a valid image using PIL.

    Parameters
    ----------
    file_path : str
        Path to the image file.
    url : str
        Original URL (for error messages).

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    try:
        PIL.Image.open(file_path)
        return True
    except KeyboardInterrupt:
        raise
    except Exception as err:
        print(f"Error while handling: {url}")
        print(err)
        return False


def _format_progress(processed, total, elapsed_time, downloads):
    """
    Format a progress message for the download process.

    Parameters
    ----------
    processed : int
        Number of URLs processed so far.
    total : int
        Total number of URLs to process.
    elapsed_time : float
        Time elapsed since start in seconds.
    downloads : int
        Number of successful downloads so far.

    Returns
    -------
    str
        Formatted progress message.
    """
    percent = 100 * processed / total
    remaining = total - processed

    if downloads > 0:
        time_remaining = elapsed_time / downloads * remaining
    else:
        time_remaining = elapsed_time / processed * remaining

    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))
    remaining_str = str(datetime.timedelta(seconds=int(time_remaining)))

    return (
        f"Processed {processed:4d}/{total} urls ({percent:6.2f}%) "
        f"in {elapsed_str} (approx. {remaining_str} remaining)"
    )


def _format_summary(n_already, n_errors, n_downloaded, total):
    """
    Format a summary message for the download results.

    Parameters
    ----------
    n_already : int
        Number of images already downloaded.
    n_errors : int
        Number of download errors.
    n_downloaded : int
        Number of newly downloaded images.
    total : int
        Total number of images attempted.

    Returns
    -------
    str
        Formatted summary message.
    """
    messages = []

    if n_already > 0:
        if n_already == total:
            prefix = "All"
        elif n_already == 1:
            prefix = "There was"
        else:
            prefix = "There were"
        suffix = "" if n_already == 1 else "s"
        messages.append(f"{prefix} {n_already} image{suffix} already downloaded.")

    if n_errors > 0:
        verb = "was" if n_errors == 1 else "were"
        suffix = "" if n_errors == 1 else "s"
        messages.append(f"There {verb} {n_errors} download error{suffix}.")

    if n_downloaded > 0:
        if n_downloaded == total:
            messages.append(f"All {n_downloaded} images were downloaded.")
        else:
            suffix = " was" if n_downloaded == 1 else "s were"
            messages.append(f"The remaining {n_downloaded} image{suffix} downloaded.")

    return " ".join(messages)


def download_images_from_dataframe(
    df,
    output_dir,
    skip_existing=True,
    check_image=True,
    inplace=True,
    verbose=1,
    use_tqdm=True,
    print_indent=0,
):
    """
    Download all images from a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset of images to download.
    output_dir : str
        Path to output directory.
    skip_existing : bool, optional
        Whether to skip downloading files for which the destination already
        exist. Default is ``True``.
    check_image : bool, default=True
        Whether to check the image can be opened with PIL. If ``True``,
        downloads which can not be opened are discarded.
    inplace : bool, optional
        Whether operations on ``df`` can be performed in place. Default is
        ``True``.
    verbose : int, optional
        Verbosity level. Default is ``1``.
    use_tqdm : bool, optional
        Whether to use tqdm to print progress. Disable if stdout is being
        written to a file. Default is ``True``.
    print_indent : int, optional
        Amount of whitespace padding to precede print statements.
        Default is ``0``.

    Returns
    -------
    pandas.DataFrame
        Like `df`, but with the ``image`` column changed to the exact basename of
        the output file within the tarball, including extension. Only entries
        which could be downloaded are included; URLs which could not be found
        are omitted.
    """
    t_start = time.time()

    padding = " " * print_indent
    innerpad = padding + "    "

    if verbose >= 1:
        print(f"{padding}Downloading {len(df)} images", flush=True)

    if not inplace:
        df = df.copy()

    if verbose >= 3:
        print(f"{padding}Sanitizing fields used to build filenames", flush=True)

    df["dataset"] = benthicnet.io.sanitize_filename_series(df["dataset"])
    df["site"] = benthicnet.io.sanitize_filename_series(df["site"])
    df["image"] = df.apply(benthicnet.io.row2basename, axis=1)
    df["url"] = df["url"].str.strip()

    if verbose != 1:
        use_tqdm = False

    n_already_downloaded = 0
    n_download = 0
    n_error = 0

    t_download_start = time.time()
    is_valid = np.zeros(len(df), dtype=bool)

    # Use a session for connection pooling
    session = requests.Session()

    try:
        for i_row, (index, row) in enumerate(
            tqdm.tqdm(df.iterrows(), total=len(df), disable=not use_tqdm)
        ):
            url = row["url"]

            # Handle missing URLs
            if pd.isna(url) or url == "":
                n_error += 1
                if verbose >= 2:
                    print(f"{innerpad}Missing URL for entry\n{row}", flush=True)
                continue

            destination = Path(output_dir) / row["dataset"] / row["site"] / row["image"]

            # Print progress periodically when not using tqdm
            should_print_progress = i_row > n_error and (
                verbose >= 3
                or (verbose >= 1 and not use_tqdm and (i_row <= 5 or i_row % 100 == 0))
            )
            if should_print_progress:
                elapsed = time.time() - t_download_start
                print(
                    padding + _format_progress(i_row, len(df), elapsed, n_download),
                    flush=True,
                )

            destination.parent.mkdir(parents=True, exist_ok=True)

            if skip_existing and destination.is_file():
                n_already_downloaded += 1
                if verbose >= 3:
                    print(
                        f"{innerpad}Skipping download of {url}\n"
                        f"{innerpad}Destination exists: {destination}",
                        flush=True,
                    )
            else:
                if verbose >= 2:
                    print(f"{innerpad}Downloading {url} to {destination}", flush=True)

                response = _download_with_retry(
                    session, url, verbose=verbose, innerpad=innerpad
                )

                if response is None:
                    n_error += 1
                    continue

                if response.status_code != 200:
                    if verbose >= 1:
                        print(
                            f"{innerpad}Bad URL (HTTP Status {response.status_code}): "
                            f"{url}"
                        )
                    n_error += 1
                    continue

                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = _save_image_to_temp(
                        response, url, temp_dir, verbose=verbose, innerpad=innerpad
                    )

                    if check_image and not _validate_image(temp_path, url):
                        n_error += 1
                        continue

                    if verbose >= 4:
                        print(
                            f"{innerpad}  Moving {temp_path} to {destination}",
                            flush=True,
                        )
                    shutil.move(temp_path, str(destination))
                    n_download += 1

            is_valid[i_row] = True
            df.at[index, "image"] = destination.name

    finally:
        session.close()

    if verbose >= 1:
        elapsed = datetime.timedelta(seconds=int(time.time() - t_start))
        print(
            f"{padding}Finished processing {len(df)} images in {elapsed}.", flush=True
        )

        summary = _format_summary(n_already_downloaded, n_error, n_download, len(df))
        if summary:
            print(padding + summary, flush=True)

    return df.loc[is_valid]


def download_images_from_csv(
    input_csv,
    output_dir,
    *args,
    output_csv=None,
    skip_existing=True,
    verbose=1,
    i_proc=None,
    n_proc=None,
    **kwargs,
):
    """
    Download all images from a CSV file.

    Parameters
    ----------
    input_csv : str
        Path to CSV file.
    output_dir : str
        Path to output directory.
    output_csv : str, optional
        Path to output CSV file, which will have rows containing invalid URLs
        dropped and columns sanitized to match the output file path.
        If omitted, no output CSV is generated.
    skip_existing : bool, optional
        Whether to skip downloading files for which the destination already
        exist. Default is ``True``.
    verbose : int, optional
        Verbosity level. Default is ``1``.
    i_proc : int or None, optional
        Run on only a partition of the CSV file. If ``None`` (default), the
        entire dataset will be downloaded by this process. Otherwise, ``n_proc``
        must also be set.
    n_proc : int or None, optional
        Number of partitions being run. Default is ``None``.
    **kwargs : optional
        Additional arguments as per :func:`download_images_from_dataframe``.

    Returns
    -------
    None
    """
    t_start = time.time()

    if (i_proc is not None) != (n_proc is not None):
        raise ValueError(
            "Both i_proc and n_proc must be defined when partitioning the CSV file."
        )

    skiprows = []
    start_idx = 0
    end_idx = 0
    part_str = ""

    if n_proc is not None and i_proc is not None:
        part_str = f"(part {i_proc} of {n_proc})"
        n_lines = benthicnet.io.count_lines(input_csv) - 1
        partition_size = n_lines / n_proc
        proc_idx = 0 if i_proc == n_proc else i_proc
        start_idx = round(proc_idx * partition_size)
        end_idx = round((proc_idx + 1) * partition_size)
        skiprows = list(range(1, 1 + start_idx)) + list(range(1 + end_idx, 1 + n_lines))

    if verbose >= 1:
        count_str = "all" if n_proc is None else str(end_idx - start_idx)
        part_info = "" if n_proc is None else f"{part_str} "
        print(f"Will download {count_str} images {part_info}listed in {input_csv}")
        print(f"To output directory {output_dir}")

        if skip_existing:
            print("Existing outputs will be skipped.")
        else:
            print("Existing outputs will be overwritten.")

        if output_csv:
            print(
                f"An output CSV file containing only the valid URLs will be"
                f" created at {output_csv}"
            )
            if os.path.isfile(output_csv):
                print(f"The existing file {output_csv} will be overwritten.")

        print(f"Reading CSV file ({benthicnet.io.file_size(input_csv)})...", flush=True)

    df = benthicnet.io.read_csv(input_csv, skiprows=skiprows)

    if verbose >= 1:
        print(f"Loaded CSV file in {time.time() - t_start:.1f} seconds", flush=True)

    output_df = download_images_from_dataframe(
        df,
        output_dir,
        *args,
        skip_existing=skip_existing,
        verbose=verbose,
        **kwargs,
    )

    if output_csv is not None:
        if verbose >= 1:
            print(f"Saving valid CSV rows to {output_csv}")
        output_df.to_csv(output_csv, index=False)

    if verbose >= 1:
        elapsed = datetime.timedelta(seconds=int(time.time() - t_start))
        print(f"Total runtime: {elapsed}", flush=True)


def get_parser():
    """
    Build CLI parser for downloading BenthicNet image dataset.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser.
    """
    import argparse
    import textwrap

    prog = os.path.split(sys.argv[0])[1]
    if prog in ("__main__.py", "__main__"):
        prog = os.path.split(__file__)[1]

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Download all images listed in a BenthicNet CSV file",
        add_help=False,
    )

    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"%(prog)s {__meta__.version}",
        help="Show program's version number and exit.",
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Input CSV file, in the BenthicNet format.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Root directory for downloaded images.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Output CSV file.",
    )
    parser.add_argument(
        "--no-progress-bar",
        dest="use_tqdm",
        action="store_false",
        help="Disable tqdm progress bar.",
    )
    parser.add_argument(
        "--nproc",
        dest="n_proc",
        metavar="NPROC",
        type=int,
        help="Number of processing partitions being run.",
    )
    parser.add_argument(
        "--iproc",
        "--proc",
        dest="i_proc",
        metavar="IPROC",
        type=int,
        help="Partition index for this process.",
    )
    parser.add_argument(
        "--clobber",
        dest="skip_existing",
        action="store_false",
        help="Overwrite existing outputs instead of skipping their download.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help=textwrap.dedent(
            """
            Increase the level of verbosity of the program. This can be
            specified multiple times, each will increase the amount of detail
            printed to the terminal. The default verbosity level is %(default)s.
        """
        ),
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="count",
        default=0,
        help=textwrap.dedent(
            """
            Decrease the level of verbosity of the program. This can be
            specified multiple times, each will reduce the amount of detail
            printed to the terminal.
        """
        ),
    )
    return parser


def main():
    """
    Run command line interface for downloading images.
    """
    parser = get_parser()
    kwargs = vars(parser.parse_args())
    kwargs["verbose"] -= kwargs.pop("quiet", 0)
    return download_images_from_csv(**kwargs)


if __name__ == "__main__":
    main()
