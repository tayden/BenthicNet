#!/usr/bin/env python

"""
Downloading BenthicNet images from CSV file.
"""

import asyncio
import datetime
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np
import pandas as pd
import PIL.Image
import tqdm

import benthicnet.io
from benthicnet import __version__

# Rate limiting defaults
DEFAULT_MIN_INTERVAL = 0.1  # 100ms between requests to same domain
PANGAEA_MIN_INTERVAL = 0.167  # pangaea.de: max 180 requests per 30 seconds
PANGAEA_COOLDOWN = 30
MAX_RETRY_ATTEMPTS = 5
RETRYABLE_STATUS_CODES = {429, 500, 503}
DEFAULT_CONCURRENT_DOWNLOADS = 8


class DomainRateLimiter:
    """
    Rate limiter that enforces per-domain request intervals.

    Tracks the last request time for each domain and enforces minimum
    intervals between requests. Also tracks domains that have returned
    Retry-After headers and blocks requests until that time passes.
    """

    def __init__(self):
        self._last_request: dict[str, float] = defaultdict(float)
        self._retry_until: dict[str, float] = defaultdict(float)
        self._lock = asyncio.Lock()

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc.lower()

    def _get_min_interval(self, domain: str) -> float:
        """Get minimum interval for a domain."""
        if "pangaea.de" in domain:
            return PANGAEA_MIN_INTERVAL
        return DEFAULT_MIN_INTERVAL

    async def acquire(self, url: str) -> None:
        """
        Wait until it's safe to make a request to the given URL's domain.

        Parameters
        ----------
        url : str
            The URL to be requested.
        """
        domain = self._get_domain(url)
        min_interval = self._get_min_interval(domain)

        async with self._lock:
            now = time.time()

            # Check if we're in a Retry-After cooldown period
            retry_until = self._retry_until[domain]
            if now < retry_until:
                wait_time = retry_until - now
                await asyncio.sleep(wait_time)
                now = time.time()

            # Check minimum interval since last request
            last_request = self._last_request[domain]
            elapsed = now - last_request
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)

            self._last_request[domain] = time.time()

    def set_retry_after(self, url: str, seconds: float) -> None:
        """
        Set a Retry-After cooldown for a domain.

        Parameters
        ----------
        url : str
            The URL that returned the Retry-After header.
        seconds : float
            Number of seconds to wait before retrying.
        """
        domain = self._get_domain(url)
        self._retry_until[domain] = time.time() + seconds


def _calculate_wait_time(response, attempt, url):
    """
    Calculate how long to wait before retrying a request.

    Parameters
    ----------
    response : httpx.Response
        The HTTP response object.
    attempt : int
        Current retry attempt number (0-indexed).
    url : str
        The URL being requested.

    Returns
    -------
    float
        Number of seconds to wait before retrying.
    """
    retry_after = response.headers.get("Retry-After", "")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            return 30.0

    if response.status_code == 429:
        return PANGAEA_COOLDOWN

    if response.status_code == 503 and "pangaea.de/" in url:
        return PANGAEA_COOLDOWN

    # Exponential backoff for other server errors
    return float(2**attempt)


async def _download_with_retry(client, url, rate_limiter, verbose=1, innerpad=""):
    """
    Download a URL with retry logic for rate limiting and server errors.

    Parameters
    ----------
    client : httpx.AsyncClient
        The async HTTP client to use.
    url : str
        The URL to download.
    rate_limiter : DomainRateLimiter
        Rate limiter for per-domain throttling.
    verbose : int, optional
        Verbosity level. Default is ``1``.
    innerpad : str, optional
        Padding for log messages. Default is ``""``.

    Returns
    -------
    httpx.Response or None
        The Response object, or None if request failed due to an exception.
    """
    response = None

    for attempt in range(MAX_RETRY_ATTEMPTS):
        await rate_limiter.acquire(url)

        try:
            response = await client.get(url)
        except httpx.RequestError as err:
            print(f"Error while handling: {url}")
            print(err)
            return None

        if response.status_code not in RETRYABLE_STATUS_CODES:
            return response

        wait_time = _calculate_wait_time(response, attempt, url)

        # Register the retry delay with the rate limiter
        rate_limiter.set_retry_after(url, wait_time)

        retry_after = response.headers.get("Retry-After", "")
        if verbose >= 1 and retry_after:
            print(f"{innerpad}Server asks us to retry after: {retry_after}")

        if verbose >= 1:
            print(
                f"{innerpad}Retrying in {wait_time} seconds "
                f"(HTTP Status {response.status_code}): {url}"
            )
        await asyncio.sleep(wait_time)

    return response


async def _save_image_to_temp(response, url, temp_dir, verbose=1, innerpad=""):
    """
    Save response content to a temporary file.

    Parameters
    ----------
    response : httpx.Response
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

    content = response.content
    await asyncio.to_thread(_write_file, temp_path, content)

    if verbose >= 4:
        print(f"{innerpad}  Wrote to {temp_path}")

    return temp_path


def _write_file(path, content):
    """Write content to a file (sync helper for asyncio.to_thread)."""
    with open(path, "wb") as f:
        f.write(content)


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


async def _download_single_image(
    client,
    rate_limiter,
    row,
    index,
    i_row,
    output_dir,
    temp_dir,
    skip_existing,
    check_image,
    verbose,
    innerpad,
):
    """
    Download a single image asynchronously.

    Returns
    -------
    tuple
        (i_row, index, status, destination_name) where status is one of:
        'already_exists', 'downloaded', 'error', 'skipped_url'
    """
    url = row["url"]

    # Handle missing URLs
    if pd.isna(url) or url == "":
        if verbose >= 2:
            print(f"{innerpad}Missing URL for entry\n{row}", flush=True)
        return (i_row, index, "error", None)

    destination = Path(output_dir) / row["dataset"] / row["site"] / row["image"]
    destination.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and destination.is_file():
        if verbose >= 3:
            print(
                f"{innerpad}Skipping download of {url}\n"
                f"{innerpad}Destination exists: {destination}",
                flush=True,
            )
        return (i_row, index, "already_exists", destination.name)

    if verbose >= 2:
        print(f"{innerpad}Downloading {url} to {destination}", flush=True)

    response = await _download_with_retry(
        client, url, rate_limiter, verbose=verbose, innerpad=innerpad
    )

    if response is None:
        return (i_row, index, "error", None)

    if response.status_code != 200:
        if verbose >= 1:
            print(f"{innerpad}Bad URL (HTTP Status {response.status_code}): " f"{url}")
        return (i_row, index, "error", None)

    # Save to a unique temp file to avoid conflicts
    temp_subdir = os.path.join(temp_dir, f"img_{i_row}")
    os.makedirs(temp_subdir, exist_ok=True)

    temp_path = await _save_image_to_temp(
        response, url, temp_subdir, verbose=verbose, innerpad=innerpad
    )

    if check_image and not _validate_image(temp_path, url):
        shutil.rmtree(temp_subdir, ignore_errors=True)
        return (i_row, index, "error", None)

    if verbose >= 4:
        print(
            f"{innerpad}  Moving {temp_path} to {destination}",
            flush=True,
        )
    shutil.move(temp_path, str(destination))
    shutil.rmtree(temp_subdir, ignore_errors=True)

    return (i_row, index, "downloaded", destination.name)


async def _download_images_async(
    df,
    output_dir,
    skip_existing=True,
    check_image=True,
    verbose=1,
    use_tqdm=True,
    print_indent=0,
    max_concurrent=DEFAULT_CONCURRENT_DOWNLOADS,
):
    """
    Async implementation of image downloading with concurrency control.
    """
    t_start = time.time()

    padding = " " * print_indent
    innerpad = padding + "    "

    if verbose >= 1:
        jobs_msg = f" with {max_concurrent} concurrent downloads"
        print(f"{padding}Downloading {len(df)} images{jobs_msg}", flush=True)

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

    is_valid = np.zeros(len(df), dtype=bool)
    image_names = [""] * len(df)

    rate_limiter = DomainRateLimiter()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_download(client, row, index, i_row, temp_dir):
        async with semaphore:
            return await _download_single_image(
                client,
                rate_limiter,
                row,
                index,
                i_row,
                output_dir,
                temp_dir,
                skip_existing,
                check_image,
                verbose,
                innerpad,
            )

    with tempfile.TemporaryDirectory() as temp_dir:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=max_concurrent * 2),
        ) as client:
            # Create all download tasks
            tasks = [
                bounded_download(client, row, index, i_row, temp_dir)
                for i_row, (index, row) in enumerate(df.iterrows())
            ]

            # Process with progress bar
            results = []
            for coro in tqdm.tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                disable=not use_tqdm,
                desc="Downloading",
            ):
                result = await coro
                results.append(result)

    # Process results
    for i_row, index, status, dest_name in results:
        if status == "already_exists":
            n_already_downloaded += 1
            is_valid[i_row] = True
            image_names[i_row] = dest_name
        elif status == "downloaded":
            n_download += 1
            is_valid[i_row] = True
            image_names[i_row] = dest_name
        else:  # error or skipped_url
            n_error += 1

    # Update dataframe with final image names
    for i_row, (index, _) in enumerate(df.iterrows()):
        if is_valid[i_row]:
            df.at[index, "image"] = image_names[i_row]

    if verbose >= 1:
        elapsed = datetime.timedelta(seconds=int(time.time() - t_start))
        print(
            f"{padding}Finished processing {len(df)} images in {elapsed}.", flush=True
        )

        summary = _format_summary(n_already_downloaded, n_error, n_download, len(df))
        if summary:
            print(padding + summary, flush=True)

    return df.loc[is_valid]


def download_images_from_dataframe(
    df,
    output_dir,
    skip_existing=True,
    check_image=True,
    inplace=True,
    verbose=1,
    use_tqdm=True,
    print_indent=0,
    max_concurrent=DEFAULT_CONCURRENT_DOWNLOADS,
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
    max_concurrent : int, optional
        Maximum number of concurrent downloads. Default is ``8``.

    Returns
    -------
    pandas.DataFrame
        Like `df`, but with the ``image`` column changed to the exact basename of
        the output file within the tarball, including extension. Only entries
        which could be downloaded are included; URLs which could not be found
        are omitted.
    """
    if not inplace:
        df = df.copy()

    return asyncio.run(
        _download_images_async(
            df,
            output_dir,
            skip_existing=skip_existing,
            check_image=check_image,
            verbose=verbose,
            use_tqdm=use_tqdm,
            print_indent=print_indent,
            max_concurrent=max_concurrent,
        )
    )


def download_images_from_csv(
    input_csv,
    output_dir,
    *args,
    output_csv=None,
    skip_existing=True,
    verbose=1,
    max_concurrent=DEFAULT_CONCURRENT_DOWNLOADS,
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
    max_concurrent : int, optional
        Maximum number of concurrent downloads. Default is ``8``.
    **kwargs : optional
        Additional arguments as per :func:`download_images_from_dataframe``.

    Returns
    -------
    None
    """
    t_start = time.time()

    if verbose >= 1:
        print(f"Will download all images listed in {input_csv}")
        print(f"To output directory {output_dir}")
        print(f"Using {max_concurrent} concurrent downloads")

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

    df = benthicnet.io.read_csv(input_csv)

    if verbose >= 1:
        print(f"Loaded CSV file in {time.time() - t_start:.1f} seconds", flush=True)

    output_df = download_images_from_dataframe(
        df,
        output_dir,
        *args,
        skip_existing=skip_existing,
        verbose=verbose,
        max_concurrent=max_concurrent,
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
        version=f"%(prog)s {__version__}",
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
        "-j",
        "--jobs",
        dest="max_concurrent",
        metavar="N",
        type=int,
        default=DEFAULT_CONCURRENT_DOWNLOADS,
        help="Number of concurrent downloads. Default is %(default)s.",
    )
    parser.add_argument(
        "--no-progress-bar",
        dest="use_tqdm",
        action="store_false",
        help="Disable tqdm progress bar.",
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
