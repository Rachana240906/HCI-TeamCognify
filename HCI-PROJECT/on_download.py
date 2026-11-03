def download(
    *,
    dataset: str,
    tag: str | None = None,
    target_dir: Path | str | None = None,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    verify_hash: bool | Literal["auto", "nemar"] = "auto",
    verify_size: bool = True,
    max_retries: int = 5,
    max_concurrent_downloads: int = 5,
    metadata_timeout: float = 15.0,
    source: Source | None = None,
) -> None:
    """Download datasets from OpenNeuro.

    Parameters
    ----------
    dataset
        The dataset to retrieve, for example `ds000248`.
    tag
        The tag (revision) of the dataset to retrieve. This always refers to an
        **OpenNeuro** revision, including when `source="nemar"`: NEMAR numbers
        its mirrors independently, so its own version numbers are never
        accepted here.

        > **Note:** NEMAR mirrors only a single snapshot per dataset, so
        > requesting any revision other than the one it currently holds will
        > fail. Use `source="openneuro"` to fetch an older revision.
    target_dir
        The directory in which to store the downloaded data. If `None`,
        create a subdirectory with the dataset name in the current working
        directory.
    include
        Files and directories to download. **Only** these files and directories
        will be retrieved. Uses glob-style matching: `*` matches any characters
        except `/`, `**` matches across directory boundaries, and `?`
        matches a single non-`/` character. Patterns without a `/` also
        match as directory prefixes (e.g., `'sub-01'` includes all files
        under `sub-01/`, and `'sub-0*'` includes all files under every
        matching directory). Use a leading `/` to restrict to the dataset
        root (e.g., `'/*.json'`). As an example, if you would like to
        download only subject '1' and run '01' files, you can do so via:
        `'sub-1/**/*run-01*'`.

        > **Note:** Consistent with `.gitignore` semantics, `*` and `**` do
        > **not** match dot-prefixed (hidden) filenames. To include such files,
        > use an explicit pattern like `'**/.*'`. The BIDS specification reserves
        > dotfiles for system use, so they are rarely needed.
    exclude
        Files and directories to exclude from downloading.
        Uses the same glob-style matching as `include`.

        > **Note:** Certain essential BIDS metadata files are always downloaded
        > regardless of `exclude` patterns: `dataset_description.json`,
        > `participants.tsv`, `participants.json`, `README`, `CHANGES`, and
        > `.bidsignore`. The dot-prefixed `.bidsignore` is downloaded even
        > though other dotfiles are skipped by default, because BIDS validators
        > rely on it to know which files to ignore.
    verify_hash
        Whether to calculate and verify a hash of each downloaded file.

        OpenNeuro publishes no checksums of its own, so verification falls back
        to the S3 `ETag`. That is an MD5 only for single-part uploads;
        multipart ones (in practice, files of 1 GiB or more) return a
        digest-of-digests instead and cannot be checked at all. An `ETag` also
        only ever describes what OpenNeuro has *stored*, so it cannot reveal a
        file that was already corrupt at rest. NEMAR mirrors OpenNeuro
        byte-for-byte and publishes a checksum for every file, which closes
        both gaps.

        - `"auto"` (the default) verifies against NEMAR's checksums when the
          files being downloaded include one at or above the multipart
          threshold, and against the `ETag` otherwise. NEMAR is contacted only
          when it can actually help, so most downloads are unaffected.
        - `True` verifies using only what the source itself announces, never
          contacting NEMAR.
        - `"nemar"` always consults NEMAR, verifying every file it covers
          regardless of size.
        - `False` disables hash checking.

        `"auto"` and `"nemar"` apply only to `source="openneuro"`; a NEMAR
        download already carries its own checksums. The single file NEMAR
        rewrites (`dataset_description.json`) is never cross-verified. If NEMAR
        is unreachable, or mirrors a different revision than the one being
        downloaded, both degrade to the `True` behaviour rather than failing.
    verify_size
        Whether to check if the downloaded file size matches what the server
        announced.
    max_retries
        Try the specified number of times to download a file before failing.
    max_concurrent_downloads
        The maximum number of downloads to run in parallel.
    metadata_timeout
        Timeout in seconds for metadata queries.
    source
        Where to fetch the data from.

        - `"openneuro"` (the default) uses OpenNeuro itself, and is the only
          source that can serve restricted datasets, arbitrary revisions, and
          non-MEEG modalities.
        - `"nemar"` uses the NEMAR mirror at UC San Diego
          (<https://nemar.org>), which carries the EEG, MEG, and iEEG datasets
          published on OpenNeuro. The files are byte-for-byte identical, and
          NEMAR publishes per-file checksums that are verified during the
          download.

        If `None`, the `OPENNEURO_SOURCE` environment variable is consulted,
        falling back to `"openneuro"`.

    """
    if max_concurrent_downloads < 1:
        raise ValueError("max_concurrent_downloads must be at least 1.")
    source = get_source(source)

    msg_problems = "problems 🤯" if unicode_ok else "problems"
    msg_bugs = "bugs 🪲" if unicode_ok else "bugs"
    msg_hello = "👋 Hello!" if unicode_ok else "Hello!"
    msg_great_to_see_you = "Great to see you!"
    if unicode_ok:
        msg_great_to_see_you += " 🤗"
    msg_please = "👉 Please" if unicode_ok else "   Please"

    msg = (
        f"\n{msg_hello} This is openneuro-py {__version__}. "
        f"{msg_great_to_see_you}\n\n"
        f"   {msg_please} report {msg_problems} and {msg_bugs} at\n"
        f"      https://github.com/openneuro-py/openneuro-py/issues\n"
    )
    cprint(msg, cli_only=True)
    where = f"{dataset} from NEMAR" if source == "nemar" else dataset
    cprint(_unicode(f"Preparing to download {where}", emoji="🌍"))

    if target_dir is None:
        target_dir = Path(dataset)
    else:
        target_dir = Path(target_dir)
    target_dir = target_dir.expanduser().resolve()

    include = [include] if isinstance(include, str) else include
    include = [] if include is None else list(include)

    exclude = [exclude] if isinstance(exclude, str) else exclude
    exclude = [] if exclude is None else list(exclude)

    retry_backoff = 0.5  # seconds
    metadata = _get_download_metadata(
        dataset_id=dataset,
        tag=tag,
        source=source,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        metadata_timeout=metadata_timeout,
    )
    del tag
    tag = metadata.id.replace(f"{dataset}:", "")
    if target_dir.exists():
        # Once we find the first child, we know the directory is not empty, so we can
        # stop iterating immediately.
        target_dir_empty = next(target_dir.iterdir(), None) is None

        if not target_dir_empty:
            local_tag = _get_local_tag(dataset_id=dataset, dataset_dir=target_dir)

            if local_tag is None and source == "nemar":
                # Expected, not a warning sign: NEMAR rewrites `DatasetDOI` to
                # its own identifier, which carries no OpenNeuro revision. The
                # per-file checksums still catch any stale or corrupt file.
                cprint(
                    "The existing files carry NEMAR's identifier rather than "
                    "an OpenNeuro revision, so the local revision cannot be "
                    "determined. Each file is verified individually regardless."
                )
            elif local_tag is None:
                cprint(
                    "Cannot determine local revision of the dataset, "
                    "and the target directory is not empty. If the "
                    "download fails, you may want to try again with a "
                    "fresh (empty) target directory."
                )
            elif local_tag != tag:
                raise FileExistsError(
                    f"You requested to download revision {tag}, but "
                    f"revision {local_tag} exists locally in the designated "
                    f"target directory. Please either remove this dataset or "
                    f"specify a different target directory, and try again."
                )

    essential_files = {
        "dataset_description.json",
        "participants.tsv",
        "participants.json",
        "README",
        # BIDS allows the README to carry an extension, and NEMAR stores it as
        # "README.md" even where OpenNeuro has a plain "README". Names that the
        # dataset does not contain are simply ignored below.
        "README.md",
        "CHANGES",
        ".bidsignore",
    }

    all_files = metadata.files
    del metadata

    filenames = [f.filename for f in all_files]

    if include:
        included = _glob.glob_filter(filenames, include)
        included_set = {f for matches in included.values() for f in matches}
    else:
        included_set = {f for f in filenames if not _glob.is_dotfile(f)}

    if exclude:
        excluded = _glob.glob_filter(filenames, exclude)
        excluded_set = {f for matches in excluded.values() for f in matches}
    else:
        excluded_set = set()

    keep = (included_set - excluded_set) | (essential_files & set(filenames))
    files: list[DatasetFile] = [f for f in all_files if f.filename in keep]

    if include:
        for pattern, matches in included.items():
            if not matches:
                has_glob = any(c in pattern for c in "*?[")
                maybe = [] if has_glob else get_close_matches(pattern, filenames)
                if maybe:
                    extra = (
                        "Perhaps you mean one of these paths:\n- "
                        + "\n- ".join(maybe)
                        + "\n"
                    )
                else:
                    extra = "There were no similar filenames found in the metadata. "
                raise RuntimeError(
                    f"Could not find path in the dataset:\n- {pattern}\n{extra}"
                    "Please check your includes."
                )

    # NEMAR's checksums only add anything to an *OpenNeuro* download; a NEMAR
    # download already carries its own. "auto" pays the extra request only when
    # the files actually being fetched include one too large for OpenNeuro's
    # `ETag` to describe.
    if source == "openneuro" and (
        verify_hash == "nemar"
        or (
            verify_hash == "auto"
            and any((f.size or 0) >= _MULTIPART_THRESHOLD for f in files)
        )
    ):
        files = _apply_nemar_checksums(
            files,
            dataset_id=dataset,
            tag=tag,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            metadata_timeout=metadata_timeout,
        )

    msg = (
        f"Checking {len(files)} files, downloading as needed "
        f"({max_concurrent_downloads} concurrent downloads)."
    )
    cprint(_unicode(msg, emoji="📥", end=""))
    if not console.is_jupyter:
        cprint("")  # Blank line before the progress bars (terminal only)

    query_str = snapshot_query_template.safe_substitute(
        tag=tag or "null",
        dataset_id=dataset,
    )
    stats = _DownloadStats()
    coroutine = _download_files(
        target_dir=target_dir,
        files=files,
        # "nemar" has done its work above, by attaching checksums to the files;
        # from here on it simply means "verify".
        verify_hash=bool(verify_hash),
        verify_size=verify_size,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        max_concurrent_downloads=max_concurrent_downloads,
        query_str=query_str,
        stats=stats,
        dataset_id=dataset,
        debug_hint=_make_nemar_debug_hint(dataset) if source == "nemar" else None,
    )

    # Block until the download actually finishes, even when a loop is already
    # running (e.g. in Jupyter), so failures surface and the summary below
    # reports the real stats rather than an empty tally (gh-329).
    failures = _run_coroutine_blocking(coroutine)

    if failures:
        _print_download_failures(failures)

    n_files = stats.n_files
    plural = "file" if n_files == 1 else "files"
    summary = (
        f"Finished downloading {dataset} "
        f"(downloaded {n_files} {plural} and {_format_size(stats.n_bytes)}).\n"
    )
    cprint(_unicode(summary, emoji="✅", end=""))
    cprint(_unicode("Please enjoy your brains.\n", emoji="🧠", end=""), cli_only=True)
