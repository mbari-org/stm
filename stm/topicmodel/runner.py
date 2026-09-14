# stm, Apache-2.0 license
# Filename: runner.py
# Description: Run the ROST topic model (Docker rost-cli or local binaries)
from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from perchtopic.features import FeatureBlock, TimeGrid, combine

DEFAULT_IMAGE = "rost-cli:latest"
MODEL_META_NAME = "model_meta.json"
DEFAULT_ALPHA = 0.01 # Lower = each document is more likely to be dominated by a small number of topics (i.e., sparse topic distribution).
DEFAULT_BETA = 0.001 # Lower = each topic is more likely to be dominated by a small number of words (i.e., sparse word distribution).
DEFAULT_GAMMA = 0.001  # Used if num_topics is None, to control the growth of topics.


@dataclass
class TopicModelResult:
    """Paths written by :class:`TopicModelRunner.run`."""

    docs_path: Path
    out_dir: Path
    topics_path: Path
    maxlikelihood_path: Path
    topicmodel_path: Path
    perplexity_path: Path
    hist_path: Path
    phi_path: Path | None = None
    theta_path: Path | None = None
    maxlikelihood_with_time_path: Path | None = None
    avg_perplexity: float | None = None
    vocab_size: int = 0
    num_topics: int | None = None
    document_seconds: float | None = None
    num_documents: int = 0


class TopicModelRunner:
    """Run ROST ``topics.refine.t`` on word documents from a feature block.

    Default is Docker image ``rost-cli:latest`` (build with perch2topic's
    ``DockerfileROST``). ``timeout`` is the wall-clock budget in seconds for
    one ROST process; ``None`` means no limit.

    Perch2 rows become words via *linear_model* (or explicit *labels*).
    If neither is given, embeddings are reduced with UMAP and clustered
    with HDBSCAN via :class:`~stm.cluster.Cluster`; PCEN frames use
    the same path. Passing both block types yields two tokens per time step
    (Perch2 word, then PCEN word with a disjoint id range).
    """

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        use_docker: bool = True,
        rost_path: Path | str | None = None,
        timeout: float | None = None,
        num_topics: int | None = None,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        gamma: float | None = DEFAULT_GAMMA,
        g_time: int = 1,
        cell_space: int = 0,
        threads: int = 64,
        online: bool = False,
        online_mint: int = 5,
        max_topic: bool = True,
        vocab_size: int | None = None,
        document_seconds: float | None = None,
    ) -> None:
        self.image = image
        self.use_docker = use_docker
        self.rost_path = Path(rost_path) if rost_path is not None else Path("./rost-cli/bin")
        self.timeout = timeout
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.g_time = g_time
        self.cell_space = cell_space
        self.threads = threads
        self.online = online
        self.online_mint = online_mint
        self.max_topic = max_topic
        self.vocab_size = vocab_size
        self.document_seconds = document_seconds

    def write_documents(
        self,
        labels: np.ndarray,
        grid: TimeGrid,
        path: Path,
        mask: np.ndarray | None = None,
        words_per_doc: int | None = None,
    ) -> tuple[Path, int]:
        """Write a ROST document CSV (``timestamp_ms,word[,word...]``).

        Negative ids (if any) become singleton word ids so every token is
        non-negative. *labels* may be one per grid cell or one per True mask
        entry. *words_per_doc* is the number of consecutive rows per
        document, defaulting to 1 (one document per grid cell).
        Returns ``(path, vocab_size)``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        labels_arr = np.asarray(labels)
        if labels_arr.ndim > 2:
            raise ValueError(f"labels must be 1-D or 2-D, got shape {labels_arr.shape}")
        if labels_arr.ndim == 2:
            words, starts = _align_label_matrix(labels_arr, grid, mask)
        else:
            words, starts = _align_labels(labels_arr, grid, mask)
            words = np.asarray(words).reshape(-1, 1)
        words = _singleton_noise(words.reshape(-1)).reshape(words.shape)
        if len(words) == 0:
            raise ValueError("write_documents() requires at least one label")
        step = 1 if words_per_doc is None else words_per_doc
        if step < 1:
            raise ValueError(f"words_per_doc must be >= 1, got {step}")

        rows: list[list[int]] = []
        for i in range(0, len(words), step):
            j = min(i + step, len(words))
            t_ms = int(round(float(starts[j - 1]) * 1000.0))
            rows.append([t_ms, *[int(w) for w in words[i:j].reshape(-1)]])
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)
        vocab_size = int(words.max()) + 1
        print(f"Wrote {len(rows)} documents to {path} (V={vocab_size})", flush=True)
        return path, vocab_size

    def run_from_block(
        self,
        block: FeatureBlock | Sequence[FeatureBlock],
        doc_dir: Path,
        model_dir: Path,
        *,
        labels: np.ndarray | None = None,
        linear_model: Path | str | object | None = None,
        target: TimeGrid | None = None,
        assign_noise: bool = True,
        random_state: int = 0,
        perch_method: str = "hdbscan",
        pcen_method: str = "kmeans",
        document_name: str = "all_docs",
    ) -> TopicModelResult:
        """Write documents from a Perch2 and/or PCEN block and run ROST.

        *block* may be one :class:`FeatureBlock` or several. Perch2 words come
        from *linear_model*, or from UMAP + HDBSCAN
        (:class:`~stm.cluster.Cluster`) when *linear_model* is omitted.
        PCEN frames are clustered with K-Means, K taken from the inertia-curve
        knee. *perch_method* / *pcen_method* select ``"hdbscan"`` (UMAP +
        HDBSCAN) or ``"kmeans"`` per stream. *assign_noise* pushes HDBSCAN
        noise onto the nearest cluster instead of leaving it as its own word;
        it has no effect on K-Means, which never emits noise. Passing both block types yields two tokens per time step (Perch2
        word, then PCEN word with a disjoint id range). Documents span
        ``document_seconds`` of audio (the analysis window length if unset),
        converted to a row count from the output grid's hop.
        """
        blocks = _as_blocks(block)
        if labels is not None and linear_model is not None:
            raise ValueError("pass labels= or linear_model=, not both")

        if labels is not None:
            fused = _fuse_blocks(blocks, target)
            word_ids = np.asarray(getattr(labels, "labels", labels)).reshape(-1)
            vocab_size = int(word_ids.max()) + 1 if len(word_ids) else 0
            grid, mask = fused.grid, fused.mask
        else:
            word_ids, vocab_size, grid, mask = _word_ids_from_blocks(
                blocks,
                linear_model,
                target=target,
                random_state=random_state,
                assign_noise=assign_noise,
                perch_method=perch_method,
                pcen_method=pcen_method,
            )

        doc_words = _rows_per_doc(grid, self.document_seconds)
        hop = _grid_hop_seconds(grid)
        print(
            f"words_per_doc={doc_words} "
            f"({doc_words * hop:g}s per document at {hop:g}s hop)",
            flush=True,
        )
        docs_path = Path(doc_dir) / f"{document_name}.csv"
        docs_path, written_v = self.write_documents(
            word_ids, grid, docs_path, mask=mask, words_per_doc=doc_words
        )
        return self.run(docs_path, model_dir, vocab_size=max(vocab_size, written_v))

    def run(
        self,
        docs_path: Path,
        out_dir: Path,
        vocab_size: int,
    ) -> TopicModelResult:
        """Run ``topics.refine.t`` then ``words.bincount`` on *docs_path*."""
        docs_path = Path(docs_path).resolve()
        out_dir = Path(out_dir).resolve()
        if not docs_path.is_file():
            raise FileNotFoundError(docs_path)
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        out_dir.mkdir(parents=True, exist_ok=True)

        topics_path = out_dir / "topics.csv"
        maxlikelihood_path = out_dir / "topics.maxlikelihood.csv"
        topicmodel_path = out_dir / "topicmodel.csv"
        perplexity_path = out_dir / "perplexity.csv"
        hist_path = out_dir / "topics.hist.csv"
        log_path = out_dir / "topics.log"

        refine = self._refine_cmd(
            docs_path,
            topics_path,
            maxlikelihood_path,
            topicmodel_path,
            perplexity_path,
            log_path,
            vocab_size,
        )
        mount = _mount_root(docs_path, out_dir)
        print(" ".join(refine), flush=True)
        self._execute(refine, mount, log_path=log_path)

        num_topics = self.num_topics
        if num_topics is None and topics_path.exists():
            topic_df = _read_ragged_csv(topics_path)
            num_topics = int(topic_df.drop(columns=[0]).max().max() + 1)

        if num_topics is not None and maxlikelihood_path.exists():
            bincount = [
                "words.bincount",
                "-i",
                str(maxlikelihood_path),
                "-o",
                str(hist_path),
                "-V",
                str(num_topics),
            ]
            print(" ".join(bincount), flush=True)
            self._execute(bincount, mount)

        phi_path = None
        theta_path = None
        if topicmodel_path.exists() and hist_path.exists() and num_topics is not None:
            phi_path, theta_path = _write_phi_theta(
                topicmodel_path,
                hist_path,
                out_dir,
                num_topics=num_topics,
                vocab_size=vocab_size,
                alpha=self.alpha,
                beta=self.beta,
            )

        time_path = None
        if maxlikelihood_path.exists():
            hop = self.document_seconds
            if hop is None:
                hop = _infer_document_seconds(docs_path)
            time_path = _write_maxlikelihood_with_time(
                maxlikelihood_path,
                out_dir / "topics.maxlikelihood_with_time.csv",
                document_seconds=hop,
                collapse=self.max_topic,
            )

        avg_ppx = None
        if perplexity_path.exists():
            ppx = pd.read_csv(perplexity_path, header=None)
            if len(ppx) and ppx.shape[1] > 1:
                avg_ppx = float(ppx.iloc[:, 1].sum() / len(ppx))
                print(f"Avg Per Doc Perplexity = {avg_ppx}", flush=True)

        document_seconds = self.document_seconds
        if document_seconds is None:
            document_seconds = _infer_document_seconds(docs_path)
        num_documents = _count_documents(docs_path)
        _write_model_meta(
            out_dir / MODEL_META_NAME,
            document_seconds=document_seconds,
            num_documents=num_documents,
            vocab_size=vocab_size,
            num_topics=num_topics,
            docs_path=docs_path,
        )

        return TopicModelResult(
            docs_path=docs_path,
            out_dir=out_dir,
            topics_path=topics_path,
            maxlikelihood_path=maxlikelihood_path,
            topicmodel_path=topicmodel_path,
            perplexity_path=perplexity_path,
            hist_path=hist_path,
            phi_path=phi_path,
            theta_path=theta_path,
            maxlikelihood_with_time_path=time_path,
            avg_perplexity=avg_ppx,
            vocab_size=vocab_size,
            num_topics=num_topics,
            document_seconds=document_seconds,
            num_documents=num_documents,
        )

    def _refine_cmd(
        self,
        docs_path: Path,
        topics_path: Path,
        maxlikelihood_path: Path,
        topicmodel_path: Path,
        perplexity_path: Path,
        log_path: Path,
        vocab_size: int,
    ) -> list[str]:
        cmd = [
            "topics.refine.t",
            "-i",
            str(docs_path),
            f"--out.topics={topics_path}",
            f"--out.topics.ml={maxlikelihood_path}",
            f"--out.topicmodel={topicmodel_path}",
            f"--ppx.out={perplexity_path}",
            f"--logfile={log_path}",
            "-V",
            str(vocab_size),
            f"--alpha={self.alpha}",
            f"--beta={self.beta}",
            "--threads",
            str(self.threads),
            f"--g.time={self.g_time}",
            f"--cell.space={self.cell_space}",
        ]
        if self.num_topics is not None:
            cmd.extend(["-K", str(self.num_topics)])
        else:
            gamma = DEFAULT_GAMMA if self.gamma is None else self.gamma
            cmd.extend(["--grow.topics.size=true", f"--gamma={gamma}"])
        if self.online:
            cmd.extend(
                [
                    "--online",
                    f"--out.topics.online={topics_path.with_name('topics.online.csv')}",
                    f"--out.ppx.online={perplexity_path.with_name('perplexity.online.csv')}",
                    "--online.mint",
                    str(self.online_mint),
                ]
            )
        return cmd

    def _execute(
        self,
        cmd: list[str],
        mount: Path,
        log_path: Path | None = None,
    ) -> str:
        mount = Path(mount).resolve()
        if self.use_docker:
            argv = [
                "docker",
                "run",
                "--rm",
                "-u",
                f"{os.getuid()}:{os.getgid()}",
                "-v",
                f"{mount}:{mount}",
                self.image,
                "stdbuf",
                "-oL",
                "-eL",
                *cmd,
            ]
        else:
            argv = [str(self.rost_path / cmd[0]), *cmd[1:]]
        print(" ".join(argv), flush=True)
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        chunks: list[str] = []
        stop_tail = threading.Event()
        tail = None
        if log_path is not None:
            tail = threading.Thread(
                target=_tail_file,
                args=(Path(log_path), stop_tail, chunks),
                daemon=True,
            )
            tail.start()

        def _read_stdout() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                chunks.append(line)
                print(line, end="", flush=True)

        reader = threading.Thread(target=_read_stdout, daemon=True)
        reader.start()
        try:
            proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.wait()
            raise TimeoutError(
                f"TopicModelRunner exceeded {self.timeout}s running {cmd[0]}"
            ) from exc
        finally:
            stop_tail.set()
            reader.join(timeout=2)
            if tail is not None:
                tail.join(timeout=2)
        output = "".join(chunks)
        if proc.returncode != 0:
            detail = output.strip() or f"exit {proc.returncode}"
            raise RuntimeError(f"{cmd[0]} failed (exit {proc.returncode}): {detail}")
        return output


def _as_blocks(block: FeatureBlock | Sequence[FeatureBlock]) -> list[FeatureBlock]:
    if isinstance(block, FeatureBlock):
        return [block]
    blocks = list(block)
    if not blocks:
        raise ValueError("run_from_block requires at least one FeatureBlock")
    if not all(isinstance(item, FeatureBlock) for item in blocks):
        raise TypeError("block must be a FeatureBlock or a sequence of FeatureBlocks")
    return blocks


def _grid_hop_seconds(grid: TimeGrid) -> float:
    """Median spacing between consecutive interval starts, in seconds."""
    starts = np.asarray(grid.starts, dtype=np.float64)
    if len(starts) < 2:
        return 0.0
    diffs = np.diff(starts)
    positive = diffs[diffs > 0.0]
    return float(np.median(positive)) if len(positive) else 0.0


def _rows_per_doc(grid: TimeGrid, document_seconds: float | None) -> int:
    """Consecutive *grid* rows that make up one ROST document.

    *document_seconds* is how much audio a document should span. When it is
    ``None`` the analysis window length is used, so each document covers one
    window's worth of audio and consecutive documents do not overlap.

    This is a count of time steps, not a feature dimension: at a 0.5 s hop a
    5 s document is 10 rows regardless of how wide each feature vector is.
    Always returns at least 1.
    """
    hop = _grid_hop_seconds(grid)
    if document_seconds is None:
        durations = np.asarray(grid.durations, dtype=np.float64)
        document_seconds = float(np.median(durations)) if len(durations) else 0.0
    if hop <= 0.0 or document_seconds <= 0.0:
        return 1
    return max(1, int(round(float(document_seconds) / hop)))

def _is_pcen(block: FeatureBlock) -> bool:
    extractor = block.provenance.get("extractor")
    if extractor == "PcenExtractor":
        return True
    nested = block.provenance.get("blocks") or []
    return bool(nested) and all(
        isinstance(item, dict) and item.get("extractor") == "PcenExtractor" for item in nested
    )


def _fuse_blocks(blocks: Sequence[FeatureBlock], target: TimeGrid | None) -> FeatureBlock:
    if len(blocks) == 1:
        if target is None or blocks[0].grid == target:
            return blocks[0]
        return combine(blocks, target)
    same_grid = all(item.grid == blocks[0].grid for item in blocks)
    if same_grid and (target is None or target == blocks[0].grid):
        values = np.concatenate([item.values for item in blocks], axis=1)
        masks = [
            np.ones(len(item.grid), dtype=bool) if item.mask is None else np.asarray(item.mask, dtype=bool)
            for item in blocks
        ]
        return FeatureBlock(
            values=values,
            grid=blocks[0].grid,
            mask=np.logical_and.reduce(masks),
            provenance={
                "combine": "concat",
                "blocks": [item.provenance for item in blocks],
            },
        )
    return combine(blocks, blocks[0].grid if target is None else target)


def _masked_values(block: FeatureBlock) -> np.ndarray:
    if block.mask is None:
        return block.values
    return block.values[np.asarray(block.mask, dtype=bool)]


def _print_word_counts(word_ids: np.ndarray, names: Sequence[str] | None = None) -> None:
    unique, counts = np.unique(word_ids, return_counts=True)
    for word, count in zip(unique, counts):
        name = (
            names[int(word)]
            if names and 0 <= int(word) < len(names)
            else "?"
        )
        print(f"  {int(word)} {name}: {int(count)}", flush=True)


def _cluster_word_ids(
    values: np.ndarray,
    method: str,
    random_state: int,
    assign_noise: bool,
) -> tuple[np.ndarray, int]:
    """Discrete word ids from :mod:`stm.cluster`.

    *method* is ``"hdbscan"`` (UMAP + HDBSCAN, :class:`~stm.cluster.Cluster`)
    or ``"kmeans"`` (:class:`~stm.cluster.KMeansCluster`, K from the
    inertia-curve knee). Returns ``(labels, vocab_size)``.

    HDBSCAN noise stays ``-1`` when *assign_noise* is false;
    :func:`_singleton_noise` later gives each noise row its own id, so
    *vocab_size* is a lower bound that :meth:`TopicModelRunner.write_documents`
    corrects upward. K-Means never emits noise, so *assign_noise* is unused there.
    """
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"values must have shape (T, D), got {values.shape}")

    method = str(method).lower()
    if method == "kmeans":
        from perchtopic.cluster import KMeansCluster

        clustered = KMeansCluster(values, random_state=random_state)
    elif method == "hdbscan":
        from perchtopic.cluster import DensityCluster

        clustered = DensityCluster(values, random_state=random_state, assign_noise=assign_noise)
    else:
        raise ValueError(f"method must be 'kmeans' or 'hdbscan', got {method!r}")

    word_ids = np.asarray(clustered.labels, dtype=np.int64).reshape(-1)
    vocab = int(word_ids.max()) + 1 if np.any(word_ids >= 0) else 0
    n_noise = int(np.count_nonzero(word_ids < 0))
    print(
        f"Clustered {len(word_ids)} frames into {vocab} words ({n_noise} noise)",
        flush=True,
    )
    _print_word_counts(word_ids)
    return word_ids, vocab


def _probe_word_ids(
    values: np.ndarray,
    linear_model: Path | str | object,
) -> tuple[np.ndarray, int]:
    from perchtopic.classify import LinearModel

    if isinstance(linear_model, (str, Path)):
        path = Path(linear_model)
        if not path.is_file():
            raise FileNotFoundError(path)
        probe = LinearModel.load(path)
    elif isinstance(linear_model, LinearModel):
        probe = linear_model
    else:
        raise TypeError(f"linear_model must be a path or LinearModel, got {type(linear_model)}")

    in_features = probe.model.dense.in_features
    if values.shape[1] != in_features:
        raise ValueError(
            f"linear_model expects D={in_features} Perch2 embeddings, got D={values.shape[1]}"
        )
    word_ids = probe.predict_labels(values)
    vocab = len(probe.classes) if probe.classes else 0
    if len(word_ids):
        vocab = max(vocab, int(word_ids.max()) + 1)
    print(f"Predicted {len(word_ids)} labels from linear probe", flush=True)
    _print_word_counts(word_ids, probe.classes)
    return word_ids, vocab


def _scatter_ids(block: FeatureBlock, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (
        np.ones(len(block.grid), dtype=bool)
        if block.mask is None
        else np.asarray(block.mask, dtype=bool)
    )
    out = np.full(len(block.grid), -1, dtype=np.int64)
    ids = np.asarray(ids).reshape(-1)
    n_valid = int(mask.sum())
    if len(ids) == n_valid:
        out[mask] = ids
    elif len(ids) == len(block.grid):
        out = np.asarray(ids, dtype=np.int64)
        out[~mask] = -1
    else:
        raise ValueError(
            f"word ids length {len(ids)} matches neither grid T={len(block.grid)} "
            f"nor mask sum {n_valid}"
        )
    return out, mask


def _map_ids_to_grid(
    ids: np.ndarray,
    src_mask: np.ndarray,
    src: TimeGrid,
    target: TimeGrid,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int64)
    src_mask = np.asarray(src_mask, dtype=bool)
    if src == target:
        return ids, src_mask
    s_edges = src.edges
    t_edges = target.edges
    overlap = np.maximum(
        0.0,
        np.minimum(s_edges[:, 1:2], t_edges[:, 1])
        - np.maximum(s_edges[:, 0:1], t_edges[:, 0]),
    )
    overlap[~src_mask, :] = 0.0
    has = overlap.max(axis=0) > 0.0
    out = np.full(len(target), -1, dtype=np.int64)
    out[has] = ids[overlap.argmax(axis=0)[has]]
    return out, has


def _word_ids_from_blocks(
    blocks: Sequence[FeatureBlock],
    linear_model: Path | str | object | None,
    target: TimeGrid | None,
    random_state: int,
    assign_noise: bool,
    perch_method: str = "hdbscan",
    pcen_method: str = "kmeans",
) -> tuple[np.ndarray, int, TimeGrid, np.ndarray | None]:
    pcen_blocks = [item for item in blocks if _is_pcen(item)]
    perch_blocks = [item for item in blocks if not _is_pcen(item)]
    if len(perch_blocks) > 1:
        raise ValueError("run_from_block accepts at most one Perch2 FeatureBlock")

    streams: list[tuple[np.ndarray, np.ndarray, TimeGrid, int]] = []
    if perch_blocks:
        perch = perch_blocks[0]
        values = _masked_values(perch)
        if linear_model is not None:
            ids, vocab = _probe_word_ids(values, linear_model)
        else:
            ids, vocab = _cluster_word_ids(
                values,
                method=perch_method,
                random_state=random_state,
                assign_noise=assign_noise,
            )
        scattered, mask = _scatter_ids(perch, ids)
        streams.append((scattered, mask, perch.grid, vocab))
    if pcen_blocks:
        pcen = pcen_blocks[0] if len(pcen_blocks) == 1 else _fuse_blocks(pcen_blocks, target)
        ids, vocab = _cluster_word_ids(
            _masked_values(pcen),
            method=pcen_method,
            random_state=random_state,
            assign_noise=assign_noise,
        )
        scattered, mask = _scatter_ids(pcen, ids)
        streams.append((scattered, mask, pcen.grid, vocab))
    if not streams:
        raise ValueError("run_from_block requires a Perch2 or PCEN FeatureBlock")

    grid = target if target is not None else streams[0][2]
    mapped: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    offset = 0
    for ids, src_mask, src_grid, vocab in streams:
        shifted, dest_mask = _map_ids_to_grid(ids, src_mask, src_grid, grid)
        shifted = shifted.copy()
        valid = dest_mask & (shifted >= 0)
        shifted[valid] += offset
        mapped.append(shifted)
        masks.append(dest_mask)
        offset += max(int(vocab), 0)
    mask = np.logical_and.reduce(masks)
    if len(mapped) == 1:
        return mapped[0], offset, grid, mask
    return np.column_stack(mapped), offset, grid, mask


def _tail_file(path: Path, stop: threading.Event, chunks: list[str]) -> None:
    """Print new bytes from *path* until *stop* is set."""
    pos = 0
    while True:
        if path.exists():
            with path.open() as handle:
                handle.seek(pos)
                data = handle.read()
                pos = handle.tell()
            if data:
                chunks.append(data)
                print(data, end="", flush=True)
        if stop.wait(0.2):
            break
    if path.exists():
        with path.open() as handle:
            handle.seek(pos)
            data = handle.read()
        if data:
            chunks.append(data)
            print(data, end="", flush=True)


def _align_labels(
    labels: np.ndarray,
    grid: TimeGrid,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).reshape(-1)
    if mask is None:
        if len(labels) != len(grid):
            raise ValueError(
                f"labels length {len(labels)} does not match grid T={len(grid)}"
            )
        return labels, np.asarray(grid.starts)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(grid),):
        raise ValueError(f"mask must have shape ({len(grid)},), got {mask.shape}")
    n_valid = int(mask.sum())
    if len(labels) == n_valid:
        return labels, np.asarray(grid.starts)[mask]
    if len(labels) == len(grid):
        return labels[mask], np.asarray(grid.starts)[mask]
    raise ValueError(
        f"labels length {len(labels)} matches neither grid T={len(grid)} "
        f"nor mask sum {n_valid}"
    )


def _align_label_matrix(
    labels: np.ndarray,
    grid: TimeGrid,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2-D, got shape {labels.shape}")
    if mask is None:
        if labels.shape[0] != len(grid):
            raise ValueError(
                f"labels T={labels.shape[0]} does not match grid T={len(grid)}"
            )
        return labels, np.asarray(grid.starts)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(grid),):
        raise ValueError(f"mask must have shape ({len(grid)},), got {mask.shape}")
    n_valid = int(mask.sum())
    if labels.shape[0] == n_valid:
        return labels, np.asarray(grid.starts)[mask]
    if labels.shape[0] == len(grid):
        return labels[mask], np.asarray(grid.starts)[mask]
    raise ValueError(
        f"labels T={labels.shape[0]} matches neither grid T={len(grid)} "
        f"nor mask sum {n_valid}"
    )


def _singleton_noise(labels: np.ndarray) -> np.ndarray:
    vals = np.asarray(labels, dtype=np.int64).copy()
    noise = vals == -1
    if not np.any(noise):
        return vals
    clustered = vals[~noise]
    start = int(clustered.max()) + 1 if len(clustered) else 0
    vals[noise] = np.arange(start, start + int(noise.sum()))
    return vals


def _mount_root(docs_path: Path, out_dir: Path) -> Path:
    return Path(os.path.commonpath([str(docs_path.resolve()), str(out_dir.resolve())]))


def _read_ragged_csv(path: Path) -> pd.DataFrame:
    with path.open() as handle:
        max_cols = max(
            (line.count(",") + 1 for line in handle if line.strip()), default=1
        )
    return pd.read_csv(path, header=None, names=range(max_cols))


def _count_documents(docs_path: Path) -> int:
    with docs_path.open() as handle:
        return sum(1 for line in handle if line.strip())


def _write_model_meta(
    path: Path,
    document_seconds: float,
    num_documents: int,
    vocab_size: int,
    num_topics: int | None,
    docs_path: Path,
) -> Path:
    """Sidecar so :class:`~stm.topicmodel.plotter.Plotter` can size its
    own time axis instead of being told."""
    payload = {
        "document_seconds": float(document_seconds),
        "num_documents": int(num_documents),
        "vocab_size": int(vocab_size),
        "num_topics": None if num_topics is None else int(num_topics),
        "docs_path": str(docs_path),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {path}", flush=True)
    return path


def _infer_document_seconds(docs_path: Path) -> float:
    times = []
    with docs_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            times.append(int(line.split(",", 1)[0]))
    if len(times) >= 2:
        diffs = np.diff(np.asarray(times, dtype=np.float64))
        positive = diffs[diffs > 0]
        if len(positive):
            return float(np.median(positive) / 1000.0)
    if times and times[0] > 0:
        return float(times[0] / 1000.0)
    return 1.0


def _dominant_topic(topic_columns: pd.DataFrame) -> pd.Series:
    def _mode(row: pd.Series) -> int:
        values = row.dropna()
        if values.empty:
            return -1
        return int(np.bincount(values.to_numpy(dtype=np.int64)).argmax())

    return topic_columns.apply(_mode, axis=1).astype(np.int64)


def _write_phi_theta(
    topicmodel_path: Path,
    hist_path: Path,
    out_dir: Path,
    num_topics: int,
    vocab_size: int,
    alpha: float,
    beta: float,
) -> tuple[Path, Path]:
    topic_model = pd.read_csv(topicmodel_path, header=None).values
    topic_hist = pd.read_csv(hist_path, header=None).drop(columns=[0]).values
    phi = np.zeros((num_topics, vocab_size))
    for z in range(num_topics):
        denominator = np.sum(topic_model[z]) + (vocab_size * beta)
        if denominator == 0:
            continue
        for w in range(min(vocab_size, topic_model.shape[1])):
            phi[z][w] = (topic_model[z][w] + beta) / denominator
    theta = np.zeros((len(topic_hist), num_topics))
    for d in range(len(topic_hist)):
        denominator = np.sum(topic_hist[d]) + (num_topics * alpha)
        if denominator == 0:
            continue
        for z in range(min(num_topics, topic_hist.shape[1])):
            theta[d][z] = (topic_hist[d][z] + alpha) / denominator
    phi_path = out_dir / "phi.csv"
    theta_path = out_dir / "theta.csv"
    np.savetxt(phi_path, phi, delimiter=",")
    np.savetxt(theta_path, theta, delimiter=",")
    return phi_path, theta_path


def _write_maxlikelihood_with_time(
    maxlikelihood_path: Path,
    out_path: Path,
    document_seconds: float,
    collapse: bool,
) -> Path:
    df = _read_ragged_csv(maxlikelihood_path)
    ms_per_doc = document_seconds * 1000.0
    doc_indices = df[0].to_numpy(dtype=np.float64) * ms_per_doc
    if collapse and df.shape[1] > 1:
        df_time = pd.DataFrame(
            {0: doc_indices, 1: _dominant_topic(df.iloc[:, 1:])}
        )
    else:
        df_time = df.copy()
        df_time[df_time.columns[0]] = doc_indices
    df_time.to_csv(out_path, index=False, header=False)
    print(f"Saved: {out_path}", flush=True)
    return out_path