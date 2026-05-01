from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import EchoLifeConfig


def plot_z_heatmaps(z_by_room: Dict[str, np.ndarray], out_path: Path, title: str) -> None:
    rooms = sorted(z_by_room.keys())
    fig, axes = plt.subplots(len(rooms), 1, figsize=(12, 2.5 * len(rooms)), sharex=True, constrained_layout=True)
    if len(rooms) == 1:
        axes = [axes]
    for ax, room in zip(axes, rooms):
        z = z_by_room[room].T
        im = ax.imshow(z, aspect="auto", origin="lower", interpolation="nearest")
        ax.set_ylabel(room)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    axes[-1].set_xlabel("frame")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_traces(feats_by_room: Dict[str, Dict[str, np.ndarray]], out_path: Path, title: str) -> None:
    rooms = sorted(feats_by_room.keys())
    fig, axes = plt.subplots(len(rooms), 3, figsize=(16, 2.5 * len(rooms)), sharex=True, constrained_layout=True)
    if len(rooms) == 1:
        axes = np.asarray([axes])
    for i, room in enumerate(rooms):
        f = feats_by_room[room]
        axes[i, 0].plot(f["S"], lw=0.8)
        axes[i, 0].set_ylabel(room)
        axes[i, 0].set_title("S")
        axes[i, 1].plot(f["c"], lw=0.8)
        axes[i, 1].set_title("centroid")
        axes[i, 2].plot(f["spread"], lw=0.8)
        axes[i, 2].set_title("spread")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_room_assignment(
    segs: List[Tuple[str, int, int]],
    out_path: Path,
    title: str,
    cfg: EchoLifeConfig,
) -> None:
    labels = ["UNKNOWN"] + list(cfg.rooms)
    ymap = {lab: i for i, lab in enumerate(labels)}
    fig = plt.figure(figsize=(14, 3))
    for lab, s, e in segs:
        y = ymap.get(lab, len(ymap))
        plt.plot([s, e], [y, y], lw=4)
    plt.yticks(list(ymap.values()), list(ymap.keys()))
    plt.xlabel("frame")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

