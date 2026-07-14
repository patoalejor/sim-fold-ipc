"""Matplotlib-based rendering: static PNGs and rotating MP4/GIF movies.

Kept dependency-light (matplotlib + imageio) so it runs on the CPU-only box.
Used to document each weekly milestone under images/.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

_BG = "#0e1117"
_FG = "#e6e6e6"
WRITE_MP4 = False   # GIFs are kept in-repo; MP4s are skipped by default
_TUBE = "#4c9be8"
_ROD = "#ff6b4a"


def _equal_aspect(ax, pts: np.ndarray):
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    c = 0.5 * (lo + hi)
    r = 0.5 * float((hi - lo).max()) + 1e-6
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _style(ax, title):
    ax.set_facecolor(_BG)
    ax.set_title(title, color=_FG, fontsize=12, pad=10)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(_FG)
        axis.set_pane_color((1, 1, 1, 0.02))
    ax.tick_params(colors=_FG, labelsize=7)
    ax.grid(False)


def _draw(ax, mesh=None, rod_pts=None, rod_radius=None):
    allpts = []
    if mesh is not None:
        tris = mesh.vertices[mesh.faces]
        alpha = 0.10 if rod_pts is not None else 0.18   # fainter wall when a rod is shown
        coll = Poly3DCollection(tris, alpha=alpha, facecolor=_TUBE,
                                edgecolor=(1, 1, 1, 0.05), linewidths=0.2)
        coll.set_zsort("min")
        ax.add_collection3d(coll)
        if mesh.centerline is not None:
            cl = mesh.centerline
            ax.plot(cl[:, 0], cl[:, 1], cl[:, 2], color=_TUBE, lw=0.8,
                    alpha=0.35, ls="--")
        allpts.append(mesh.vertices)
    if rod_pts is not None:
        segs = np.stack([rod_pts[:-1], rod_pts[1:]], axis=1)
        lc = Line3DCollection(segs, colors=_ROD, linewidths=4.5, zorder=10)
        ax.add_collection3d(lc)
        ax.scatter(rod_pts[:, 0], rod_pts[:, 1], rod_pts[:, 2],
                   color="#ffd23f", s=9, depthshade=False, zorder=11)
        # highlight the distal tip
        ax.scatter([rod_pts[-1, 0]], [rod_pts[-1, 1]], [rod_pts[-1, 2]],
                   color="#ffffff", s=28, depthshade=False, zorder=12)
        allpts.append(rod_pts)
    if allpts:
        _equal_aspect(ax, np.concatenate(allpts, axis=0))


def render_png(path, title, mesh=None, rod_pts=None, elev=22, azim=-60):
    fig = plt.figure(figsize=(7, 6), dpi=140, facecolor=_BG)
    ax = fig.add_subplot(111, projection="3d")
    _draw(ax, mesh, rod_pts)
    _style(ax, title)
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    fig.savefig(path, facecolor=_BG)
    plt.close(fig)
    return path


def _frame_to_array(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf.reshape(h, w, 4)[..., :3].copy()


def render_orbit(path_mp4, path_gif, title, mesh=None, rod_pts=None,
                 nframes=48, elev=22, fps=20):
    """Render a full-azimuth orbit; write MP4 and GIF."""
    frames = []
    for k in range(nframes):
        azim = -180 + 360 * k / nframes
        fig = plt.figure(figsize=(6, 5), dpi=110, facecolor=_BG)
        ax = fig.add_subplot(111, projection="3d")
        _draw(ax, mesh, rod_pts)
        _style(ax, title)
        ax.view_init(elev=elev, azim=azim)
        fig.tight_layout()
        frames.append(_frame_to_array(fig))
        plt.close(fig)
    imageio.mimsave(path_gif, frames, duration=1.0 / fps, loop=0)
    if WRITE_MP4 and path_mp4:
        try:
            imageio.mimsave(path_mp4, frames, fps=fps, quality=8,
                            macro_block_size=1)
        except Exception as exc:  # ffmpeg missing
            print(f"  (mp4 skipped: {exc})")
            path_mp4 = None
    else:
        path_mp4 = None
    return path_mp4, path_gif


def render_sequence(path_mp4, path_gif, title, mesh, rod_frames,
                    elev=22, azim=-60, fps=20):
    """Animate a rod moving through a static mesh (list of (N,3) arrays)."""
    frames = []
    base = [mesh.vertices] if mesh is not None else []
    allpts = np.concatenate(base + list(rod_frames), axis=0)
    for rp in rod_frames:
        fig = plt.figure(figsize=(6, 5), dpi=110, facecolor=_BG)
        ax = fig.add_subplot(111, projection="3d")
        _draw(ax, mesh, rp)
        _equal_aspect(ax, allpts)
        _style(ax, title)
        ax.view_init(elev=elev, azim=azim)
        fig.tight_layout()
        frames.append(_frame_to_array(fig))
        plt.close(fig)
    imageio.mimsave(path_gif, frames, duration=1.0 / fps, loop=0)
    try:
        imageio.mimsave(path_mp4, frames, fps=fps, quality=8,
                        macro_block_size=1)
    except Exception as exc:
        print(f"  (mp4 skipped: {exc})")
        path_mp4 = None
    return path_mp4, path_gif
