"""Dash/Plotly visualisation app for skeleton + muscle activations."""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def build_dash_app(
    mot_path: "str | Path",
    activations: np.ndarray,
    muscle_names: list[str],
    config: dict[str, Any],
    smplx_motion: np.ndarray | None = None,
) -> "dash.Dash":
    """
    Build and return a Dash app that visualises skeleton motion + muscle
    activations. The app is NOT started here; call app.run() at the call site.
    """
    try:
        from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
        import dash
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("Install plotly and dash: poetry add plotly dash") from exc

    from src.visualization import (
        _SMPL_PARENTS,
        _mean_act_for_segment,
        coords_to_skeleton_joints,
        get_smplx_skeleton_joints,
        load_mot_coords,
    )
    import matplotlib.cm as cm

    vis = config.get("visualization", {}) or {}
    scene_bg = str(vis.get("dash_scene_bg", "#1a1a1a"))
    bone_width = int(vis.get("dash_bone_width", 6))
    joint_size = int(vis.get("dash_joint_size", 5))
    frame_interval_ms = int(vis.get("frame_interval_ms", 33))

    coord_frames, times = load_mot_coords(mot_path)
    if len(coord_frames) == 0:
        raise ValueError(f"No frames found in .mot file: {mot_path}")
    n_frames = min(len(coord_frames), int(activations.shape[0]))
    coord_frames = coord_frames[:n_frames]
    times = times[:n_frames]
    activations = activations[:n_frames].astype(np.float32, copy=False)
    if smplx_motion is not None:
        smplx_motion = np.asarray(smplx_motion)[:n_frames]

    segs = [(int(_SMPL_PARENTS[i]), i) for i in range(24) if _SMPL_PARENTS[i] >= 0]

    tab20 = cm.get_cmap("tab20")
    muscle_rgba: dict[str, list[int]] = {}
    for i, mname in enumerate(muscle_names):
        rgba = tab20(i % 20)
        muscle_rgba[mname] = [int(v * 255) for v in rgba]

    def _joints_for_frame(t: int) -> np.ndarray:
        cf = coord_frames[t]
        tx = cf.get("pelvis_tx", 0.0)
        ty = cf.get("pelvis_ty", 0.9)
        tz = cf.get("pelvis_tz", 0.0)
        return coords_to_skeleton_joints(
            cf,
            pelvis_translation=np.array([tx, ty, tz], dtype=np.float64),
        ).astype(np.float32, copy=False)

    def _bone_hex(parent_idx: int, child_idx: int, t: int) -> str:
        act = _mean_act_for_segment(parent_idx, child_idx, t, activations, muscle_names)
        rgba = cm.coolwarm(float(act))
        rgb = tuple(int(255 * v) for v in rgba[:3])
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def _skeleton_figure(t: int) -> "go.Figure":
        joints = _joints_for_frame(t)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=joints[:, 0],
                y=joints[:, 1],
                z=joints[:, 2],
                mode="markers",
                marker={"size": joint_size, "color": "#666666"},
                name="joints",
            )
        )
        for p, c in segs:
            fig.add_trace(
                go.Scatter3d(
                    x=[joints[p, 0], joints[c, 0]],
                    y=[joints[p, 1], joints[c, 1]],
                    z=[joints[p, 2], joints[c, 2]],
                    mode="lines",
                    line={"width": bone_width, "color": _bone_hex(p, c, t)},
                    showlegend=False,
                )
            )

        if smplx_motion is not None:
            smplx_joints = get_smplx_skeleton_joints(smplx_motion[t]).astype(np.float32, copy=False)
            fig.add_trace(
                go.Scatter3d(
                    x=smplx_joints[:, 0],
                    y=smplx_joints[:, 1],
                    z=smplx_joints[:, 2],
                    mode="markers",
                    marker={"size": max(2, joint_size - 2), "color": "#7cdfff", "opacity": 0.55},
                    name="raw_smplx",
                )
            )

        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            scene={
                "xaxis_title": "X (right)",
                "yaxis_title": "Y (up)",
                "zaxis_title": "Z (fwd)",
                "aspectmode": "data",
                "bgcolor": scene_bg,
                # Keep world Y as the visual vertical axis in 3D view.
                "camera": {
                    "up": {"x": 0.0, "y": 1.0, "z": 0.0},
                    "eye": {"x": 1.6, "y": 0.8, "z": 1.6},
                },
            },
            paper_bgcolor=scene_bg,
            plot_bgcolor=scene_bg,
            font={"color": "#e5e5e5"},
        )
        return fig

    frame_idx = np.arange(n_frames)
    heatmap_base = go.Figure(
        data=[
            go.Heatmap(
                x=frame_idx,
                y=muscle_names,
                z=activations.T,
                colorscale="RdBu_r",
                zmin=0.0,
                zmax=1.0,
                colorbar={"title": "Activation"},
            )
        ]
    )
    heatmap_base.update_layout(
        margin={"l": 0, "r": 0, "t": 10, "b": 35},
        xaxis_title="Frame",
        yaxis_title="Muscle",
    )

    def _heatmap_figure(t: int) -> "go.Figure":
        hfig = go.Figure(heatmap_base)
        hfig.update_layout(shapes=[{
            "type": "line",
            "xref": "x",
            "yref": "paper",
            "x0": t,
            "x1": t,
            "y0": 0,
            "y1": 1,
            "line": {"color": "white", "width": 2},
        }])
        return hfig

    app = Dash(__name__)

    slider_step = max(1, n_frames // 10)
    slider_marks = {i: str(i) for i in range(0, n_frames, slider_step)}
    if (n_frames - 1) not in slider_marks:
        slider_marks[n_frames - 1] = str(n_frames - 1)

    app.layout = html.Div(
        [
            html.Div(
                [
                    dcc.Graph(id="skeleton-graph", figure=_skeleton_figure(0), style={"height": "78vh"}),
                ],
                style={"width": "55%", "display": "inline-block", "verticalAlign": "top"},
            ),
            html.Div(
                [
                    dcc.Graph(id="heatmap-graph", figure=_heatmap_figure(0), style={"height": "48vh"}),
                    html.Div(
                        [
                            dcc.Slider(
                                id="frame-slider",
                                min=0,
                                max=n_frames - 1,
                                step=1,
                                value=0,
                                marks=slider_marks,
                            ),
                            html.Div(
                                id="frame-label",
                                children=f"Frame 0 / {n_frames - 1}  —  t = {times[0]:.2f}s",
                                style={"marginTop": "8px"},
                            ),
                            html.Div(
                                [
                                    html.Button("Play", id="play-btn"),
                                    html.Button("Pause", id="pause-btn", style={"marginLeft": "8px"}),
                                ],
                                style={"marginTop": "8px"},
                            ),
                            html.Button("Export HTML", id="export-btn"),
                            dcc.Download(id="download"),
                            dcc.Interval(
                                id="play-interval",
                                interval=frame_interval_ms,
                                n_intervals=0,
                                disabled=True,
                            ),
                        ],
                        style={"padding": "12px"},
                    ),
                ],
                style={"width": "45%", "display": "inline-block", "verticalAlign": "top"},
            ),
        ],
        style={"width": "100%"},
    )

    @app.callback(
        Output("skeleton-graph", "figure"),
        Output("heatmap-graph", "figure"),
        Output("frame-label", "children"),
        Input("frame-slider", "value"),
    )
    def _on_frame_change(frame_value: int):
        t = int(frame_value)
        return (
            _skeleton_figure(t).to_dict(),
            _heatmap_figure(t).to_dict(),
            f"Frame {t} / {n_frames - 1}  —  t = {times[t]:.2f}s",
        )

    @app.callback(
        Output("play-interval", "disabled"),
        Input("play-btn", "n_clicks"),
        Input("pause-btn", "n_clicks"),
        State("play-interval", "disabled"),
        prevent_initial_call=True,
    )
    def _toggle_play(_play_clicks: int | None, _pause_clicks: int | None, current_disabled: bool):
        triggered = ctx.triggered_id
        if triggered == "play-btn":
            return False
        if triggered == "pause-btn":
            return True
        return current_disabled

    @app.callback(
        Output("frame-slider", "value"),
        Input("play-interval", "n_intervals"),
        State("frame-slider", "value"),
        prevent_initial_call=True,
    )
    def _advance_frame(_n_intervals: int, current_value: int):
        return (int(current_value) + 1) % n_frames

    @app.callback(
        Output("download", "data"),
        Input("export-btn", "n_clicks"),
    )
    def _export_html(n_clicks: int | None):
        if n_clicks is None:
            return no_update

        base_fig = _skeleton_figure(0)
        frames: list[go.Frame] = []
        for t in range(n_frames):
            f = _skeleton_figure(t)
            frames.append(go.Frame(name=str(t), data=f.data))

        base_fig.frames = frames
        base_fig.update_layout(
            updatemenus=[{
                "type": "buttons",
                "buttons": [{
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 40, "redraw": True}}],
                }],
            }]
        )

        buf = io.BytesIO()
        base_fig.write_html(buf, include_plotlyjs="cdn")
        _b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        logger.info("Prepared Plotly export payload (base64 size=%d).", len(_b64))
        return dcc.send_bytes(buf.getvalue(), "motion_preview.html")

    logger.info("Dash app prepared with %d frames and %d muscles.", n_frames, len(muscle_names))
    return app


if __name__ == "__main__":
    import argparse
    import yaml  # noqa: F401
    from src.visualization import load_mot_coords  # noqa: F401
    from src.utils import load_config

    p = argparse.ArgumentParser()
    p.add_argument("mot_path")
    p.add_argument("activations_npy")
    p.add_argument("muscle_names_json")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    acts = np.load(args.activations_npy).astype(np.float32)
    with open(args.muscle_names_json, encoding="utf-8") as f:
        names = json.load(f)
    app = build_dash_app(args.mot_path, acts, names, cfg)
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)

# ── Notebook usage ──────────────────────────────────────────────
# from src.visualization import show_dash_app
#
# # After running the OpenSim pipeline in the notebook:
# show_dash_app(
#     mot_path=mot_path,          # Path from run_pipeline_with_progress
#     activations=activations,    # [T, N_muscles] np.ndarray
#     muscle_names=muscle_names,  # list[str]
#     config=cfg,
#     smplx_motion=smplx_motion,  # optional: shows raw SMPL-X trace too
# )
#
# # Standalone CLI (no notebook needed):
# python -m src.dash_app motion.mot activations.npy muscle_names.json
# ────────────────────────────────────────────────────────────────
