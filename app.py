"""Simulador numérico de nivel para un banco de tres celdas de flotación.

El modelo se ejecuta en pasos discretos de un segundo. No es una animación:
cada ejecución calcula toda la trayectoria y sus indicadores de desempeño.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Banco de tres celdas | Simulador PI",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f6f8fc; color: #17233d; }
      [data-testid="stSidebar"] { background: #10233f; }
      [data-testid="stSidebar"] * { color: #eef5ff; }
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px; }
      .hero { background: linear-gradient(110deg, #071d38, #0b6075 60%, #1a8d94); color: white;
              padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1rem;
              box-shadow: 0 12px 25px rgba(10, 57, 87, .18); }
      .hero h1 { margin: 0; font-size: 2rem; }
      .hero p { margin: .35rem 0 0; color: #d6f4fa; }
      .eyebrow { text-transform: uppercase; letter-spacing: .11em; font-size: .72rem; font-weight: 700; color: #98e8e9; }
      .credit { color: #d6f4fa; font-size: .9rem; margin-top: .75rem; }
      .note { background:#e9f5f7; border-left: 5px solid #0b8ca6; border-radius: 8px;
              padding: .8rem 1rem; margin: .7rem 0; }
      div[data-testid="stMetric"] { background: white; border: 1px solid #dce5ef;
              border-radius: 12px; padding: .65rem .8rem; }
      .section-card { background: white; border: 1px solid #dce5ef; border-radius: 16px;
              padding: 1rem 1.1rem; box-shadow: 0 6px 16px rgba(20, 47, 76, .05); }
      .section-kicker { font-size: .72rem; font-weight: 700; letter-spacing: .1em;
              color:#157b87; text-transform:uppercase; }
      .plant-shell { background:linear-gradient(140deg,#0b2036,#123b4d); border-radius:16px;
              padding:18px 16px 14px; color:#edf8ff; overflow-x:auto; }
      .plant-topline { display:flex; justify-content:space-between; gap:12px; font-size:.76rem;
              color:#b7d7e5; margin-bottom:12px; }
      .plant-flow { color:#7fe1e6; font-weight:700; }
      .plant-row { display:flex; align-items:flex-end; gap:8px; min-width:620px; }
      .tank-wrap { width:31%; min-width:155px; }
      .tank-name { display:flex; justify-content:space-between; font-size:.78rem; margin:0 5px 6px; }
      .tank-tag { color:#a9d7e5; font-size:.64rem; text-transform:uppercase; letter-spacing:.06em; }
      .tank { position:relative; height:200px; border:2px solid #89bdcf; border-top-color:#d2edf5;
              border-radius:7px 7px 14px 14px; overflow:hidden; background:linear-gradient(#26495d,#153044); }
      .tank:before { content:''; position:absolute; z-index:3; left:-2px; right:-2px; top:25%; border-top:1px dashed rgba(213,244,250,.35); }
      .liquid { position:absolute; bottom:0; left:0; right:0; background:linear-gradient(180deg,#30c7dd,#057aa8);
              box-shadow:inset 0 8px 10px rgba(255,255,255,.20); transition:height .35s ease; }
      .liquid:after { content:''; position:absolute; left:0; right:0; top:0; border-top:3px solid #9df5ff; }
      .sp-line { position:absolute; z-index:4; left:4px; right:4px; border-top:2px dashed #ffbe57; }
      .level-readout { position:absolute; z-index:5; right:7px; top:7px; font-size:.76rem; font-weight:700;
              background:#f5ffff; color:#123246; border-radius:5px; padding:3px 5px; }
      .sensor { position:absolute; z-index:5; left:7px; top:7px; font-size:.60rem; letter-spacing:.04em; color:#eaffff; }
      .tank-foot { margin-top:7px; font-size:.69rem; color:#b8d9e7; display:flex; justify-content:space-between; }
      .connector { width:4%; align-self:center; padding-bottom:54px; color:#70dae2; font-size:1.25rem; text-align:center; }
      .monitor-card { background:#0e2945; color:white; border-radius:14px; padding:1rem; min-height:280px; }
      .monitor-card .sub { color:#a9ccdc; font-size:.79rem; }
      .signal { display:flex; align-items:center; justify-content:space-between; border-top:1px solid rgba(186,224,239,.20);
               padding:.56rem 0; font-size:.83rem; }
      .signal b { color:#71e5e6; font-size:.94rem; }
      .formula { background: #0f1e35; color: #e5efff; padding: 1rem; border-radius: 9px;
                  font-family: Consolas, monospace; white-space: pre-wrap; }
      code { color: #0a7188; }
    </style>
    """,
    unsafe_allow_html=True,
)

NAMES = ["Celda 1", "Celda 2", "Celda 3"]
COLORS = ["#1377d4", "#e26a1e", "#3b9c62"]
KPAPER = -0.001018  # cm/(s·lpm), Table 2
MAX_LEVEL = 39.0  # cm, rango de los sensores del montaje
MAX_FLOW = 95.0  # lpm, límite operativo de la bomba en este simulador
TAILING_SLOPE = 0.0029  # lpm / bit, dato de calibración de P-1
FEED_SLOPE = 0.0035  # lpm / bit, dato de calibración de alimentación


@dataclass(frozen=True)
class Parameters:
    duration: int
    scenario: str
    base_feed: float
    step_flow: float
    event_time: int
    setpoints: tuple[float, float, float]
    setpoints_after: tuple[float, float, float]
    initial_levels: tuple[float, float, float]
    kc: tuple[float, float, float]
    ti: tuple[float, float, float]
    theta: tuple[int, int, int]
    kff: tuple[float, float, float]
    noise: float


def defaults() -> Parameters:
    """Parámetros finales publicados para el banco; todos son editables."""
    return Parameters(
        duration=900,
        scenario="Perturbación de alimentación",
        base_feed=45.0,
        step_flow=15.0,
        event_time=180,
        setpoints=(29.0, 33.0, 32.0),
        setpoints_after=(30.0, 34.0, 33.0),
        initial_levels=(30.0, 34.0, 33.0),
        kc=(-62.0, -76.0, -85.0),
        ti=(56.0, 30.0, 44.0),
        theta=(7, 6, 3),
        kff=(0.8, 0.8, 1.0),
        noise=0.03,
    )


def median_sensor(history: list[float], value: float, window: int = 10) -> float:
    """Filtro de mediana de 10 s como el empleado en el trabajo de referencia."""
    history.append(value)
    if len(history) > window:
        history.pop(0)
    return float(np.median(history))


def pump_register(flow_lpm: float, slope: float = TAILING_SLOPE) -> float:
    """Convierte caudal al registro PLC de 16 bits y lo limita al rango válido."""
    return float(np.clip(flow_lpm / slope, 0.0, 32767.0))


def simulate(p: Parameters, feedforward: bool) -> pd.DataFrame:
    """Integra numéricamente las tres celdas interconectadas en pasos de 1 s.

    k' procede de la identificación experimental: dh/dt = k' (Qout - Qin).
    El signo negativo hace que mayor caudal de colas disminuya el nivel.
    """
    rng = np.random.default_rng(20261116)  # resultados repetibles para comparar estrategias
    n = int(p.duration) + 1
    time = np.arange(n, dtype=float)
    feed = np.full(n, p.base_feed, dtype=float)
    setpoint_schedule = np.tile(np.asarray(p.setpoints, dtype=float), (n, 1))
    is_feed_disturbance = p.scenario in {"Perturbación de alimentación", "Consigna y perturbación"}
    is_setpoint_change = p.scenario in {"Cambio de consigna", "Consigna y perturbación"}
    if is_feed_disturbance:
        feed[time >= p.event_time] += p.step_flow
    if is_setpoint_change:
        setpoint_schedule[time >= p.event_time] = np.asarray(p.setpoints_after, dtype=float)

    h = np.zeros((n, 3), dtype=float)
    y = np.zeros((n, 3), dtype=float)
    qin = np.zeros((n, 3), dtype=float)
    qout = np.zeros((n, 3), dtype=float)
    qcmd = np.zeros((n, 3), dtype=float)
    err = np.zeros((n, 3), dtype=float)

    h[0] = np.asarray(p.initial_levels, dtype=float)
    y[0] = h[0]
    qout[0] = p.base_feed
    qcmd[0] = p.base_feed
    qin[0] = (feed[0], p.base_feed, p.base_feed)

    integrals = np.zeros(3, dtype=float)
    sensor_histories: list[list[float]] = [[float(v)] for v in h[0]]
    delay_lines = [[p.base_feed] * (int(delay) + 1) for delay in p.theta]
    previous_disturbance = np.array(qin[0], dtype=float)

    for k in range(1, n):
        # El caudal que realmente llega a la etapa siguiente proviene de la bomba anterior.
        qin[k] = (feed[k], qout[k - 1, 0], qout[k - 1, 1])

        # Medición óptica más filtro de mediana; el ruido permite observar regulación realista.
        for i in range(3):
            measured = h[k - 1, i] + rng.normal(0.0, p.noise)
            y[k, i] = median_sensor(sensor_histories[i], measured)
        err[k] = setpoint_schedule[k] - y[k]

        # PI de posición con anti-windup. Kc negativo mantiene el signo físico del proceso.
        new_commands = np.empty(3, dtype=float)
        for i in range(3):
            candidate_integral = np.clip(integrals[i] + err[k, i], -25.0, 25.0)
            ff = p.kff[i] * (qin[k, i] - previous_disturbance[i]) if feedforward else 0.0
            raw = p.base_feed + p.kc[i] * (err[k, i] + candidate_integral / p.ti[i]) + ff
            command = float(np.clip(raw, 0.0, MAX_FLOW))
            # Solo acumula si no empuja más hacia saturación.
            if np.isclose(command, raw) or (command <= 0.0 and err[k, i] * p.kc[i] > 0) or (
                command >= MAX_FLOW and err[k, i] * p.kc[i] < 0
            ):
                integrals[i] = candidate_integral
            else:
                raw = p.base_feed + p.kc[i] * (err[k, i] + integrals[i] / p.ti[i]) + ff
                command = float(np.clip(raw, 0.0, MAX_FLOW))
            new_commands[i] = command

        qcmd[k] = new_commands
        # Retardo identificado de cada celda: 7 s, 6 s y 3 s por defecto.
        for i in range(3):
            delay_lines[i].append(new_commands[i])
            qout[k, i] = delay_lines[i].pop(0)

        # Balance de nivel discreto para los tres tanques en serie.
        dh = KPAPER * (qout[k] - qin[k])
        h[k] = np.clip(h[k - 1] + dh, 0.0, MAX_LEVEL)
        previous_disturbance = qin[k].copy()

    data: dict[str, np.ndarray] = {"t_s": time, "Q_feed_lpm": feed}
    for i in range(3):
        j = i + 1
        data[f"SP{j}_cm"] = setpoint_schedule[:, i]
        data[f"H{j}_cm"] = h[:, i]
        data[f"PV{j}_cm"] = y[:, i]
        data[f"E{j}_cm"] = err[:, i]
        data[f"Qin{j}_lpm"] = qin[:, i]
        data[f"Qout{j}_lpm"] = qout[:, i]
        data[f"Qcmd{j}_lpm"] = qcmd[:, i]
        data[f"AO_P{j}_bits"] = np.clip(qcmd[:, i] / TAILING_SLOPE, 0, 32767)
    return pd.DataFrame(data)


def iae(df: pd.DataFrame) -> list[float]:
    return [float(np.abs(df[f"E{i}_cm"]).sum()) for i in range(1, 4)]


def make_level_figure(feedback: pd.DataFrame, ff: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[f"Nivel de pulpa — {name}" for name in NAMES])
    for i in range(1, 4):
        color = COLORS[i - 1]
        fig.add_trace(go.Scatter(x=feedback.t_s, y=feedback[f"PV{i}_cm"], mode="lines",
                                 name=f"C{i}: PI", line=dict(color=color, dash="dot", width=2)), row=i, col=1)
        fig.add_trace(go.Scatter(x=ff.t_s, y=ff[f"PV{i}_cm"], mode="lines",
                                 name=f"C{i}: PI + FF", line=dict(color=color, width=3)), row=i, col=1)
        fig.add_trace(go.Scatter(x=ff.t_s, y=ff[f"SP{i}_cm"], mode="lines", name=f"SP C{i}",
                                 line=dict(color="#d7233f", dash="dash", width=1.5)), row=i, col=1)
        fig.update_yaxes(title_text="cm", range=[0, MAX_LEVEL + 1], row=i, col=1)
    fig.update_xaxes(title_text="Tiempo [s]", row=3, col=1)
    fig.update_layout(height=720, margin=dict(l=35, r=20, t=65, b=35),
                      legend=dict(orientation="h", y=1.05), hovermode="x unified",
                      paper_bgcolor="white", plot_bgcolor="white")
    fig.update_yaxes(gridcolor="#dbe4ee")
    fig.update_xaxes(gridcolor="#dbe4ee")
    return fig


def make_flow_figure(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[f"Caudales — {name}" for name in NAMES])
    for i in range(1, 4):
        fig.add_trace(go.Scatter(x=df.t_s, y=df[f"Qin{i}_lpm"], mode="lines", name=f"Q entrada C{i}",
                                 line=dict(color="#08a6b9", width=2)), row=i, col=1)
        fig.add_trace(go.Scatter(x=df.t_s, y=df[f"Qout{i}_lpm"], mode="lines", name=f"P-{i}: Q salida",
                                 line=dict(color="#19253d", width=2)), row=i, col=1)
        fig.add_trace(go.Scatter(x=df.t_s, y=df[f"Qcmd{i}_lpm"], mode="lines", name=f"Comando P-{i}",
                                 line=dict(color=COLORS[i-1], dash="dot")), row=i, col=1)
        fig.update_yaxes(title_text="lpm", range=[0, MAX_FLOW + 3], row=i, col=1)
    fig.update_xaxes(title_text="Tiempo [s]", row=3, col=1)
    fig.update_layout(height=720, margin=dict(l=35, r=20, t=65, b=35),
                      legend=dict(orientation="h", y=1.05), hovermode="x unified",
                      paper_bgcolor="white", plot_bgcolor="white")
    fig.update_yaxes(gridcolor="#dbe4ee")
    fig.update_xaxes(gridcolor="#dbe4ee")
    return fig


def make_plc_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, color in enumerate(COLORS, 1):
        fig.add_trace(go.Scatter(x=df.t_s, y=df[f"AO_P{i}_bits"], mode="lines", name=f"AO P-{i}",
                                 line=dict(color=color, width=2)))
    fig.add_hline(y=32767, line_dash="dash", line_color="#d7233f", annotation_text="Límite 16-bit")
    fig.update_layout(title="Salidas analógicas calculadas por el PLC", xaxis_title="Tiempo [s]",
                      yaxis_title="Registro PLC [bits]", height=390, hovermode="x unified",
                      paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=50, r=20, t=55, b=40))
    fig.update_yaxes(gridcolor="#dbe4ee", range=[0, 34000])
    fig.update_xaxes(gridcolor="#dbe4ee")
    return fig


def make_calibration_figure() -> go.Figure:
    bits = np.linspace(0, 30000, 120)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Bomba de alimentación P-4", "Bombas de colas P-1 / P-2 / P-3"])
    fig.add_trace(go.Scatter(x=bits, y=FEED_SLOPE * bits, mode="lines", name="Qfeed = 0.0035·bits",
                             line=dict(color="#167bbf", width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bits, y=TAILING_SLOPE * bits, mode="lines", name="Qcolas = 0.0029·bits",
                             line=dict(color="#e46b28", width=3)), row=1, col=2)
    # Puntos de referencia visuales; no son nuevas mediciones experimentales.
    for col, slope in [(1, FEED_SLOPE), (2, TAILING_SLOPE)]:
        samples = np.array([0, 5000, 10000, 15000, 20000, 28500])
        fig.add_trace(go.Scatter(x=samples, y=slope * samples, mode="markers", showlegend=False,
                                 marker=dict(color="#d7233f", size=7)), row=1, col=col)
        fig.update_xaxes(title_text="Registro entero de 16 bits", row=1, col=col)
        fig.update_yaxes(title_text="Caudal [lpm]", row=1, col=col)
    fig.update_layout(height=355, margin=dict(l=40, r=20, t=50, b=40), paper_bgcolor="white", plot_bgcolor="white")
    fig.update_xaxes(gridcolor="#dbe4ee")
    fig.update_yaxes(gridcolor="#dbe4ee")
    return fig


def scenario_note(scenario: str, p: Parameters) -> str:
    notes = {
        "Perturbación de alimentación": (
            f"La alimentación cambia de {p.base_feed:.1f} a {p.base_feed + p.step_flow:.1f} lpm "
            f"en t = {p.event_time} s; las consignas permanecen constantes."
        ),
        "Cambio de consigna": (
            f"Las consignas cambian en t = {p.event_time} s; la alimentación se mantiene en {p.base_feed:.1f} lpm."
        ),
        "Consigna y perturbación": (
            f"En t = {p.event_time} s cambian simultáneamente alimentación y consignas."
        ),
        "Equilibrio sin perturbaciones": "No se aplican eventos: sirve para observar el equilibrio del banco y el ruido de medición.",
    }
    return notes[scenario]


def plant_dashboard(row: pd.Series, selected: int, strategy: str, scale: str) -> None:
    """Dibuja el banco como una planta SVG, no como una animación ni texto HTML.

    El SVG se entrega dentro de un componente de Streamlit para que el navegador
    interprete todos los elementos del P&ID (tuberías, bombas y tanques).
    """

    tank_x = [245, 550, 855]
    tank_w, tank_top, tank_bottom = 160, 135, 360
    tank_height = tank_bottom - tank_top

    def tank_svg(cell: int, x: int) -> str:
        level = float(row[f"PV{cell}_cm"])
        sp = float(row[f"SP{cell}_cm"])
        qout = float(row[f"Qout{cell}_lpm"])
        if scale == "Nivel ampliado (±3 cm alrededor de SP)":
            fill_fraction = np.clip((level - (sp - 3.0)) / 6.0, 0.0, 1.0)
            sp_fraction = 0.5
            scale_label = "SP ±3 cm"
        else:
            fill_fraction = np.clip(level / MAX_LEVEL, 0.0, 1.0)
            sp_fraction = np.clip(sp / MAX_LEVEL, 0.0, 1.0)
            scale_label = "0–39 cm"

        water_y = tank_bottom - fill_fraction * tank_height
        sp_y = tank_bottom - sp_fraction * tank_height
        active = " • MONITOREO" if cell == selected else ""
        return f"""
          <g>
            <rect x="{x + 18}" y="72" width="124" height="27" rx="7" class="cell-label"/>
            <circle cx="{x + 33}" cy="85.5" r="4.2" class="ok-dot"/>
            <text x="{x + 47}" y="90" class="cell-title">CELDA {cell:02d}{active}</text>
            <rect x="{x + 50}" y="104" width="60" height="32" rx="5" class="lt-display"/>
            <text x="{x + 80}" y="118" text-anchor="middle" class="lt-small">LT-{100 + cell}</text>
            <text x="{x + 80}" y="130" text-anchor="middle" class="lt-value">{level:.1f}</text>
            <path d="M {x} {tank_top} L {x + 18} {tank_top - 16} L {x + tank_w - 18} {tank_top - 16} L {x + tank_w} {tank_top} Z" class="tank-roof"/>
            <rect x="{x}" y="{tank_top}" width="{tank_w}" height="{tank_height}" rx="3" class="tank-body"/>
            <rect x="{x + 7}" y="{water_y:.1f}" width="{tank_w - 14}" height="{tank_bottom - water_y:.1f}" class="water"/>
            <line x1="{x + 7}" x2="{x + tank_w - 7}" y1="{sp_y:.1f}" y2="{sp_y:.1f}" class="sp-line-svg"/>
            <text x="{x + 10}" y="{sp_y - 5:.1f}" class="sp-text">SP {sp:.1f}</text>
            <line x1="{x + 14}" x2="{x + 22}" y1="{tank_top + 42}" y2="{tank_top + 42}" class="scale-mark"/>
            <line x1="{x + 14}" x2="{x + 22}" y1="{tank_top + 98}" y2="{tank_top + 98}" class="scale-mark"/>
            <line x1="{x + 14}" x2="{x + 22}" y1="{tank_top + 154}" y2="{tank_top + 154}" class="scale-mark"/>
            <path d="M {x + tank_w - 20} {tank_top + 18} v150" class="ladder"/>
            <path d="M {x + tank_w - 38} {tank_top + 18} v150" class="ladder"/>
            <path d="M {x + tank_w - 38} {tank_top + 43} h18 M {x + tank_w - 38} {tank_top + 68} h18 M {x + tank_w - 38} {tank_top + 93} h18 M {x + tank_w - 38} {tank_top + 118} h18 M {x + tank_w - 38} {tank_top + 143} h18" class="ladder"/>
            <line x1="{x + 16}" x2="{x + 16}" y1="{tank_bottom}" y2="382" class="leg"/>
            <line x1="{x + tank_w - 16}" x2="{x + tank_w - 16}" y1="{tank_bottom}" y2="382" class="leg"/>
            <text x="{x + 80}" y="405" text-anchor="middle" class="value-title">PV {level:.2f} cm · SP {sp:.2f} cm</text>
            <text x="{x + 80}" y="422" text-anchor="middle" class="value-sub">P-{cell}: {qout:.1f} lpm · {scale_label}</text>
          </g>"""

    tank_shapes = "".join(tank_svg(cell, x) for cell, x in enumerate(tank_x, start=1))
    qfeed = float(row["Q_feed_lpm"])
    q1, q2, q3 = [float(row[f"Qout{i}_lpm"]) for i in range(1, 4)]
    selected_name = f"C-{selected:02d}"
    html = f"""
    <!DOCTYPE html><html><head><style>
      * {{ box-sizing:border-box; }} body {{ margin:0; font-family:Inter,Segoe UI,Arial,sans-serif; background:#fff; color:#15324b; }}
      .plant-card {{ border:1px solid #d7e7e9; border-radius:16px; padding:16px 18px 10px; background:linear-gradient(180deg,#ffffff,#f7fcfc); }}
      .top {{ display:flex; flex-wrap:wrap; justify-content:space-between; gap:10px; align-items:center; margin-bottom:10px; }}
      h2 {{ margin:0; font-size:20px; color:#0d3047; }} .sub {{ margin-top:4px; font-size:12px; color:#607e8e; }}
      .badge {{ border:1px solid #bad9dc; padding:7px 10px; border-radius:8px; color:#076d7d; font-size:12px; font-weight:700; background:#effafa; }}
      svg {{ width:100%; min-width:760px; height:auto; display:block; }}
      .grid {{ fill:url(#grid); }} .pipe {{ fill:none; stroke:#86aeb9; stroke-width:7; stroke-linecap:round; }} .pipe-flow {{ fill:none; stroke:#149bb0; stroke-width:2.5; stroke-dasharray:9 7; stroke-linecap:round; }}
      .flow-label {{ font-size:13px; font-weight:700; fill:#127a91; }} .equip-label {{ font-size:10px; fill:#4d7281; letter-spacing:1px; }}
      .tank-body {{ fill:#e6f1f3; stroke:#8eaeba; stroke-width:2; }} .tank-roof {{ fill:#f1f7f8; stroke:#8eaeba; stroke-width:2; }} .water {{ fill:url(#water); opacity:.93; }}
      .sp-line-svg {{ stroke:#f2b640; stroke-width:2.5; stroke-dasharray:6 5; }} .sp-text {{ font-size:10px; fill:#86651e; }} .scale-mark {{ stroke:#8caab6; stroke-width:1; }}
      .ladder {{ fill:none; stroke:#e5b631; stroke-width:3; }} .leg {{ stroke:#91aeb6; stroke-width:5; }} .cell-label {{ fill:#edf7f8; stroke:#c8dfe3; }} .ok-dot {{ fill:#35ac8a; }}
      .cell-title {{ font-size:11px; fill:#48717e; letter-spacing:1px; }} .lt-display {{ fill:#168faa; stroke:#0a6680; stroke-width:2; }} .lt-small {{ font-size:8px; fill:#d9fcff; }} .lt-value {{ font-size:13px; fill:white; font-weight:700; }}
      .pump {{ fill:#128ca5; stroke:#0b6880; stroke-width:2; }} .pump-plus {{ font-size:24px; fill:white; font-weight:300; }} .pump-foot {{ fill:#89b8c1; }} .plc {{ fill:#0c3853; stroke:#76cbd1; stroke-width:1.5; }} .plc-text {{ font-size:11px; fill:#def9fb; font-weight:700; letter-spacing:.5px; }} .control {{ stroke:#6caeb8; stroke-width:1.5; stroke-dasharray:5 5; fill:none; }}
      .value-title {{ font-size:13px; font-weight:700; fill:#113d54; }} .value-sub {{ font-size:11px; fill:#557783; }} .legend {{ display:flex; flex-wrap:wrap; gap:15px; padding:3px 0 7px; font-size:11px; color:#456a78; }} .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; background:#1da6de; margin-right:5px; }} .dash {{ display:inline-block; width:18px; border-top:2px dashed #f2b640; margin:0 5px 3px 0; }}
      .footer {{ border-top:1px solid #dcebed; padding-top:9px; font-size:11px; color:#617e88; display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap; }}
    </style></head><body>
      <div class="plant-card">
        <div class="top"><div><h2>Banco de flotación</h2><div class="sub">La salida de una celda alimenta la siguiente. Lectura calculada para {strategy}.</div></div><div class="badge">PLC-01 · ciclo de control 1 s</div></div>
        <div class="legend"><span><i class="dot"></i>Nivel simulado (PV)</span><span><i class="dash"></i>Consigna (SP)</span><span>━━ Flujo de pulpa</span><span>┄┄ Señal de control PLC</span></div>
        <svg viewBox="0 0 1210 445" role="img" aria-label="Diagrama del banco de tres celdas con caudales y niveles calculados">
          <defs><pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse"><path d="M 28 0 L 0 0 0 28" fill="none" stroke="#e8f2f3" stroke-width="1"/></pattern><linearGradient id="water" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#47c6ee"/><stop offset="1" stop-color="#168ec7"/></linearGradient><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#149bb0"/></marker></defs>
          <rect class="grid" x="0" y="0" width="1210" height="445" rx="9"/>
          <rect x="530" y="12" width="150" height="37" rx="7" class="plc"/><text x="605" y="28" text-anchor="middle" class="plc-text">PLC-01</text><text x="605" y="41" text-anchor="middle" class="lt-small">LIC-1 · LIC-2 · LIC-3</text>
          <path d="M 605 49 V 63 H 325 M 605 63 H 630 M 605 63 H 935" class="control"/><path d="M 325 63 V 104 M 630 63 V 104 M 935 63 V 104" class="control"/>
          <rect x="28" y="285" width="126" height="78" rx="7" fill="#d7e9ed" stroke="#91b5bd" stroke-width="2"/><text x="91" y="309" text-anchor="middle" class="equip-label">T-1 · RESERVORIO</text><text x="91" y="336" text-anchor="middle" class="cell-title">ALIMENTACIÓN</text>
          <path d="M 154 324 H 184" class="pipe"/><path d="M 154 324 H 184" class="pipe-flow" marker-end="url(#arrow)"/>
          <circle cx="204" cy="324" r="20" class="pump"/><text x="204" y="332" text-anchor="middle" class="pump-plus">+</text><text x="204" y="361" text-anchor="middle" class="equip-label">P-4</text>
          <path d="M 224 324 H 245" class="pipe"/><path d="M 224 324 H 245" class="pipe-flow" marker-end="url(#arrow)"/><text x="154" y="268" class="flow-label">Qfeed {qfeed:.1f} lpm</text>
          {tank_shapes}
          <path d="M405 329 H 425 M 465 329 H550 M710 329 H730 M770 329 H855 M1015 329 H1035 M1075 329 H1125 V385 H91 V363" class="pipe"/>
          <path d="M405 329 H 425 M465 329 H550 M710 329 H730 M770 329 H855 M1015 329 H1035 M1075 329 H1125 V385 H91 V363" class="pipe-flow" marker-end="url(#arrow)"/>
          <circle cx="445" cy="329" r="20" class="pump"/><text x="445" y="337" text-anchor="middle" class="pump-plus">+</text><text x="445" y="361" text-anchor="middle" class="equip-label">P-1</text>
          <circle cx="750" cy="329" r="20" class="pump"/><text x="750" y="337" text-anchor="middle" class="pump-plus">+</text><text x="750" y="361" text-anchor="middle" class="equip-label">P-2</text>
          <circle cx="1055" cy="329" r="20" class="pump"/><text x="1055" y="337" text-anchor="middle" class="pump-plus">+</text><text x="1055" y="361" text-anchor="middle" class="equip-label">P-3</text>
          <text x="478" y="310" class="flow-label">{q1:.1f} lpm</text><text x="783" y="310" class="flow-label">{q2:.1f} lpm</text><text x="1088" y="310" class="flow-label">{q3:.1f} lpm</text>
          <text x="570" y="397" class="equip-label">RETORNO A T-1 · DEPÓSITO NO DINÁMICO EN ESTE MODELO</text>
        </svg>
        <div class="footer"><span><b>Celda observada:</b> {selected_name}. El recuadro superior LT mide PV; la línea amarilla marca SP.</span><span><b>Interpretación:</b> si PV está sobre SP, el PLC ordena más descarga.</span></div>
      </div>
    </body></html>"""
    components.html(html, height=525, scrolling=False)


def published_benchmark() -> pd.DataFrame:
    """IAE informado en la Fig. 14 del artículo, para contraste experimental."""
    data = pd.DataFrame({
        "Celda": NAMES,
        "IAE publicado: PI [cm·s]": [198, 325, 260],
        "IAE publicado: PI + FF [cm·s]": [164, 187, 221],
    })
    data["Reducción publicada [%]"] = 100 * (
        data["IAE publicado: PI [cm·s]"] - data["IAE publicado: PI + FF [cm·s]"]
    ) / data["IAE publicado: PI [cm·s]"]
    return data


def p_and_id() -> None:
    st.graphviz_chart(
        """digraph bank {
        rankdir=LR; bgcolor="transparent"; node [shape=box style="rounded,filled" fontname="Arial" fontsize=13];
        feed [label="P-4 / alimentación\nQfeed" fillcolor="#cceff3"];
        c1 [label="C-1\nLIC-1" fillcolor="#d9ecff"];
        p1 [label="P-1\ncolas" fillcolor="#f9dfbd"];
        c2 [label="C-2\nLIC-2" fillcolor="#d9ecff"];
        p2 [label="P-2\ncolas" fillcolor="#f9dfbd"];
        c3 [label="C-3\nLIC-3" fillcolor="#d9ecff"];
        p3 [label="P-3\nretorno" fillcolor="#f9dfbd"];
        feed -> c1 -> p1 -> c2 -> p2 -> c3 -> p3;
        plc [label="PLC industrial\nPI + feedforward · Ts = 1 s" fillcolor="#d8f1d2"];
        plc -> p1 [style=dashed]; plc -> p2 [style=dashed]; plc -> p3 [style=dashed];
        c1 -> plc [style=dashed]; c2 -> plc [style=dashed]; c3 -> plc [style=dashed];
        }""",
        use_container_width=True,
    )


def parameter_form(current: Parameters) -> tuple[Parameters | None, bool]:
    with st.sidebar:
        st.header("Diseño del ensayo")
        st.caption("Configure un caso, calcule toda la trayectoria y compare las dos estrategias. No es una animación decorativa.")
        with st.expander("📘 Guía rápida: ¿qué parámetro debo cambiar?"):
            st.markdown(
                """**Para crear una perturbación:** cambie **ΔQfeed**. Representa que entra más o menos pulpa a C-01.

**Para pedir otro nivel:** cambie **SP inicial** o **SP posterior**. Es el nivel objetivo del controlador.

**Kc:** define qué tan fuerte reacciona la bomba ante un error. En este banco es negativo porque abrir la descarga baja el nivel.

**Ti:** define cuánto corrige el error acumulado. Un Ti menor elimina antes el error, pero puede volver la respuesta oscilatoria.

**θ (retardo):** tiempo que tarda una orden de bomba en afectar el nivel medido.

**Kff:** anticipa el efecto de una perturbación medida; déjelo en 0 para observar solo PI.

**Ruido del sensor:** imita incertidumbre de LT-101/LT-102/LT-103 sin cambiar la dinámica física."""
            )
        with st.form("simulation_parameters", border=False):
            scenarios = [
                "Perturbación de alimentación", "Cambio de consigna", "Consigna y perturbación", "Equilibrio sin perturbaciones"
            ]
            scenario = st.selectbox(
                "Escenario", scenarios, index=scenarios.index(current.scenario),
                help="Seleccione qué evento ocurrirá durante el ensayo numérico."
            )
            duration = st.slider(
                "Duración del ensayo [s]", 180, 1200, current.duration, 30,
                help="Tiempo total que se integra el modelo. Cada punto de las gráficas equivale a un segundo."
            )
            event_enabled = scenario != "Equilibrio sin perturbaciones"
            event_time = st.slider(
                "Primer evento [s]", 1, duration - 1, min(current.event_time, duration - 1), disabled=not event_enabled,
                help="Segundo en que se aplica el cambio de alimentación o de consigna."
            )
            base_feed = st.number_input(
                "Caudal base de alimentación [lpm]", 5.0, 80.0, current.base_feed, 0.5,
                help="Caudal nominal de pulpa que entra al banco por la bomba P-4."
            )
            feed_enabled = scenario in {"Perturbación de alimentación", "Consigna y perturbación"}
            step_flow = st.slider(
                "Cambio de alimentación ΔQfeed [lpm]", -25.0, 25.0, current.step_flow, 0.5, disabled=not feed_enabled,
                help="Perturbación aplicada a P-4. Un valor positivo aumenta la entrada de pulpa."
            )
            st.caption(scenario_note(scenario, Parameters(
                duration, scenario, base_feed, step_flow, event_time, current.setpoints, current.setpoints_after,
                current.initial_levels, current.kc, current.ti, current.theta, current.kff, current.noise
            )))
            noise = st.slider(
                "Ruido del sensor σ [cm]", 0.0, 0.15, current.noise, 0.01,
                help="Variación aleatoria que se añade a los sensores de nivel antes del filtro de mediana."
            )
            st.divider()
            st.subheader("Niveles y consignas [cm]")
            st.caption("SP inicial / SP después del evento / nivel inicial")
            sps, sps_after, h0 = [], [], []
            sp_change_enabled = scenario in {"Cambio de consigna", "Consigna y perturbación"}
            for i in range(3):
                st.markdown(f"**Celda {i + 1:02d}**")
                x, y, z = st.columns(3)
                sps.append(x.number_input(
                    "SP inicial", 5.0, 37.0, current.setpoints[i], 0.1, key=f"sp_initial_{i}",
                    help="Nivel objetivo antes del evento."
                ))
                sps_after.append(y.number_input(
                    "SP posterior", 5.0, 37.0, current.setpoints_after[i], 0.1,
                    key=f"sp_after_{i}", disabled=not sp_change_enabled,
                    help="Nuevo nivel objetivo después del evento; se habilita en escenarios con cambio de consigna."
                ))
                h0.append(z.number_input(
                    "Nivel inicial", 0.0, 39.0, current.initial_levels[i], 0.1, key=f"h0_{i}",
                    help="Nivel con el que inicia la integración de la celda."
                ))
            st.divider()
            st.subheader("PI y feedforward")
            kcs, tis, thetas, kffs = [], [], [], []
            for i in range(3):
                with st.expander(f"Parámetros de celda {i+1}", expanded=(i == 0)):
                    kcs.append(st.number_input(
                        f"Kc{i+1} [lpm/cm]", -150.0, -1.0, current.kc[i], 1.0, key=f"kc{i}",
                        help="Ganancia proporcional. Mayor magnitud = corrección de bomba más agresiva."
                    ))
                    tis.append(st.number_input(
                        f"Ti{i+1} [s]", 5.0, 180.0, current.ti[i], 1.0, key=f"ti{i}",
                        help="Tiempo integral. Menor valor = el error persistente se corrige más rápido."
                    ))
                    thetas.append(st.slider(
                        f"Retardo θ{i+1} [s]", 0, 20, int(current.theta[i]), key=f"theta{i}",
                        help="Retraso de transporte entre la orden de bomba y la respuesta medida."
                    ))
                    kffs.append(st.slider(
                        f"Kff{i+1}", 0.0, 1.5, current.kff[i], 0.05, key=f"kff{i}",
                        help="Ganancia de compensación anticipada. Use 0 para desactivarla."
                    ))
            submitted = st.form_submit_button("▶ Calcular ensayo completo", type="primary", use_container_width=True)

    if not submitted:
        return None, False
    return Parameters(
        duration=int(duration), scenario=scenario, base_feed=float(base_feed), step_flow=float(step_flow),
        event_time=int(event_time), setpoints=tuple(map(float, sps)),
        setpoints_after=tuple(map(float, sps_after if sp_change_enabled else sps)),
        initial_levels=tuple(map(float, h0)), kc=tuple(map(float, kcs)), ti=tuple(map(float, tis)),
        theta=tuple(map(int, thetas)), kff=tuple(map(float, kffs)), noise=float(noise),
    ), True


def main() -> None:
    if "parameters" not in st.session_state:
        st.session_state.parameters = defaults()
        st.session_state.feedback = simulate(st.session_state.parameters, feedforward=False)
        st.session_state.feedforward = simulate(st.session_state.parameters, feedforward=True)

    selected, submitted = parameter_form(st.session_state.parameters)
    if submitted and selected is not None:
        st.session_state.parameters = selected
        with st.spinner("Integrando el balance de los tres tanques y calculando los dos controles..."):
            st.session_state.feedback = simulate(selected, feedforward=False)
            st.session_state.feedforward = simulate(selected, feedforward=True)
        st.success("Simulación completada: se calcularon todas las muestras de 1 s.")

    p: Parameters = st.session_state.parameters
    feedback: pd.DataFrame = st.session_state.feedback
    ff: pd.DataFrame = st.session_state.feedforward
    iae_fb, iae_ff = iae(feedback), iae(ff)
    total_fb, total_ff = sum(iae_fb), sum(iae_ff)
    improvement = 100 * (total_fb - total_ff) / total_fb if total_fb else 0.0

    st.markdown(
        """<section class="hero">
          <div class="eyebrow">Automatización de Procesos · Banco piloto de flotación</div>
          <h1>Simulador industrial de tres celdas de flotación</h1>
          <p>Modelo discreto de C-1 → C-2 → C-3 · PI con retardo, sensores filtrados y compensación feedforward.</p>
          <div class="credit"><b>Elaborado por:</b> Antonio Robles Meza &nbsp;|&nbsp; <b>Docente:</b> José Luis Sosa Sánchez</div>
        </section>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='note'><b>Simulación numérica:</b> cada ensayo integra el balance de masa a 1 s y luego calcula IAE, caudales y registros PLC. Las curvas son resultados del modelo, no una animación decorativa.</div>",
        unsafe_allow_html=True,
    )

    tab_res, tab_model, tab_plc, tab_cal = st.tabs([
        "Banco y resultados", "Modelo matemático", "Control PLC", "Calibración de bombas"
    ])

    with tab_res:
        a, b, c, d = st.columns(4)
        a.metric("IAE total · solo PI", f"{total_fb:.1f} cm·s")
        b.metric("IAE total · PI + FF", f"{total_ff:.1f} cm·s", f"{improvement:+.1f}%")
        c.metric("Muestreo", "1 s", f"{p.duration + 1} muestras")
        event_label = "Sin eventos" if p.scenario == "Equilibrio sin perturbaciones" else f"t = {p.event_time} s"
        d.metric("Escenario activo", p.scenario.replace(" de ", "\n"), event_label)

        view_a, view_b = st.columns([1, 1.25])
        with view_a:
            strategy = st.selectbox(
                "Estrategia mostrada en el banco",
                ["PI + feedforward", "Solo PI"],
                help="Cambia las lecturas del diagrama y del monitor. Las dos curvas se conservan en la gráfica comparativa."
            )
        with view_b:
            scale = st.selectbox(
                "Escala visual del nivel",
                ["Nivel real (0–39 cm)", "Nivel ampliado (±3 cm alrededor de SP)"],
                help="La escala ampliada hace visible una desviación pequeña; las cifras PV y SP siempre son los valores reales."
            )

        instant = st.slider(
            "Instante de inspección del banco [s]", 0, p.duration, min(p.duration, p.event_time + 60), key="instant"
        )
        display_df = ff if strategy == "PI + feedforward" else feedback
        row = display_df.iloc[instant]
        bank, monitor = st.columns([1.7, 0.8], gap="large")
        with bank:
            st.markdown("<div class='section-kicker'>Banco conectado · variables calculadas</div>", unsafe_allow_html=True)
            st.subheader("Banco de flotación en serie")
            selected_cell = st.radio("Celda para monitorear", [1, 2, 3], horizontal=True, format_func=lambda x: f"C-{x:02d}")
            plant_dashboard(row, selected_cell, strategy, scale)
            st.caption("Nivel azul: PV filtrado. Línea amarilla: consigna. Cada bomba descarga la celda anterior y alimenta la siguiente.")
        with monitor:
            i = selected_cell
            error = row[f"SP{i}_cm"] - row[f"PV{i}_cm"]
            status = "EN BANDA" if abs(error) <= 0.25 else "CORRIGIENDO"
            st.markdown(
                f"""<div class="monitor-card"><div class="section-kicker" style="color:#7fe1e6">Monitor de instrumentación</div>
                <h3 style="margin:.25rem 0 .2rem">Celda C-{i:02d}</h3><div class="sub">{strategy} · segundo {instant} · {status}</div>
                <div class="signal"><span>LT-{100 + i} · PV</span><b>{row[f'PV{i}_cm']:.2f} cm</b></div>
                <div class="signal"><span>LIC-{i:02d} · SP</span><b>{row[f'SP{i}_cm']:.2f} cm</b></div>
                <div class="signal"><span>Error e(t)</span><b>{error:+.2f} cm</b></div>
                <div class="signal"><span>AI · Qin</span><b>{row[f'Qin{i}_lpm']:.2f} lpm</b></div>
                <div class="signal"><span>AO · P-{i}</span><b>{row[f'AO_P{i}_bits']:.0f} bits</b></div>
                <div class="signal"><span>Q colas</span><b>{row[f'Qout{i}_lpm']:.2f} lpm</b></div></div>""",
                unsafe_allow_html=True,
            )

        st.markdown(f"<div class='note'><b>{p.scenario}.</b> {scenario_note(p.scenario, p)}</div>", unsafe_allow_html=True)

        st.subheader("Nivel de pulpa: comparación de estrategias")
        st.plotly_chart(make_level_figure(feedback, ff), width="stretch")
        st.caption("Línea punteada: PI por realimentación. Línea continua: PI + feedforward. Línea roja: consigna.")

        st.subheader("Caudales con PI + feedforward")
        st.plotly_chart(make_flow_figure(ff), width="stretch")

        st.subheader("Indicadores cuantitativos")
        summary = pd.DataFrame({
            "Celda": NAMES + ["Total"],
            "IAE solo PI [cm·s]": iae_fb + [total_fb],
            "IAE PI + FF [cm·s]": iae_ff + [total_ff],
        })
        summary["Mejora IAE [%]"] = 100 * (summary["IAE solo PI [cm·s]"] - summary["IAE PI + FF [cm·s]"]) / summary["IAE solo PI [cm·s]"]
        st.dataframe(summary.style.format({
            "IAE solo PI [cm·s]": "{:.2f}", "IAE PI + FF [cm·s]": "{:.2f}", "Mejora IAE [%]": "{:+.1f}%"
        }), width="stretch", hide_index=True)

        st.subheader("Referencia experimental del artículo")
        st.caption("El cuadro contrasta los IAE publicados para el banco físico; no sustituye la simulación calculada arriba.")
        benchmark = published_benchmark()
        st.dataframe(benchmark.style.format({"Reducción publicada [%]": "{:+.1f}%"}), width="stretch", hide_index=True)

        csv = ff.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar resultados PI + FF en CSV", csv, "resultados_tres_celdas.csv", "text/csv")

    with tab_model:
        left, right = st.columns([1.1, 1])
        with left:
            st.subheader("Interconexión del banco")
            p_and_id()
        with right:
            st.subheader("Balance y dinámica identificada")
            st.markdown(
                """<div class="formula">Para i = 1, 2, 3:
dhᵢ/dt = k′ · (Qoutᵢ − Qinᵢ)
k′ = −0.001018 cm·s⁻¹·lpm⁻¹

Qin₁ = Qfeed
Qin₂ = Qout₁
Qin₃ = Qout₂

Gpᵢ(s) = (−0.001018 / s) · e^(−θᵢs)</div>""",
                unsafe_allow_html=True,
            )
            st.write("El signo negativo significa que al aumentar la bomba de colas, el nivel de esa celda disminuye.")
            st.markdown("**Estrategia PI + feedforward**")
            st.latex(r"u_i(t)=K_{c,i}\left[e_i(t)+\frac{1}{T_{i,i}}\int_0^t e_i(\tau)d\tau\right]+K_{ff,i}\Delta Q_{d,i}")
            st.latex(r"\Delta Q_{d,i}=Q_{d,i}(t)-Q_{d,i}(t-1)")
            st.caption("Para C-1 la perturbación medida es Qfeed; para C-2 es la salida de C-1; para C-3 es la salida de C-2.")

        st.subheader("Parámetros del ensayo activo")
        model_table = pd.DataFrame({
            "Celda": NAMES,
            "k′ [cm/(s·lpm)]": [KPAPER] * 3,
            "θ [s]": p.theta,
            "Kc [lpm/cm]": p.kc,
            "Ti [s]": p.ti,
            "Kff": p.kff,
        })
        st.dataframe(model_table.style.format({"k′ [cm/(s·lpm)]": "{:.6f}", "Kc [lpm/cm]": "{:.1f}", "Kff": "{:.2f}"}),
                     width="stretch", hide_index=True)
        st.caption("Los valores de inicio reproducen los parámetros finales de la Tabla 2; se pueden cambiar en el panel lateral para experimentar con la sintonización.")

    with tab_plc:
        st.subheader("Señales de salida calculadas")
        st.plotly_chart(make_plc_figure(ff), width="stretch")
        st.caption("El PLC trabaja con registros 0–32767. El simulador convierte la orden de caudal de las bombas de colas con 0.0029 lpm/bit.")
        st.subheader("Lógica PLC equivalente (ciclo cada 1 s)")
        st.code(
            """NETWORK 1  // Leer y filtrar niveles
PV_i := MEDIANA_10S(AI_NIVEL_i);
e_i  := SP_i - PV_i;

NETWORK 2  // PI con anti-windup
I_i := LIMIT(-25.0, I_i + e_i * 1.0, 25.0);
Q_PI_i := Kc_i * (e_i + I_i / Ti_i);

NETWORK 3  // Feedforward de la perturbación medida
DeltaQd_i := Qin_i - Qin_i_ANTERIOR;
Q_CMD_i := LIMIT(0.0, Q_BASE + Q_PI_i + Kff_i * DeltaQd_i, 95.0);
AO_P_i := LIMIT(0, Q_CMD_i / 0.0029, 32767);
Qin_i_ANTERIOR := Qin_i;""",
            language="text",
        )
        st.info("Esta lógica ilustra el algoritmo para un PLC. Para un PLC real se deben adaptar tipos de datos, escalamiento y bloques PID al fabricante (Siemens, Allen-Bradley, Schneider, etc.).")

    with tab_cal:
        st.subheader("Curvas de calibración utilizadas")
        st.plotly_chart(make_calibration_figure(), width="stretch")
        st.markdown(
            """- Alimentación P-4: `Qfeed = 0.0035 × registro` lpm.
            - Colas P-1: `Qtailings = 0.0029 × registro` lpm.
            - Para P-2 y P-3 el simulador usa la misma pendiente de P-1 como aproximación editable a nivel de código; en una planta se calibran por separado.""")
        st.warning("Las marcas rojas de esta gráfica representan la recta de calibración, no datos nuevos. Los caudales de las gráficas de resultados se obtienen al integrar el modelo de control.")


if __name__ == "__main__":
    main()
