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
    base_feed: float
    step_flow: float
    step_start: int
    step_end: int
    setpoints: tuple[float, float, float]
    initial_levels: tuple[float, float, float]
    kc: tuple[float, float, float]
    ti: tuple[float, float, float]
    theta: tuple[int, int, int]
    kff: tuple[float, float, float]
    noise: float


def defaults() -> Parameters:
    """Parámetros finales publicados para el banco; todos son editables."""
    return Parameters(
        duration=600,
        base_feed=45.0,
        step_flow=12.0,
        step_start=180,
        step_end=300,
        setpoints=(30.0, 33.0, 33.0),
        initial_levels=(30.0, 33.0, 33.0),
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
    feed[(time >= p.step_start) & (time < p.step_end)] += p.step_flow

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
        err[k] = np.asarray(p.setpoints) - y[k]

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
        data[f"SP{j}_cm"] = np.full(n, p.setpoints[i])
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
        st.header("Configuración numérica")
        st.caption("Cambie valores y pulse **Ejecutar simulación**. Los resultados no se dibujan como una animación.")
        with st.form("simulation_parameters", border=False):
            duration = st.slider("Duración del ensayo [s]", 120, 1200, current.duration, 30)
            base_feed = st.number_input("Caudal base de alimentación [lpm]", 5.0, 80.0, current.base_feed, 0.5)
            step_flow = st.slider("Perturbación ΔQfeed [lpm]", -25.0, 25.0, current.step_flow, 0.5)
            step_start = st.slider("Inicio de perturbación [s]", 0, duration - 1, min(current.step_start, duration - 1))
            step_end = st.slider("Fin de perturbación [s]", step_start + 1, duration, min(max(current.step_end, step_start + 1), duration))
            noise = st.slider("Ruido del sensor σ [cm]", 0.0, 0.15, current.noise, 0.01)
            st.divider()
            st.subheader("Niveles [cm]")
            sps, h0 = [], []
            for i in range(3):
                x, z = st.columns(2)
                sps.append(x.number_input(f"SP C{i+1}", 5.0, 37.0, current.setpoints[i], 0.1, key=f"sp{i}"))
                h0.append(z.number_input(f"Nivel inicial C{i+1}", 0.0, 39.0, current.initial_levels[i], 0.1, key=f"h0{i}"))
            st.divider()
            st.subheader("PI y feedforward")
            kcs, tis, thetas, kffs = [], [], [], []
            for i in range(3):
                with st.expander(f"Parámetros de celda {i+1}", expanded=(i == 0)):
                    kcs.append(st.number_input(f"Kc{i+1} [lpm/cm]", -150.0, -1.0, current.kc[i], 1.0, key=f"kc{i}"))
                    tis.append(st.number_input(f"Ti{i+1} [s]", 5.0, 180.0, current.ti[i], 1.0, key=f"ti{i}"))
                    thetas.append(st.slider(f"Retardo θ{i+1} [s]", 0, 20, int(current.theta[i]), key=f"theta{i}"))
                    kffs.append(st.slider(f"Kff{i+1}", 0.0, 1.5, current.kff[i], 0.05, key=f"kff{i}"))
            submitted = st.form_submit_button("▶ Ejecutar simulación", type="primary", use_container_width=True)

    if not submitted:
        return None, False
    return Parameters(
        duration=int(duration), base_feed=float(base_feed), step_flow=float(step_flow),
        step_start=int(step_start), step_end=int(step_end), setpoints=tuple(map(float, sps)),
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
        "Resultados del ensayo", "Modelo matemático", "Control PLC", "Calibración de bombas"
    ])

    with tab_res:
        a, b, c, d = st.columns(4)
        a.metric("IAE total · solo PI", f"{total_fb:.1f} cm·s")
        b.metric("IAE total · PI + FF", f"{total_ff:.1f} cm·s", f"{improvement:+.1f}%")
        c.metric("Muestreo", "1 s", f"{p.duration + 1} muestras")
        d.metric("Perturbación", f"{p.step_flow:+.1f} lpm", f"t = {p.step_start}–{p.step_end} s")

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

        st.subheader("Punto de inspección")
        instant = st.slider("Segundo que desea inspeccionar", 0, p.duration, min(p.duration, 300), key="instant")
        row = ff.iloc[instant]
        cols = st.columns(3)
        for i, col in enumerate(cols, 1):
            col.metric(f"C-{i} — Nivel medido", f"{row[f'PV{i}_cm']:.2f} cm", f"SP = {row[f'SP{i}_cm']:.1f} cm")
            col.caption(f"Q entrada = {row[f'Qin{i}_lpm']:.2f} lpm · P-{i} = {row[f'Qout{i}_lpm']:.2f} lpm · AO = {row[f'AO_P{i}_bits']:.0f} bits")

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
