"""Builds the Plotly figure (HA candles + EMAs + WaveTrend/money flow + equity
curve) and returns it as an embeddable HTML div -- no file writes, no browser."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.strategy import Params


def build_html(d: pd.DataFrame, tr: pd.DataFrame, a: Params, symbol: str, tf: str,
                div_id: str) -> str:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                         row_heights=[0.55, 0.30, 0.15])
    fig.add_trace(go.Candlestick(x=d.index, open=d.ha_open, high=d.ha_high, low=d.ha_low,
                                  close=d.ha_close, name="Heikin Ashi" if a.use_ha else "Price"), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d.fast, name=f"EMA {a.fast}", line=dict(color="#3b44e0", width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d.slow, name=f"EMA {a.slow}", line=dict(color="#5FA8B5", width=2)), 1, 1)
    for lvl in a.levels:
        fig.add_hline(y=lvl, line=dict(color="gold", width=2), row=1, col=1)
    if len(tr):
        fig.add_trace(go.Scatter(x=tr.entry_t, y=tr.entry, mode="markers", name="Buy",
                                  marker=dict(symbol="triangle-up", size=13, color="lime")), 1, 1)
        fig.add_trace(go.Scatter(x=tr.exit_t, y=tr.exit, mode="markers", name="Sell",
                                  marker=dict(symbol="triangle-down", size=13, color="red")), 1, 1)

    fig.add_trace(go.Scatter(x=d.index, y=d.mf, name="Money Flow", fill="tozeroy",
                              line=dict(color="rgba(255,255,255,0.8)", width=0)), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d.wt1, name="WT1", fill="tozeroy",
                              line=dict(color="#8FC7FF", width=2), fillcolor="rgba(73,148,236,0.45)"), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d.wt2, name="WT2", fill="tozeroy",
                              line=dict(color="#2a2080", width=1), fillcolor="rgba(31,21,89,0.75)"), 2, 1)
    g, r = d[d.green_dot], d[d.red_dot]
    fig.add_trace(go.Scatter(x=g.index, y=g.wt2, mode="markers", name="Green dot",
                              marker=dict(color="lime", size=9)), 2, 1)
    fig.add_trace(go.Scatter(x=r.index, y=r.wt2, mode="markers", name="Red dot",
                              marker=dict(color="red", size=9)), 2, 1)
    for y in (a.ob, 0, -a.ob):
        fig.add_hline(y=y, line=dict(color="gray", dash="dot" if y else "solid", width=1), row=2, col=1)

    fig.add_trace(go.Scatter(x=d.index, y=d.equity, name="Equity", line=dict(color="orange")), 3, 1)
    fig.update_layout(template="plotly_dark", title=f"{symbol} {tf} - HA WaveTrend",
                       xaxis_rangeslider_visible=False, height=850, hovermode="x unified",
                       margin=dict(t=50, b=30))
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)
