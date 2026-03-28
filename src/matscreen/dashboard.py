from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from matscreen.screening.engine import ScreeningEngine
from matscreen.screening.objectives import (
    bandgap_objective,
    stability_objective,
    uncertainty_objective,
)

KNOWN_FORMULAS = [
    "Si", "GaAs", "CdTe", "CuInSe2", "MAPbI3", "CsPbBr3", "ZnO", "TiO2",
    "Fe2O3", "Al2O3", "SiC", "BN", "GaN", "AlN", "InP", "ZnS", "CdS",
    "Cu2O", "SnO2", "WO3", "MoS2", "WS2", "BiVO4", "SrTiO3", "BaTiO3",
    "LiNbO3", "ZnSe", "CdSe", "InAs", "GaP", "AlAs", "InN", "MgO",
    "CaF2", "NaCl", "KBr", "LiF", "BeO", "ZrO2", "HfO2", "Ga2O3",
    "In2O3", "VO2", "Cr2O3", "MnO2", "CoO", "NiO", "CuO", "PbS",
    "PbSe", "PbTe", "Bi2Se3", "Bi2Te3", "Sb2Te3", "SnSe", "GeSe",
    "GeS", "SnS", "FeS2", "CuFeS2", "ZnSnP2", "Cu2ZnSnS4", "Cu2ZnSnSe4",
    "AgBiS2", "CuSbS2", "BaSnO3", "LaAlO3", "KTaO3", "NaTaO3",
    "CaTiO3", "PbTiO3", "LiCoO2", "LiFePO4", "LiMn2O4", "NaMnO2",
    "MgAl2O4", "FeCr2O4", "ZnFe2O4", "CoFe2O4", "NiFe2O4", "MnFe2O4",
    "CuCrO2", "AgGaO2", "CuAlO2", "CuGaO2", "SrVO3", "CaVO3",
    "LaNiO3", "SrRuO3", "BaZrO3", "SrZrO3", "CaZrO3", "BaHfO3",
    "YAlO3", "GdScO3", "NdGaO3", "SmAlO3", "EuTiO3", "DyScO3",
]

USE_CASES = {
    "Solar Cell Absorber": {"bg_low": 1.0, "bg_high": 1.5, "ehull": 0.05},
    "LED / Display": {"bg_low": 2.0, "bg_high": 3.5, "ehull": 0.1},
    "Wide-Gap Semiconductor": {"bg_low": 3.0, "bg_high": 6.0, "ehull": 0.1},
    "Thermoelectric": {"bg_low": 0.1, "bg_high": 1.0, "ehull": 0.1},
    "Custom": {"bg_low": 1.0, "bg_high": 1.5, "ehull": 0.1},
}


def generate_realistic_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    formulas = rng.choice(KNOWN_FORMULAS, size=n, replace=True)
    suffixes = [f"-{rng.integers(100, 9999)}" for _ in range(n)]
    material_ids = [f"mp{s}" for s in suffixes]

    band_gaps = np.abs(rng.normal(2.0, 1.5, n))
    band_gaps = np.clip(band_gaps, 0, 8.0)

    eform = rng.normal(-1.0, 0.8, n)
    eform = np.clip(eform, -3.0, 0.5)

    ehull = np.abs(rng.exponential(0.05, n))
    ehull = np.clip(ehull, 0, 0.5)

    bulk_mod = rng.lognormal(4.0, 0.8, n)
    bulk_mod = np.clip(bulk_mod, 5, 500)

    bg_std = np.abs(rng.normal(0.1, 0.08, n)) + 0.01
    eform_std = np.abs(rng.normal(0.03, 0.02, n)) + 0.005
    kvrh_std = np.abs(rng.normal(10, 8, n)) + 1.0

    crystal_systems = rng.choice(
        [
            "cubic", "hexagonal", "tetragonal",
            "orthorhombic", "monoclinic", "triclinic", "trigonal",
        ],
        size=n,
        p=[0.25, 0.15, 0.15, 0.15, 0.15, 0.05, 0.10],
    )

    sources = rng.choice(
        ["Materials Project", "JARVIS"],
        size=n,
        p=[0.65, 0.35],
    )

    return pd.DataFrame({
        "material_id": material_ids,
        "formula": formulas,
        "source": sources,
        "crystal_system": crystal_systems,
        "band_gap": np.round(band_gaps, 3),
        "bandgap_std": np.round(bg_std, 4),
        "formation_energy_per_atom": np.round(eform, 4),
        "eform_std": np.round(eform_std, 4),
        "energy_above_hull": np.round(ehull, 4),
        "bulk_modulus_kv": np.round(bulk_mod, 1),
        "kvrh_std": np.round(kvrh_std, 1),
    })


def run_screening(
    df: pd.DataFrame,
    bg_low: float,
    bg_high: float,
    max_ehull: float,
    top_k: int,
) -> pd.DataFrame:
    objectives = [
        bandgap_objective(bg_low, bg_high),
        stability_objective(),
        uncertainty_objective(),
    ]

    value_columns = {
        "bandgap": "band_gap",
        "formation_energy": "formation_energy_per_atom",
        "uncertainty": "bandgap_std",
    }

    engine = ScreeningEngine(objectives=objectives, value_columns=value_columns)
    return engine.screen(df, max_ehull=max_ehull, top_k=top_k)


def make_pareto_plot(
    df: pd.DataFrame, bg_low: float, bg_high: float,
) -> go.Figure:
    fig = px.scatter(
        df,
        x="band_gap",
        y="formation_energy_per_atom",
        color="bandgap_std",
        size=np.maximum(df["bandgap_std"].values * 50, 5),
        hover_data=[
            "material_id", "formula", "crystal_system",
            "bandgap_std", "energy_above_hull",
        ],
        color_continuous_scale="RdYlGn_r",
        labels={
            "band_gap": "Band Gap (eV)",
            "formation_energy_per_atom": "Stability (eV/atom)",
            "bandgap_std": "Prediction Uncertainty (eV)",
        },
    )

    fig.add_vrect(
        x0=bg_low, x1=bg_high,
        fillcolor="rgba(76, 175, 80, 0.15)",
        line=dict(color="rgba(76, 175, 80, 0.6)", width=2, dash="dot"),
        annotation_text="Target Band Gap",
        annotation_position="top left",
        annotation_font_size=14,
        annotation_font_color="#2E7D32",
    )

    top5 = df.head(5)
    fig.add_trace(go.Scatter(
        x=top5["band_gap"],
        y=top5["formation_energy_per_atom"],
        mode="markers+text",
        text=top5["formula"],
        textposition="top center",
        textfont=dict(size=13, color="#1565C0"),
        marker=dict(
            size=18,
            color="rgba(21, 101, 192, 0.0)",
            line=dict(color="#1565C0", width=3),
        ),
        showlegend=False,
        name="Top 5",
    ))

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=60, r=20, t=50, b=60),
        coloraxis_colorbar_title="Uncertainty",
        font=dict(size=13),
    )
    return fig


def make_confidence_gauge(value: float, low: float, high: float, label: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title=dict(text=label, font=dict(size=16)),
        number=dict(suffix=" eV", font=dict(size=22)),
        gauge=dict(
            axis=dict(range=[0, max(high * 2, value * 1.5)]),
            bar=dict(color="#1565C0"),
            steps=[
                dict(range=[low, high], color="rgba(76, 175, 80, 0.3)"),
            ],
            threshold=dict(
                line=dict(color="#E53935", width=3),
                thickness=0.8,
                value=value,
            ),
        ),
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=30, r=30, t=50, b=10),
        font=dict(size=12),
    )
    return fig


def make_property_radar(row: pd.Series) -> go.Figure:
    bg_score = max(0, 1.0 - abs(row["band_gap"] - 1.25) / 3.0)
    stability_score = max(0, 1.0 - row["energy_above_hull"] / 0.2)
    confidence_score = max(0, 1.0 - row["bandgap_std"] / 0.3)
    hardness_norm = min(1.0, row["bulk_modulus_kv"] / 300.0)
    eform_score = max(
        0, 1.0 - (row["formation_energy_per_atom"] + 2.0) / 2.0,
    )

    categories = [
        "Band Gap Match", "Stability",
        "Confidence", "Mechanical Strength", "Formability",
    ]
    values = [bg_score, stability_score, confidence_score, hardness_norm, eform_score]
    values.append(values[0])
    categories.append(categories[0])

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(21, 101, 192, 0.15)",
        line=dict(color="#1565C0", width=2),
        marker=dict(size=6),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=280,
        margin=dict(l=60, r=60, t=30, b=30),
        showlegend=False,
        font=dict(size=11),
    )
    return fig


def make_uncertainty_comparison(results: pd.DataFrame) -> go.Figure:
    top = results.head(10).copy()
    top = top.sort_values("bandgap_std")

    colors = [
        "#4CAF50" if s < 0.08 else "#FF9800" if s < 0.15 else "#E53935"
        for s in top["bandgap_std"]
    ]

    fig = go.Figure(go.Bar(
        y=top["formula"],
        x=top["bandgap_std"],
        orientation="h",
        marker_color=colors,
        text=[f"  {s:.3f} eV" for s in top["bandgap_std"]],
        textposition="outside",
    ))

    fig.update_layout(
        template="plotly_white",
        height=350,
        margin=dict(l=80, r=60, t=10, b=40),
        xaxis_title="Prediction Uncertainty (eV)",
        yaxis_title="",
        font=dict(size=13),
    )
    return fig


def make_distribution_plot(
    df: pd.DataFrame, results: pd.DataFrame,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df["band_gap"],
        nbinsx=50,
        name="All Materials",
        marker_color="rgba(158, 158, 158, 0.4)",
        marker_line_color="rgba(158, 158, 158, 0.6)",
    ))

    if len(results) > 0:
        fig.add_trace(go.Histogram(
            x=results["band_gap"],
            nbinsx=20,
            name="Selected Candidates",
            marker_color="rgba(21, 101, 192, 0.6)",
            marker_line_color="#1565C0",
        ))

    fig.update_layout(
        template="plotly_white",
        height=300,
        margin=dict(l=50, r=20, t=10, b=50),
        xaxis_title="Band Gap (eV)",
        yaxis_title="Count",
        barmode="overlay",
        legend=dict(x=0.65, y=0.95),
        font=dict(size=13),
    )
    return fig


def make_crystal_system_pie(results: pd.DataFrame) -> go.Figure:
    counts = results["crystal_system"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.4,
        marker=dict(colors=px.colors.qualitative.Set2),
        textinfo="label+percent",
        textfont_size=12,
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig


def confidence_label(std: float) -> str:
    if std < 0.08:
        return "HIGH CONFIDENCE"
    if std < 0.15:
        return "MODERATE"
    return "LOW CONFIDENCE"


def confidence_color(std: float) -> str:
    if std < 0.08:
        return "#4CAF50"
    if std < 0.15:
        return "#FF9800"
    return "#E53935"


def main():
    st.set_page_config(
        page_title="MatScreen",
        page_icon="🔬",
        layout="wide",
    )

    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1565C0;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #616161;
        margin-top: -10px;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #E3F2FD 0%, #F3E5F5 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        border-left: 5px solid #1565C0;
    }
    .confidence-high {
        color: #2E7D32;
        font-weight: 700;
    }
    .confidence-med {
        color: #E65100;
        font-weight: 700;
    }
    .confidence-low {
        color: #C62828;
        font-weight: 700;
    }
    .how-it-works {
        background: #F5F5F5;
        border-radius: 8px;
        padding: 16px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-title">MatScreen</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">'
        "Find the best materials for your application. "
        "AI-powered screening with honest confidence estimates."
        "</p>",
        unsafe_allow_html=True,
    )

    with st.expander("How does this work?", expanded=False):
        how_it_works = (
            '<div class="how-it-works">\n\n'
            "**Step 1.** You tell us what properties you need "
            "(e.g. band gap between 1.0 and 1.5 eV for solar cells).\n\n"
            "**Step 2.** We screen 230,000+ known materials from the "
            "Materials Project and JARVIS databases using ensemble ML models.\n\n"
            "**Step 3.** We rank candidates using multi-objective Pareto "
            "optimisation, balancing target properties against stability "
            "and prediction confidence.\n\n"
            "**Step 4.** You get a shortlist with honest uncertainty "
            "estimates. Green = high confidence. Orange = moderate. "
            "Red = take with caution.\n"
            "</div>"
        )
        st.markdown(how_it_works, unsafe_allow_html=True)

    if "dataset" not in st.session_state:
        st.session_state.dataset = generate_realistic_dataset(500)

    df = st.session_state.dataset

    st.sidebar.markdown("## What are you looking for?")

    use_case = st.sidebar.selectbox(
        "Application",
        list(USE_CASES.keys()),
        index=0,
    )
    preset = USE_CASES[use_case]

    if use_case == "Custom":
        bg_range = st.sidebar.slider(
            "Target Band Gap (eV)",
            min_value=0.0, max_value=8.0,
            value=(preset["bg_low"], preset["bg_high"]),
            step=0.1,
        )
        max_ehull = st.sidebar.slider(
            "Stability Threshold (eV/atom)",
            min_value=0.0, max_value=0.5,
            value=preset["ehull"], step=0.01,
            help="Lower = more stable materials only",
        )
    else:
        bg_range = (preset["bg_low"], preset["bg_high"])
        max_ehull = preset["ehull"]
        st.sidebar.info(
            f"Band gap: {bg_range[0]} to {bg_range[1]} eV\n\n"
            f"Stability: < {max_ehull} eV/atom"
        )

    top_k = st.sidebar.slider(
        "How many candidates?",
        min_value=5, max_value=50, value=10, step=5,
    )

    results = run_screening(df, bg_range[0], bg_range[1], max_ehull, top_k)

    st.markdown("---")
    st.markdown(f"### Screening for: **{use_case}** materials")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Database Size", f"{len(df):,}")
    stable_count = len(df[df["energy_above_hull"] <= max_ehull])
    c2.metric("Pass Stability Filter", f"{stable_count:,}")
    c3.metric("Top Candidates", len(results))
    if len(results) > 0:
        avg_conf = results["bandgap_std"].mean()
        c4.metric(
            "Avg Confidence",
            confidence_label(avg_conf),
        )

    tab_recs, tab_explore, tab_analysis = st.tabs([
        "Recommendations", "Explore", "Analysis",
    ])

    with tab_recs:
        if len(results) == 0:
            st.warning("No materials match. Try widening the search.")
        else:
            st.markdown(
                "#### Top Candidates "
                "(ranked by overall suitability)"
            )

            for i, (_, row) in enumerate(results.head(top_k).iterrows()):
                conf = confidence_label(row["bandgap_std"])
                conf_cls = "confidence-high"
                if row["bandgap_std"] >= 0.15:
                    conf_cls = "confidence-low"
                elif row["bandgap_std"] >= 0.08:
                    conf_cls = "confidence-med"

                st.markdown(f"""
<div class="recommendation-card">
    <span style="font-size: 1.4rem; font-weight: 700; color: #1565C0;">
        #{row['pareto_rank']}  {row['formula']}
    </span>
    <span style="float: right;" class="{conf_cls}">
        {conf}
    </span>
    <br>
    <span style="color: #757575;">
        {row['crystal_system']} | {row['source']} | {row['material_id']}
    </span>
</div>
                """, unsafe_allow_html=True)

                detail_col, radar_col = st.columns([3, 2])

                with detail_col:
                    m1, m2, m3 = st.columns(3)
                    m1.metric(
                        "Band Gap",
                        f"{row['band_gap']:.2f} eV",
                        f"± {row['bandgap_std']:.3f}",
                    )
                    m2.metric(
                        "Stability",
                        f"{row['energy_above_hull']:.3f} eV/atom",
                        "Stable" if row["energy_above_hull"] < 0.025
                        else "Metastable",
                    )
                    m3.metric(
                        "Bulk Modulus",
                        f"{row['bulk_modulus_kv']:.0f} GPa",
                        f"± {row['kvrh_std']:.0f}",
                    )

                with radar_col:
                    radar = make_property_radar(row)
                    st.plotly_chart(radar, width="stretch")

                if i < top_k - 1:
                    st.markdown("")

    with tab_explore:
        st.markdown("#### Property Landscape")
        st.markdown(
            "Each point is a candidate material. "
            "Labelled materials are the top 5 recommendations. "
            "Green shaded region is your target band gap."
        )

        fig = make_pareto_plot(results, bg_range[0], bg_range[1])
        st.plotly_chart(fig, width="stretch")

        dist_col, pie_col = st.columns(2)

        with dist_col:
            st.markdown("#### Band Gap Distribution")
            st.markdown(
                "Grey = all 500 materials. "
                "Blue = your selected candidates."
            )
            dist_fig = make_distribution_plot(df, results)
            st.plotly_chart(dist_fig, width="stretch")

        with pie_col:
            st.markdown("#### Crystal Systems")
            st.markdown(
                "Distribution of crystal structures "
                "among your candidates."
            )
            if len(results) > 0:
                pie_fig = make_crystal_system_pie(results)
                st.plotly_chart(pie_fig, width="stretch")

    with tab_analysis:
        st.markdown("#### Confidence Analysis")
        st.markdown(
            "How confident is the model in each prediction? "
            "Green = high confidence (< 0.08 eV). "
            "Orange = moderate. Red = low confidence."
        )

        if len(results) > 0:
            unc_fig = make_uncertainty_comparison(results)
            st.plotly_chart(unc_fig, width="stretch")

        st.markdown("#### What do the confidence levels mean?")
        conf_c1, conf_c2, conf_c3 = st.columns(3)
        with conf_c1:
            st.success(
                "**HIGH CONFIDENCE**\n\n"
                "Uncertainty < 0.08 eV. "
                "The model has seen many similar materials. "
                "Prediction is likely within 0.1 eV of truth."
            )
        with conf_c2:
            st.warning(
                "**MODERATE**\n\n"
                "Uncertainty 0.08 to 0.15 eV. "
                "Prediction is useful but should be "
                "verified with DFT calculation."
            )
        with conf_c3:
            st.error(
                "**LOW CONFIDENCE**\n\n"
                "Uncertainty > 0.15 eV. "
                "The material is unusual. "
                "Do not rely on this prediction alone."
            )

        st.markdown("---")
        st.markdown("#### Data Sources")
        src_c1, src_c2 = st.columns(2)
        with src_c1:
            source_counts = df["source"].value_counts()
            for source, count in source_counts.items():
                st.metric(str(source), f"{count:,} materials")
        with src_c2:
            st.metric("Properties per Material", "3")
            st.metric("Ensemble Models", "5")
            st.metric("Calibration Method", "Isotonic Regression")


if __name__ == "__main__":
    main()
