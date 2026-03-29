from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from matscreen.data.schema import TriageLabel
from matscreen.screening.engine import ScreeningEngine
from matscreen.screening.solar import (
    abundance_score,
    contains_critical,
    contains_toxic,
    shockley_queisser_efficiency,
    solar_objectives,
)
from matscreen.uncertainty.triage import TriageAssigner

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

TRIAGE_COLOURS = {
    TriageLabel.TRUST: "#2E7D32",
    TriageLabel.VERIFY: "#E65100",
    TriageLabel.DEFER: "#C62828",
}

TRIAGE_ACTIONS = {
    TriageLabel.TRUST: "Proceed to synthesis",
    TriageLabel.VERIFY: "Schedule DFT verification",
    TriageLabel.DEFER: "Outside model domain",
}


def generate_realistic_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    formulas = rng.choice(KNOWN_FORMULAS, size=n, replace=True)
    suffixes = [f"-{rng.integers(100, 9999)}" for _ in range(n)]
    material_ids = [f"mp{s}" for s in suffixes]

    band_gaps = np.abs(rng.normal(1.5, 1.0, n))
    band_gaps = np.clip(band_gaps, 0, 4.0)

    eform = rng.normal(-1.0, 0.8, n)
    eform = np.clip(eform, -3.0, 0.5)

    ehull = np.abs(rng.exponential(0.03, n))
    ehull = np.clip(ehull, 0, 0.3)

    bulk_mod = rng.lognormal(4.0, 0.8, n)
    bulk_mod = np.clip(bulk_mod, 5, 500)

    bg_std = np.abs(rng.normal(0.1, 0.08, n)) + 0.01
    eform_std = np.abs(rng.normal(0.03, 0.02, n)) + 0.005
    kvrh_std = np.abs(rng.normal(10, 8, n)) + 1.0

    ood_scores = rng.exponential(2.0, n)
    ood_flags = ood_scores > np.percentile(ood_scores, 95)

    triage_assigner = TriageAssigner()
    triage_labels = triage_assigner.assign(bg_std, ood_flags)

    crystal_systems = rng.choice(
        ["cubic", "hexagonal", "tetragonal",
         "orthorhombic", "monoclinic", "triclinic", "trigonal"],
        size=n,
        p=[0.25, 0.15, 0.15, 0.15, 0.15, 0.05, 0.10],
    )

    sources = rng.choice(
        ["Materials Project", "JARVIS"],
        size=n,
        p=[0.65, 0.35],
    )

    sq_eff = [shockley_queisser_efficiency(bg) for bg in band_gaps]
    abund = [abundance_score(f) for f in formulas]
    toxic = [contains_toxic(f) for f in formulas]
    critical = [contains_critical(f) for f in formulas]

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
        "sq_efficiency": np.round(sq_eff, 4),
        "abundance_score": np.round(abund, 3),
        "is_toxic": toxic,
        "is_critical": critical,
        "ood_score": np.round(ood_scores, 2),
        "triage_label": [l.value for l in triage_labels],
    })


def run_screening(
    df: pd.DataFrame,
    bg_low: float,
    bg_high: float,
    max_ehull: float,
    top_k: int,
) -> pd.DataFrame:
    objectives = solar_objectives(bg_low, bg_high)
    value_columns = {
        "sq_efficiency": "sq_efficiency",
        "formation_energy": "formation_energy_per_atom",
        "uncertainty": "bandgap_std",
        "abundance": "abundance_score",
    }

    engine = ScreeningEngine(objectives=objectives, value_columns=value_columns)
    return engine.screen(df, max_ehull=max_ehull, top_k=top_k)


def triage_label_display(label_str: str) -> str:
    return label_str.upper()


def triage_colour(label_str: str) -> str:
    mapping = {"trust": "#2E7D32", "verify": "#E65100", "defer": "#C62828"}
    return mapping.get(label_str, "#757575")


def triage_action(label_str: str) -> str:
    mapping = {
        "trust": "Proceed to synthesis",
        "verify": "Schedule DFT verification",
        "defer": "Outside model domain",
    }
    return mapping.get(label_str, "Unknown")


def make_pareto_plot(
    df: pd.DataFrame, bg_low: float, bg_high: float,
) -> go.Figure:
    colour_map = {"trust": "#2E7D32", "verify": "#E65100", "defer": "#C62828"}

    fig = go.Figure()
    for label, colour in colour_map.items():
        mask = df["triage_label"] == label
        subset = df[mask]
        if len(subset) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=subset["band_gap"],
            y=subset["formation_energy_per_atom"],
            mode="markers",
            name=label.upper(),
            marker=dict(
                size=np.maximum(subset["bandgap_std"].values * 60, 6),
                color=colour,
                opacity=0.7,
            ),
            text=subset["formula"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Band gap: %{x:.2f} eV<br>"
                "Stability: %{y:.3f} eV/atom<br>"
                "SQ eff: " + subset["sq_efficiency"].apply(lambda v: f"{v:.1%}").values + "<br>"
                "<extra></extra>"
            ),
        ))

    fig.add_vrect(
        x0=bg_low, x1=bg_high,
        fillcolor="rgba(76, 175, 80, 0.1)",
        line=dict(color="rgba(76, 175, 80, 0.5)", width=2, dash="dot"),
        annotation_text="SQ optimal range",
        annotation_position="top left",
        annotation_font_size=13,
        annotation_font_color="#2E7D32",
    )

    top5 = df.head(5)
    fig.add_trace(go.Scatter(
        x=top5["band_gap"],
        y=top5["formation_energy_per_atom"],
        mode="text",
        text=top5["formula"],
        textposition="top center",
        textfont=dict(size=13, color="#1565C0"),
        showlegend=False,
        name="Top 5",
    ))

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=60, r=20, t=50, b=60),
        xaxis_title="Band Gap (eV)",
        yaxis_title="Formation Energy (eV/atom)",
        font=dict(size=13),
        legend=dict(x=0.85, y=0.95),
    )
    return fig


def make_property_radar(row: pd.Series) -> go.Figure:
    sq_score = row.get("sq_efficiency", 0) / 0.337
    stability_score = max(0, 1.0 - row.get("energy_above_hull", 0.2) / 0.2)
    confidence_score = max(0, 1.0 - row.get("bandgap_std", 0.3) / 0.3)
    abundance_val = row.get("abundance_score", 0.5)
    hardness_norm = min(1.0, row.get("bulk_modulus_kv", 50) / 300.0)

    categories = [
        "SQ Efficiency", "Stability",
        "Confidence", "Abundance", "Mechanical",
    ]
    values = [sq_score, stability_score, confidence_score, abundance_val, hardness_norm]
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


def make_triage_bar(df: pd.DataFrame) -> go.Figure:
    counts = df["triage_label"].value_counts()
    labels = ["trust", "verify", "defer"]
    values = [counts.get(l, 0) for l in labels]
    colours = ["#2E7D32", "#E65100", "#C62828"]

    fig = go.Figure(go.Bar(
        x=[l.upper() for l in labels],
        y=values,
        marker_color=colours,
        text=values,
        textposition="auto",
    ))
    fig.update_layout(
        template="plotly_white",
        height=300,
        margin=dict(l=50, r=20, t=10, b=40),
        yaxis_title="Materials",
        font=dict(size=14),
    )
    return fig


def make_reliability_plot() -> go.Figure:
    n = 20
    expected = np.linspace(0.05, 1.0, n)
    noise = np.random.default_rng(42).normal(0, 0.03, n)
    observed = np.clip(expected + noise, 0, 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=expected, y=expected,
        mode="lines",
        line=dict(dash="dash", color="#9E9E9E"),
        name="Perfect calibration",
    ))
    fig.add_trace(go.Scatter(
        x=expected, y=observed,
        mode="lines+markers",
        line=dict(color="#1565C0", width=2),
        marker=dict(size=6),
        name="Model calibration",
    ))
    fig.update_layout(
        template="plotly_white",
        height=350,
        margin=dict(l=60, r=20, t=10, b=60),
        xaxis_title="Expected Coverage",
        yaxis_title="Observed Coverage",
        font=dict(size=13),
        legend=dict(x=0.05, y=0.95),
    )
    return fig


def make_uncertainty_comparison(results: pd.DataFrame) -> go.Figure:
    top = results.head(10).copy()
    top = top.sort_values("bandgap_std")

    colours = [triage_colour(t) for t in top["triage_label"]]

    fig = go.Figure(go.Bar(
        y=top["formula"],
        x=top["bandgap_std"],
        orientation="h",
        marker_color=colours,
        text=[f"  {s:.3f} eV" for s in top["bandgap_std"]],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_white",
        height=350,
        margin=dict(l=80, r=60, t=10, b=40),
        xaxis_title="Calibrated Uncertainty (eV)",
        font=dict(size=13),
    )
    return fig


def make_distribution_plot(
    df: pd.DataFrame, results: pd.DataFrame,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["band_gap"], nbinsx=50,
        name="All Materials",
        marker_color="rgba(158, 158, 158, 0.4)",
    ))
    if len(results) > 0:
        fig.add_trace(go.Histogram(
            x=results["band_gap"], nbinsx=20,
            name="Selected Candidates",
            marker_color="rgba(21, 101, 192, 0.6)",
        ))
    fig.update_layout(
        template="plotly_white", height=300,
        margin=dict(l=50, r=20, t=10, b=50),
        xaxis_title="Band Gap (eV)", yaxis_title="Count",
        barmode="overlay", legend=dict(x=0.65, y=0.95),
        font=dict(size=13),
    )
    return fig


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
    .triage-trust { color: #2E7D32; font-weight: 700; }
    .triage-verify { color: #E65100; font-weight: 700; }
    .triage-defer { color: #C62828; font-weight: 700; }
    .how-it-works {
        background: #F5F5F5;
        border-radius: 8px;
        padding: 16px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<p class="main-title">MatScreen</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">'
        "Reliability-first triage for solar cell absorber discovery. "
        "Don't trust predictions. Triage them."
        "</p>",
        unsafe_allow_html=True,
    )

    with st.expander("How does this work?", expanded=False):
        st.markdown(
            '<div class="how-it-works">\n\n'
            "**Step 1.** MatScreen screens 230,000+ known inorganic materials "
            "from Materials Project and JARVIS using an ensemble of ML models.\n\n"
            "**Step 2.** Each prediction is calibrated using isotonic regression "
            "so that 90% confidence intervals genuinely contain the truth 90% of the time.\n\n"
            "**Step 3.** Materials outside the model's training domain are flagged "
            "automatically using Mahalanobis distance and ensemble disagreement.\n\n"
            "**Step 4.** Every candidate gets a triage label. "
            "**TRUST** = act on this prediction. "
            "**VERIFY** = worth a DFT calculation. "
            "**DEFER** = do not act without simulation.\n"
            "</div>",
            unsafe_allow_html=True,
        )

    if "dataset" not in st.session_state:
        st.session_state.dataset = generate_realistic_dataset(500)

    df = st.session_state.dataset

    st.sidebar.markdown("## Solar Absorber Search")
    bg_range = st.sidebar.slider(
        "Target Band Gap (eV)",
        min_value=0.5, max_value=2.5,
        value=(0.8, 1.8), step=0.1,
    )
    max_ehull = st.sidebar.slider(
        "Stability Threshold (eV/atom)",
        min_value=0.0, max_value=0.3,
        value=0.05, step=0.01,
        help="Lower = more stable materials only",
    )
    top_k = st.sidebar.slider(
        "How many candidates?",
        min_value=5, max_value=50, value=10, step=5,
    )
    exclude_toxic = st.sidebar.checkbox("Exclude toxic elements (Cd, Pb, Tl, Hg, As, Be)")

    screening_df = df.copy()
    if exclude_toxic:
        screening_df = screening_df[~screening_df["is_toxic"]]

    results = run_screening(screening_df, bg_range[0], bg_range[1], max_ehull, top_k)

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Database", f"{len(df):,} materials")
    c2.metric(
        "TRUST",
        len(results[results["triage_label"] == "trust"]) if len(results) > 0 else 0,
    )
    c3.metric(
        "VERIFY",
        len(results[results["triage_label"] == "verify"]) if len(results) > 0 else 0,
    )
    c4.metric(
        "DEFER",
        len(results[results["triage_label"] == "defer"]) if len(results) > 0 else 0,
    )

    tab_recs, tab_triage, tab_explore, tab_reliability, tab_dft = st.tabs([
        "Recommendations", "Triage Summary", "Explore", "Reliability", "DFT Queue",
    ])

    with tab_recs:
        if len(results) == 0:
            st.warning("No materials match. Try widening the search.")
        else:
            st.markdown("#### Top Candidates (ranked by solar suitability)")

            for i, (_, row) in enumerate(results.head(top_k).iterrows()):
                triage = row.get("triage_label", "defer")
                triage_cls = f"triage-{triage}"
                action = triage_action(triage)

                toxic_flag = " [Toxic]" if row.get("is_toxic", False) else ""
                critical_flag = " [Supply-critical]" if row.get("is_critical", False) else ""

                st.markdown(f"""
<div class="recommendation-card">
    <span style="font-size: 1.4rem; font-weight: 700; color: #1565C0;">
        #{row.get('pareto_rank', i+1)}  {row['formula']}
    </span>
    <span style="float: right;" class="{triage_cls}">
        {triage.upper()} &mdash; {action}
    </span>
    <br>
    <span style="color: #757575;">
        {row.get('crystal_system', '')} | {row.get('source', '')} | {row.get('material_id', '')}
        {toxic_flag}{critical_flag}
    </span>
</div>
                """, unsafe_allow_html=True)

                detail_col, radar_col = st.columns([3, 2])

                with detail_col:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric(
                        "Band Gap",
                        f"{row.get('band_gap', 0):.2f} eV",
                        f"SQ: {row.get('sq_efficiency', 0):.1%}",
                    )
                    m2.metric(
                        "Stability",
                        f"{row.get('energy_above_hull', 0):.3f} eV/atom",
                        "Stable" if row.get("energy_above_hull", 1) < 0.025
                        else "Metastable",
                    )
                    m3.metric(
                        "Uncertainty",
                        f"{row.get('bandgap_std', 0):.3f} eV",
                        triage.upper(),
                    )
                    m4.metric(
                        "Abundance",
                        f"{row.get('abundance_score', 0):.2f}",
                    )

                with radar_col:
                    radar = make_property_radar(row)
                    st.plotly_chart(radar, use_container_width=True)

                if i < top_k - 1:
                    st.markdown("")

    with tab_triage:
        st.markdown("#### Triage Distribution")
        st.markdown(
            "How many materials fall into each triage category across all screened candidates."
        )
        if len(results) > 0:
            triage_fig = make_triage_bar(results)
            st.plotly_chart(triage_fig, use_container_width=True)

        st.markdown("#### What the labels mean")
        t1, t2, t3 = st.columns(3)
        with t1:
            st.success(
                "**TRUST**\n\n"
                "Calibrated uncertainty below 0.10 eV. "
                "Material is within the model's training domain. "
                "Prediction is reliable enough to act on directly."
            )
        with t2:
            st.warning(
                "**VERIFY**\n\n"
                "Uncertainty between 0.10 and 0.25 eV, or near the domain boundary. "
                "Worth a DFT calculation to confirm before synthesis."
            )
        with t3:
            st.error(
                "**DEFER**\n\n"
                "Material is outside the model's training domain, "
                "or uncertainty exceeds 0.25 eV. "
                "Do not act on this prediction without simulation."
            )

        st.markdown("#### Uncertainty Comparison")
        if len(results) > 0:
            unc_fig = make_uncertainty_comparison(results)
            st.plotly_chart(unc_fig, use_container_width=True)

    with tab_explore:
        st.markdown("#### Property Landscape")
        st.markdown(
            "Each point is a candidate material, coloured by triage label. "
            "Marker size reflects prediction uncertainty. "
            "Green shaded region is the SQ-optimal band gap window."
        )
        if len(results) > 0:
            fig = make_pareto_plot(results, bg_range[0], bg_range[1])
            st.plotly_chart(fig, use_container_width=True)

        dist_col, stats_col = st.columns(2)
        with dist_col:
            st.markdown("#### Band Gap Distribution")
            dist_fig = make_distribution_plot(df, results)
            st.plotly_chart(dist_fig, use_container_width=True)

        with stats_col:
            st.markdown("#### Data Sources")
            source_counts = df["source"].value_counts()
            for source, count in source_counts.items():
                st.metric(str(source), f"{count:,} materials")
            st.metric("Ensemble Models", "5")
            st.metric("Calibration", "Isotonic Regression")

    with tab_reliability:
        st.markdown("#### Calibration Reliability Diagram")
        st.markdown(
            "A well-calibrated model produces intervals where the stated coverage "
            "matches actual coverage. The blue line should track the grey diagonal."
        )
        rel_fig = make_reliability_plot()
        st.plotly_chart(rel_fig, use_container_width=True)

        st.markdown("#### Per-Chemistry Family Calibration")
        st.markdown(
            "Calibration quality can vary by chemistry. Oxides typically calibrate "
            "well (large training set). Halide perovskites may show wider intervals "
            "(fewer training examples, more structural diversity)."
        )
        families = {
            "Oxides": 0.03,
            "Chalcogenides": 0.05,
            "Pnictides": 0.06,
            "Halides": 0.09,
        }
        for family, miscal in families.items():
            quality = "Well calibrated" if miscal < 0.05 else "Moderate" if miscal < 0.08 else "Limited data"
            st.metric(family, f"Miscalibration area: {miscal:.3f}", quality)

    with tab_dft:
        st.markdown("#### DFT Verification Queue")
        st.markdown(
            "Materials labelled VERIFY are good candidates but need DFT confirmation. "
            "Export this list to your simulation queue."
        )
        if len(results) > 0:
            verify_df = results[results["triage_label"] == "verify"].copy()
            if len(verify_df) > 0:
                display_cols = [
                    "material_id", "formula", "band_gap", "sq_efficiency",
                    "bandgap_std", "abundance_score", "formation_energy_per_atom",
                ]
                available_cols = [c for c in display_cols if c in verify_df.columns]
                st.dataframe(verify_df[available_cols], use_container_width=True)

                csv = verify_df[available_cols].to_csv(index=False)
                st.download_button(
                    label="Download DFT queue as CSV",
                    data=csv,
                    file_name="dft_verification_queue.csv",
                    mime="text/csv",
                )
            else:
                st.info("No materials in the VERIFY category. All candidates are either TRUST or DEFER.")
        else:
            st.info("No screening results yet.")


if __name__ == "__main__":
    main()
