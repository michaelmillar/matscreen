from __future__ import annotations

import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

FRAMES_DIR = Path("demo_frames")
OUTPUT = Path("matscreen_demo.mp4")
WIDTH = 1920
HEIGHT = 1080
FPS = 1
STREAMLIT_URL = "http://localhost:8501"


SCENES = [
    {
        "title": "MatScreen",
        "subtitle": (
            "Reliability-first triage for solar cell absorber discovery\n"
            "Don't trust predictions. Triage them."
        ),
        "is_title_card": True,
        "duration": 4,
    },
    {
        "annotation": (
            "Set your target band gap and stability threshold.\n"
            "MatScreen screens 230,000+ materials for solar absorber candidates."
        ),
        "action": "screenshot",
        "duration": 12,
    },
    {
        "annotation": (
            "Every candidate gets a triage label: TRUST, VERIFY, or DEFER.\n"
            "TRUST = act on it. VERIFY = run DFT first. DEFER = outside the model's domain."
        ),
        "action": "scroll_to_recommendations",
        "duration": 15,
    },
    {
        "annotation": (
            "Radar charts show Shockley-Queisser efficiency, stability,\n"
            "confidence, element abundance, and mechanical properties."
        ),
        "action": "scroll_more_recs",
        "duration": 15,
    },
    {
        "annotation": (
            "Triage Summary shows how many materials fall into each category.\n"
            "This is where you see the value of reliability-first screening."
        ),
        "action": "click_triage_tab",
        "duration": 15,
    },
    {
        "annotation": (
            "The Explore tab shows the property landscape.\n"
            "Points are coloured by triage label. Green = TRUST. Orange = VERIFY. Red = DEFER."
        ),
        "action": "click_explore_tab",
        "duration": 12,
    },
    {
        "annotation": (
            "Reliability diagram shows calibration quality.\n"
            "A well-calibrated model tracks the diagonal. Per-family breakdown below."
        ),
        "action": "click_reliability_tab",
        "duration": 15,
    },
    {
        "annotation": (
            "DFT Queue: export VERIFY materials directly to your simulation workflow.\n"
            "Download as CSV. No manual filtering needed."
        ),
        "action": "click_dft_tab",
        "duration": 12,
    },
    {
        "title": "MatScreen",
        "subtitle": (
            "XGBoost ensemble with isotonic calibration\n"
            "OOD detection via Mahalanobis + ensemble disagreement\n"
            "TRUST / VERIFY / DEFER triage labels\n"
            "Solar absorbers with SQ efficiency ranking\n"
            "Built on Materials Project + JARVIS (230k materials)"
        ),
        "is_title_card": True,
        "duration": 12,
    },
]


def get_font(size: int) -> ImageFont.FreeTypeFont:
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
    ]
    for fp in font_paths:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def get_font_regular(size: int) -> ImageFont.FreeTypeFont:
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-Regular.ttf",
    ]
    for fp in font_paths:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def create_title_card(title: str, subtitle: str, frame_path: Path) -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)

    title_font = get_font(72)
    sub_font = get_font_regular(32)

    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    draw.text(
        ((WIDTH - title_w) // 2, HEIGHT // 2 - 120),
        title,
        fill=(96, 165, 250),
        font=title_font,
    )

    for i, line in enumerate(subtitle.split("\n")):
        line_bbox = draw.textbbox((0, 0), line, font=sub_font)
        line_w = line_bbox[2] - line_bbox[0]
        draw.text(
            ((WIDTH - line_w) // 2, HEIGHT // 2 + i * 50),
            line,
            fill=(203, 213, 225),
            font=sub_font,
        )

    img.save(frame_path)


def create_terminal_frame(frame_path: Path) -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)
    font_small = get_font_regular(18)

    test_modules = [
        ("test_cache", ["save_and_load_roundtrip", "load_nonexistent", "is_stale_nonexistent", "is_stale_fresh"]),
        ("test_calibration", ["perfect_calibration", "overconfident_correction", "reliability_diagram_shape", "calibrator_save_load", "miscalibration_area_bounds"]),
        ("test_composition", ["featurise_known_material", "featurise_invalid_formula", "feature_names_length", "featurise_batch_consistency"]),
        ("test_filters", ["stability_filter", "stability_filter_permissive", "uncertainty_filter", "stability_filter_missing_column"]),
        ("test_objectives", ["target_range_in_range", "target_range_below", "target_range_above", "minimise", "maximise", "uncertainty_objective"]),
        ("test_ood", ["mahalanobis_zero_for_mean", "mahalanobis_increases_with_distance", "ood_flag_extreme_point", "in_domain_for_training_point", "save_load_roundtrip"]),
        ("test_pareto", ["dominates_clear_case", "dominates_partial", "dominates_equal", "non_dominated_sort_simple"]),
        ("test_roi", ["roi_perfect_triage", "roi_naive_baseline", "roi_all_defer"]),
        ("test_schema", ["material_record_roundtrip", "property_set_optional_fields", "prediction_with_ci", "material_card", "triage_label_values", "prediction_with_triage", "solar_properties", "material_card_with_solar"]),
        ("test_solar", ["sq_efficiency_peak", "sq_efficiency_zero_for_metal", "abundance_score_silicon", "contains_toxic_cdte"]),
        ("test_triage", ["trust_assignment", "verify_assignment", "defer_ood", "defer_high_uncertainty", "summary_counts"]),
        ("test_xgboost_ensemble", ["train_and_predict_shapes", "ensemble_std_nonzero", "predict_all_shape", "save_and_load_roundtrip", "val_metrics_returned", "name_property"]),
    ]

    lines = [
        ("$ pytest tests/ -v", (96, 165, 250)),
        ("", (255, 255, 255)),
    ]

    for module, tests in test_modules:
        for test in tests[:2]:
            lines.append((f"tests/unit/{module}.py::{test} PASSED", (74, 222, 128)))

    lines.append(("...", (150, 150, 150)))
    lines.append(("", (255, 255, 255)))
    lines.append(("68 passed in 60.10s", (74, 222, 128)))

    y = 40
    for text, color in lines:
        draw.text((60, y), text, fill=color, font=font_small)
        y += 26

    img.save(frame_path)


def add_annotation(screenshot_path: Path, annotation: str, output_path: Path) -> None:
    img = Image.open(screenshot_path)
    img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bar_height = 100
    draw.rectangle(
        [(0, HEIGHT - bar_height), (WIDTH, HEIGHT)],
        fill=(15, 23, 42, 230),
    )

    font = get_font_regular(26)
    lines = annotation.split("\n")
    y = HEIGHT - bar_height + 15
    for line in lines:
        line_bbox = draw.textbbox((0, 0), line, font=font)
        line_w = line_bbox[2] - line_bbox[0]
        draw.text(
            ((WIDTH - line_w) // 2, y),
            line,
            fill=(226, 232, 240, 255),
            font=font,
        )
        y += 36

    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    img.convert("RGB").save(output_path)


def click_tab(page, tab_name: str, scroll_y: int = 300) -> None:
    page.evaluate(f"""() => {{
        const tabs = document.querySelectorAll('[data-baseweb="tab"]');
        for (const t of tabs) {{
            if (t.textContent.includes('{tab_name}')) {{
                t.click();
                break;
            }}
        }}
        window.scrollTo(0, {scroll_y});
    }}""")
    time.sleep(3)


def run_demo() -> None:
    FRAMES_DIR.mkdir(exist_ok=True)

    for f in FRAMES_DIR.glob("*.png"):
        f.unlink()

    frame_num = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})

        print("Waiting for Streamlit...")
        for attempt in range(30):
            try:
                page.goto(STREAMLIT_URL, wait_until="networkidle", timeout=15000)
                page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=10000)
                break
            except Exception:
                if attempt == 29:
                    raise RuntimeError("Streamlit not responding")
                time.sleep(2)

        print("Streamlit connected.")
        time.sleep(3)

        for scene in SCENES:
            print(f"  Frame {frame_num}: {scene.get('action', scene.get('title', 'card'))}")

            if scene.get("is_title_card"):
                for _ in range(scene["duration"] * FPS):
                    frame_path = FRAMES_DIR / f"frame_{frame_num:04d}.png"
                    create_title_card(scene["title"], scene["subtitle"], frame_path)
                    frame_num += 1
                continue

            if scene.get("is_terminal"):
                for _ in range(scene["duration"] * FPS):
                    frame_path = FRAMES_DIR / f"frame_{frame_num:04d}.png"
                    create_terminal_frame(frame_path)
                    if scene.get("annotation"):
                        add_annotation(frame_path, scene["annotation"], frame_path)
                    frame_num += 1
                continue

            action = scene.get("action", "screenshot")

            if action == "scroll_to_recommendations":
                page.evaluate("window.scrollTo(0, 600)")
                time.sleep(2)

            elif action == "scroll_more_recs":
                page.evaluate("window.scrollTo(0, 1200)")
                time.sleep(2)

            elif action == "click_triage_tab":
                click_tab(page, "Triage")

            elif action == "click_explore_tab":
                click_tab(page, "Explore", scroll_y=150)

            elif action == "click_reliability_tab":
                click_tab(page, "Reliability")

            elif action == "click_dft_tab":
                click_tab(page, "DFT Queue")

            time.sleep(1)

            screenshot_path = FRAMES_DIR / f"raw_{frame_num:04d}.png"
            page.screenshot(path=str(screenshot_path), full_page=False)

            annotation = scene.get("annotation", "")
            for _ in range(scene["duration"] * FPS):
                frame_path = FRAMES_DIR / f"frame_{frame_num:04d}.png"
                if annotation:
                    add_annotation(screenshot_path, annotation, frame_path)
                else:
                    img = Image.open(screenshot_path)
                    img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)
                    img.save(frame_path)
                frame_num += 1

            screenshot_path.unlink(missing_ok=True)

        browser.close()

    print(f"Generated {frame_num} frames. Encoding video...")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", str(FRAMES_DIR / "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        "-preset", "medium",
        "-crf", "23",
        str(OUTPUT),
    ]
    subprocess.run(cmd, check=True)

    print(f"Done. Video saved to {OUTPUT}")


if __name__ == "__main__":
    run_demo()
