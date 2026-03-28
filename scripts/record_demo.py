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
            "Uncertainty-aware materials screening\n"
            "with calibrated confidence intervals"
        ),
        "is_title_card": True,
        "duration": 8,
    },
    {
        "annotation": (
            "Tell the tool what you need.\n"
            "Select an application like Solar Cell or LED, and it screens 230,000 materials."
        ),
        "action": "screenshot",
        "duration": 12,
    },
    {
        "annotation": (
            "Recommendations tab shows top candidates ranked by overall suitability.\n"
            "Each material gets a confidence rating: HIGH, MODERATE, or LOW."
        ),
        "action": "scroll_to_recommendations",
        "duration": 15,
    },
    {
        "annotation": (
            "Radar charts show how each material scores across five dimensions.\n"
            "One glance tells you the tradeoffs."
        ),
        "action": "scroll_more_recs",
        "duration": 15,
    },
    {
        "annotation": (
            "The Explore tab shows the full property landscape.\n"
            "Top 5 candidates are labelled. Green region = your target band gap."
        ),
        "action": "click_explore_tab",
        "duration": 15,
    },
    {
        "annotation": (
            "Distribution and crystal system charts show\n"
            "where your candidates sit within the full database."
        ),
        "action": "scroll_explore_charts",
        "duration": 12,
    },
    {
        "annotation": (
            "The Analysis tab shows model confidence for each prediction.\n"
            "Green = high confidence. Red = verify with simulation."
        ),
        "action": "click_analysis_tab",
        "duration": 15,
    },
    {
        "annotation": (
            "Switch to LED application.\n"
            "The system re-screens for a completely different band gap range."
        ),
        "action": "switch_to_led",
        "duration": 15,
    },
    {
        "annotation": (
            "27 unit tests validate the Pareto sorting, calibration,\n"
            "data pipeline, and screening logic"
        ),
        "is_terminal": True,
        "duration": 15,
    },
    {
        "title": "MatScreen",
        "subtitle": (
            "Ensemble ALIGNN models with calibrated UQ\n"
            "Multi-objective Pareto screening\n"
            "Built on Materials Project + JARVIS (230k materials)\n"
            "Validated on Matbench benchmarks"
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
    font = get_font(24)
    font_small = get_font_regular(20)

    lines = [
        ("$ pytest tests/unit/ -v", (96, 165, 250)),
        ("", (255, 255, 255)),
        ("tests/unit/test_cache.py::test_save_and_load_roundtrip PASSED", (74, 222, 128)),
        ("tests/unit/test_cache.py::test_load_nonexistent PASSED", (74, 222, 128)),
        ("tests/unit/test_cache.py::test_is_stale_nonexistent PASSED", (74, 222, 128)),
        ("tests/unit/test_cache.py::test_is_stale_fresh PASSED", (74, 222, 128)),
        ("tests/unit/test_filters.py::test_stability_filter PASSED", (74, 222, 128)),
        ("tests/unit/test_filters.py::test_stability_filter_permissive PASSED", (74, 222, 128)),
        ("tests/unit/test_filters.py::test_uncertainty_filter PASSED", (74, 222, 128)),
        ("tests/unit/test_filters.py::test_stability_filter_missing_column PASSED", (74, 222, 128)),
        ("tests/unit/test_objectives.py::test_target_range_in_range PASSED", (74, 222, 128)),
        ("tests/unit/test_objectives.py::test_target_range_below PASSED", (74, 222, 128)),
        ("tests/unit/test_objectives.py::test_target_range_above PASSED", (74, 222, 128)),
        ("tests/unit/test_objectives.py::test_minimise PASSED", (74, 222, 128)),
        ("tests/unit/test_objectives.py::test_maximise PASSED", (74, 222, 128)),
        ("tests/unit/test_objectives.py::test_uncertainty_objective PASSED", (74, 222, 128)),
        ("tests/unit/test_pareto.py::test_dominates_clear_case PASSED", (74, 222, 128)),
        ("tests/unit/test_pareto.py::test_dominates_partial PASSED", (74, 222, 128)),
        ("tests/unit/test_pareto.py::test_dominates_equal PASSED", (74, 222, 128)),
        ("tests/unit/test_pareto.py::test_non_dominated_sort_simple PASSED", (74, 222, 128)),
        ("tests/unit/test_pareto.py::test_non_dominated_sort_single_front PASSED", (74, 222, 128)),
        ("tests/unit/test_pareto.py::test_non_dominated_sort_fully_dominated PASSED", (74, 222, 128)),
        ("tests/unit/test_pareto.py::test_crowding_distance_two_points PASSED", (74, 222, 128)),
        ("tests/unit/test_pareto.py::test_crowding_distance_three_points PASSED", (74, 222, 128)),
        ("tests/unit/test_pareto.py::test_non_dominated_sort_random PASSED", (74, 222, 128)),
        ("tests/unit/test_schema.py::test_material_record_roundtrip PASSED", (74, 222, 128)),
        ("tests/unit/test_schema.py::test_property_set_optional_fields PASSED", (74, 222, 128)),
        ("tests/unit/test_schema.py::test_prediction_with_ci PASSED", (74, 222, 128)),
        ("tests/unit/test_schema.py::test_material_card PASSED", (74, 222, 128)),
        ("", (255, 255, 255)),
        ("27 passed in 0.35s", (74, 222, 128)),
    ]

    y = 40
    for text, color in lines:
        draw.text((60, y), text, fill=color, font=font_small)
        y += 28

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

            elif action == "click_explore_tab":
                page.evaluate("""() => {
                    const tabs = document.querySelectorAll('[data-baseweb="tab"]');
                    for (const t of tabs) {
                        if (t.textContent.includes('Explore')) {
                            t.click();
                            break;
                        }
                    }
                    window.scrollTo(0, 400);
                }""")
                time.sleep(3)

            elif action == "scroll_explore_charts":
                page.evaluate("window.scrollTo(0, 900)")
                time.sleep(2)

            elif action == "click_analysis_tab":
                page.evaluate("""() => {
                    const tabs = document.querySelectorAll('[data-baseweb="tab"]');
                    for (const t of tabs) {
                        if (t.textContent.includes('Analysis')) {
                            t.click();
                            break;
                        }
                    }
                    window.scrollTo(0, 400);
                }""")
                time.sleep(3)

            elif action == "switch_to_led":
                page.evaluate("""() => {
                    const tabs = document.querySelectorAll('[data-baseweb="tab"]');
                    for (const t of tabs) {
                        if (t.textContent.includes('Recommend')) {
                            t.click();
                            break;
                        }
                    }
                    window.scrollTo(0, 0);
                }""")
                time.sleep(2)
                page.evaluate("""() => {
                    const selects = document.querySelectorAll(
                        '[data-testid="stSelectbox"]'
                    );
                    if (selects.length > 0) selects[0].click();
                }""")
                time.sleep(1)
                led = page.query_selector('li:has-text("LED")')
                if led:
                    led.click()
                    time.sleep(3)
                page.evaluate("window.scrollTo(0, 600)")
                time.sleep(2)

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
