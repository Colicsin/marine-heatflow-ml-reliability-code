from pathlib import Path

from PIL import Image, ImageEnhance


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"

SRC = FIG_DIR / "step11_global_pred_map.png"
DST = FIG_DIR / "global_heatflow_prediction.png"
BACKUP = FIG_DIR / "global_heatflow_prediction_old.png"
ALT = FIG_DIR / "global_heatflow_prediction_paper.png"


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Source figure not found: {SRC}")

    # Keep the old figure for comparison if it exists.
    if DST.exists() and not BACKUP.exists():
        DST.replace(BACKUP)

    img = Image.open(SRC).convert("RGB")

    # Small paper-oriented polish: slightly stronger contrast and sharpness
    # so the global heat-flow gradients read more clearly after scaling in Word.
    img = ImageEnhance.Contrast(img).enhance(1.06)
    img = ImageEnhance.Sharpness(img).enhance(1.08)

    img.save(DST, quality=95)
    img.save(ALT, quality=95)

    print(f"Saved main figure to: {DST}")
    print(f"Saved paper copy to: {ALT}")
    if BACKUP.exists():
        print(f"Old figure kept at: {BACKUP}")


if __name__ == "__main__":
    main()
