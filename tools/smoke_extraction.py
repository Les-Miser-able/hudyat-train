"""Read a few existing collector clips into an isolated extraction smoke workspace."""
from pathlib import Path
import argparse
import csv
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hudyat_train.config import ROOT, load_settings
from hudyat_train.extraction import extract
from hudyat_train.storage import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=ROOT.parent / "hudyat-track")
    parser.add_argument("--work", type=Path, default=ROOT / "tmp" / "real-extraction")
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    write_json(work / "project.json", read_json(ROOT / "project.json"))
    shutil.copyfile(ROOT / "signs.csv", work / "signs.csv")
    (work / "models").mkdir(exist_ok=True)
    for name in ("hand_landmarker.task", "face_landmarker.task"):
        shutil.copyfile(ROOT / "models" / name, work / "models" / name)
    with (args.track / "dataset" / "metadata.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    settings = load_settings(work)
    results = []
    for category, sign in [("alphabet", "letter_a"), ("family", "father")]:
        row = next(r for r in rows if r["category"] == category and r["sign_id"] == sign and r["status"] == "kept")
        source = args.track / row["file_path"]
        target = settings.path("raw_dir") / category / sign / row["signer"] / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        for variant in ("sequence", "static"):
            result = extract(settings, category, variant)
            results.append({"category": category, "variant": variant, "items": result["items"]})
            if result["items"][0]["status"] != "accepted":
                raise RuntimeError(f"Real clip was rejected: {result['items'][0]}")
        # Test the IMAGE detector path using a clean held-pose frame from the static extraction.
        import cv2
        pose = result["items"][0]["quality"]["pose_frame"]
        cap = cv2.VideoCapture(str(target))
        cap.set(cv2.CAP_PROP_POS_FRAMES, pose)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Could not decode the static pose frame for the still-image test.")
        still = target.with_name("image-smoke.jpg")
        if not cv2.imwrite(str(still), frame):
            raise RuntimeError("Could not write the isolated still-image fixture.")
        result = extract(settings, category, "static")
        image = next(it for it in result["items"] if it["kind"] == "image")
        if category == "alphabet" and image["status"] != "accepted":
            raise RuntimeError(f"Still-image extraction failed: {image}")
        results.append({"category": category, "variant": "static-image", "items": [image]})
    write_json(work / "smoke-results.json", {"purpose": "extraction mechanics only; not study evaluation", "results": results})
    print("Real video and still-image extraction passed:", work)


if __name__ == "__main__":
    main()
