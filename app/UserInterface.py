import csv
import glob
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import rasterio, cv2, matplotlib.pyplot as plt
from dotenv import load_dotenv

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QLineEdit, QMessageBox, QDateEdit, QFileDialog
)
from PyQt5.QtCore import Qt, QDate, QObject, pyqtSlot, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel

from torchvision import models

# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Sentinel Hub (online fetch)
from sentinelhub import (
    SentinelHubRequest, MimeType, BBox, CRS, bbox_to_dimensions,
    SentinelHubCatalog, SHConfig, DataCollection
)

EVALSCRIPT = """
//VERSION=3
function setup(){return{input:["B8A","B11"],output:{bands:2,sampleType:"FLOAT32"}};}
function evaluatePixel(s){return[s.B8A,s.B11];}
"""

# Model setup (exactly as in training)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3
IMG_SIZE = 224

configured_model_path = Path(
    os.getenv("SOIL_MODEL_PATH", PROJECT_ROOT / "models" / "classification_3km.pth")
).expanduser()
MODEL_WEIGHTS_PATH = (
    configured_model_path
    if configured_model_path.is_absolute()
    else PROJECT_ROOT / configured_model_path
)

def build_model():
    # The saved state dictionary contains the complete trained weights, so no
    # network download of ImageNet weights is required at application runtime.
    m = models.efficientnet_v2_s(weights=None)

    # First conv: 2 input channels, copy weights from first two ImageNet channels
    orig_conv = m.features[0][0]
    m.features[0][0] = nn.Conv2d(
        in_channels=2,
        out_channels=orig_conv.out_channels,
        kernel_size=orig_conv.kernel_size,
        stride=orig_conv.stride,
        padding=orig_conv.padding,
        bias=orig_conv.bias is not None
    )
    with torch.no_grad():
        m.features[0][0].weight[:] = orig_conv.weight[:, :2]

    # Classifier head: Dropout(0.5) + Linear → 3 classes
    n_in = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.5, inplace=True),
        nn.Linear(n_in, NUM_CLASSES)
    )
    return m

model = None


def get_model():
    """Load the trusted state dictionary only when a prediction is requested."""
    global model
    if model is not None:
        return model
    if not MODEL_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_WEIGHTS_PATH}. "
            "Set SOIL_MODEL_PATH in .env; see models/README.md."
        )
    loaded_model = build_model()
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
    loaded_model.load_state_dict(state_dict)
    model = loaded_model.to(DEVICE).eval()
    return model

# Labels aligned with training thresholds (0–10, 10–20, >20)
IDX2LABEL = {0: "Low (0–10)", 1: "Moderate (10–20)", 2: "High (>20)"}

# ─────────── Pre-process + predict (identical to notebook) ───────────
def preprocess_tiff(path: str) -> torch.Tensor:
    """Returns (2, 224, 224) float32 tensor scaled 0–1 (per band min–max)."""
    with rasterio.open(path) as src:
        img = src.read()  # expects 2 bands: [B8A, B11]
    img = np.array([(b - b.min()) / (b.max() - b.min() + 1e-6) for b in img], dtype=np.float32)
    img = np.array([cv2.resize(b, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR) for b in img],
                   dtype=np.float32)
    return torch.from_numpy(img)

@torch.inference_mode()
def predict_image(path: str):
    x = preprocess_tiff(path).unsqueeze(0).to(DEVICE)
    logits = get_model()(x)
    probs = F.softmax(logits, dim=1).cpu().squeeze(0).numpy()
    cls = int(probs.argmax())
    return cls, probs

# Helpers: km to degrees (for bigger, sharper patches)
def km_to_deg_lat(km: float) -> float:
    return km / 110.574

def km_to_deg_lon(km: float, lat_deg: float) -> float:
    return km / (111.320 * max(0.01, math.cos(math.radians(lat_deg))))

def get_sentinel_hub_clients():
    """Create authenticated clients without storing credentials in source code."""
    client_id = os.getenv("SH_CLIENT_ID", "").strip()
    client_secret = os.getenv("SH_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError(
            "Sentinel Hub credentials are missing. Copy .env.example to .env and "
            "set SH_CLIENT_ID and SH_CLIENT_SECRET."
        )

    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    return config, SentinelHubCatalog(config=config)


# Sentinel Hub fetch (with patch size control)
def get_image_for(lat, lon, date_str, max_cloud=30, patch_km=6.0):
    """
    Fetch a ~patch_km x patch_km box at 20 m (native for B8A/B11).
    6 km is approximately 300 px per side.
    """
    config, catalog = get_sentinel_hub_clients()
    start, end = f"{date_str}T00:00:00Z", f"{date_str}T23:59:59Z"
    half_lat = km_to_deg_lat(patch_km / 2.0)
    half_lon = km_to_deg_lon(patch_km / 2.0, lat)
    bbox = BBox([lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat], CRS.WGS84)

    # expect roughly (patch_km*1000 / 20) pixels per side
    size = bbox_to_dimensions(bbox, resolution=20)

    items = list(catalog.search(
        DataCollection.SENTINEL2_L2A,
        bbox=bbox, time=(start, end),
        filter=f"eo:cloud_cover < {max_cloud}",
        fields={"include": ["properties.datetime"], "exclude": []}
    ))
    if not items:
        raise RuntimeError("No image available for that date (or too cloudy). Try another date or raise cloud limit.")

    tmpdir = tempfile.mkdtemp(prefix="earthsense_")
    req = SentinelHubRequest(
        evalscript=EVALSCRIPT, data_folder=tmpdir, config=config,
        input_data=[{"type": "sentinel-2-l2a",
                     "dataFilter": {"timeRange": {"from": start, "to": end}}}],
        responses=[{"identifier": "default", "format": {"type": MimeType.TIFF.get_string()}}],
        bbox=bbox, size=size
    )
    req.get_data(save_data=True)

    resp_dir = next(iter(os.scandir(tmpdir))).path
    merged = os.path.join(resp_dir, "response.tiff")
    if not os.path.exists(merged):
        raise RuntimeError("Download succeeded but no TIFF found.")
    return merged, tmpdir

# Visualisation (composite + MSI)
def visualise(path: str, label: str, date_text: str = ""):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    if arr.shape[0] != 2:
        raise RuntimeError(f"Expected 2 bands (B8A,B11), got shape {arr.shape}")
    b8a, b11 = arr[0], arr[1]
    composite = (b8a + b11) / 2.0
    msi = np.clip(b11 / (b8a + 1e-6), 0, 10)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(composite, cmap="gray")
    plt.title("Grayscale Composite")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    im = plt.imshow(msi, cmap="viridis")
    plt.title("Moisture Stress Index (B11/B8A)")
    plt.axis("off")
    plt.colorbar(im, fraction=.046, pad=.04, label="MSI")

    title = f"Predicted Class: {label}"
    if date_text:
        title += f" | Date: {date_text}"
    plt.suptitle(title, color="darkblue")
    plt.tight_layout()
    plt.show()

# Batch mode helpers (merge B8A/B11 by date like the notebook)
def group_bands_by_date(folder: str):
    """Return dict[date] -> {'B8A': path, 'B11': path} by parsing filename prefix before '_'."""
    all_files = glob.glob(os.path.join(folder, '**', '*.tif*'), recursive=True)
    grouped = defaultdict(dict)
    for f in all_files:
        base = os.path.basename(f)
        date_key = base.split('_')[0]
        up = base.upper()
        if 'B8A' in up:
            grouped[date_key]['B8A'] = f
        elif 'B11' in up:
            grouped[date_key]['B11'] = f
    return grouped

def merge_to_two_band_tif(b8a_path: str, b11_path: str, out_path: str):
    with rasterio.open(b8a_path) as s8:
        b8a = s8.read(1).astype(np.float32)
        profile = s8.profile
    with rasterio.open(b11_path) as s11:
        b11 = s11.read(1).astype(np.float32)
    stacked = np.stack([b8a, b11]).astype(np.float32)
    profile.update(count=2, dtype="float32")
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(stacked)

# Map popup (Leaflet + WebEngine)
class CoordBridge(QObject):
    def __init__(self, cb): super().__init__(); self.cb = cb
    @pyqtSlot(float, float)
    def sendCoordinates(self, lat, lon):
        print(f"[DEBUG] Received coords from map: {lat}, {lon}")
        self.cb(lat, lon)

class MapPopup(QWidget):
    def __init__(self, on_select):
        super().__init__()
        self.setWindowTitle("Select Location on Map")
        self.setGeometry(300, 300, 600, 500)
        self.view = QWebEngineView(self)
        self.view.setGeometry(0, 0, 600, 500)
        channel = QWebChannel()
        channel.registerObject("pyjs", CoordBridge(on_select))
        self.view.page().setWebChannel(channel)
        html_path = QUrl.fromLocalFile(str(Path(__file__).with_name("leaflet_map.html")))
        self.view.load(html_path)

# PyQt GUI
class EarthSenseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EarthSense AI - Sentinel-2 Moisture Classifier")
        self.setGeometry(200, 200, 740, 460)

        main = QVBoxLayout(); main.setSpacing(10)

        # ------- coordinate, date, patch size ---------
        row = QHBoxLayout()
        self.lat_edit = QLineEdit(); self.lat_edit.setPlaceholderText("Latitude");  self.lat_edit.setFixedWidth(160)
        self.lon_edit = QLineEdit(); self.lon_edit.setPlaceholderText("Longitude"); self.lon_edit.setFixedWidth(160)
        self.date_edit = QDateEdit(); self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd"); self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setFixedHeight(32); self.date_edit.setFixedWidth(160)

        self.patch_edit = QLineEdit(); self.patch_edit.setPlaceholderText("Patch size km"); self.patch_edit.setFixedWidth(120)
        self.patch_edit.setText("6")  # default 6 km, approximately 300 px at 20 m

        row.addWidget(self.lat_edit); row.addWidget(self.lon_edit); row.addWidget(self.date_edit); row.addWidget(self.patch_edit)
        main.addLayout(row)

        # ------- buttons row ----------------------
        btn_row = QHBoxLayout()

        self.map_btn = QPushButton("Pick from Map")
        self.map_btn.clicked.connect(self.open_map)
        btn_row.addWidget(self.map_btn)

        self.fetch_btn = QPushButton("Fetch & Predict (Sentinel Hub)")
        self.fetch_btn.clicked.connect(self.fetch_and_predict)
        btn_row.addWidget(self.fetch_btn)

        self.batch_btn = QPushButton("Batch: Folder of B8A/B11")
        self.batch_btn.clicked.connect(self.run_batch_folder)
        btn_row.addWidget(self.batch_btn)

        main.addLayout(btn_row)

        # ------- legend ---------------------------
        legend = ("Output Classes (aligned with training):\n"
                  "Low (0-10%)       - Low moisture\n"
                  "Moderate (10-20%) - Moderate moisture\n"
                  "High (>20%)       - High moisture")
        lg = QLabel(legend); lg.setStyleSheet("color:gray; font-size:10pt;")
        main.addWidget(lg)

        # ------- result label ---------------------
        self.result = QLabel(""); self.result.setStyleSheet("font-weight:bold;")
        main.addWidget(self.result)

        self.setLayout(main)

    # --- open map popup -------------------------
    def open_map(self):
        def _set_coords(lat, lon):
            self.lat_edit.setText(str(lat))
            self.lon_edit.setText(str(lon))
        self.map_win = MapPopup(_set_coords)
        self.map_win.show()

    # --- online fetch + predict -----------------
    def fetch_and_predict(self):
        lat_txt = self.lat_edit.text().strip()
        lon_txt = self.lon_edit.text().strip()
        date    = self.date_edit.date().toString("yyyy-MM-dd")

        try:
            lat, lon = float(lat_txt), float(lon_txt)
        except ValueError:
            QMessageBox.warning(self, "Input error", "Latitude and longitude must be numeric.")
            return

        try:
            patch_km = float(self.patch_edit.text().strip())
            if patch_km <= 0:
                raise ValueError
        except Exception:
            QMessageBox.warning(self, "Input error", "Patch size must be a positive number (km).")
            return

        try:
            merged, tmpdir = get_image_for(lat, lon, date, patch_km=patch_km)
            cls_idx, _     = predict_image(merged)
            label          = IDX2LABEL[cls_idx]
            self.result.setText(f"Prediction: {label}")
            visualise(merged, label, date)
        except RuntimeError as e:
            QMessageBox.information(self, "No image", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed: {e}")
        finally:
            if 'tmpdir' in locals() and os.path.isdir(tmpdir):
                shutil.rmtree(tmpdir, ignore_errors=True)

    # --- batch: local folder with B8A/B11 -------
    def run_batch_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder containing B8A/B11 TIFFs")
        if not folder:
            return

        grouped = group_bands_by_date(folder)
        if not grouped:
            QMessageBox.information(self, "No files", "No .tif/.tiff files found.")
            return

        out_dir = tempfile.mkdtemp(prefix="earthsense_batch_")
        csv_path = os.path.join(out_dir, "batch_predictions.csv")
        results = []

        processed = 0
        for date_key, bands in grouped.items():
            if 'B8A' in bands and 'B11' in bands:
                merged_path = os.path.join(out_dir, f"{date_key}_merged.tif")
                try:
                    merge_to_two_band_tif(bands['B8A'], bands['B11'], merged_path)
                    cls_idx, _ = predict_image(merged_path)
                    label = IDX2LABEL[cls_idx]
                    results.append((date_key, label, merged_path))
                    processed += 1
                except Exception as e:
                    print(f"[WARN] Skipping {date_key}: {e}")
            else:
                print(f"[INFO] Skipping {date_key} (missing band)")

        if not results:
            shutil.rmtree(out_dir, ignore_errors=True)
            QMessageBox.information(self, "No results", "No valid B8A/B11 pairs were found.")
            return

        # Save a small CSV summary.
        try:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["date", "prediction", "merged_path"])
                writer.writerows(results)
        except Exception as e:
            print(f"[WARN] Could not save CSV: {e}")

        # Show a quick preview and visualize the first one
        msg = f"Processed {processed} pairs.\nSaved merged TIFFs and CSV to:\n{out_dir}\n\nShowing first result."
        QMessageBox.information(self, "Batch complete", msg)

        first_date, first_label, first_path = results[0]
        self.result.setText(f"Batch example - {first_date}: {first_label}")
        try:
            visualise(first_path, first_label, first_date)
        except Exception as e:
            print(f"[WARN] Could not visualize first result: {e}")

# Run app
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = EarthSenseApp(); win.show()
    sys.exit(app.exec_())
