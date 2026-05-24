# Projet_GPI_Jad_EL-Ayoubi

An educational project done as an assignment in my M1 year, in the educational unit Gestion de Projet Informatique GPI  
•ᴗ• 

## Description:
##Cryo-EM Template Matching Pipeline

A single-file Python pipeline for template-based particle picking in cryo-EM micrographs. Given an MRC micrograph and a PDB structure, it generates 2D projections of the structure, matches them to the micrograph via FFT normalised cross-correlation, and exports the detected particles as an MRC stack, a RELION-compatible STAR file, and a PDF gallery.

---

## Requirements

- Python 3.10 or higher
- The packages listed in `requirements.txt`

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Usage

Edit the `CONFIGURATION` block at the top of `gpi_pipeline.py` to set your defaults, then run:

```bash
# Use values from the CONFIG block
python gpi_pipeline.py

# Override MRC path only
python gpi_pipeline.py path/to/micrograph.mrc

# Override both MRC path and PDB ID
python gpi_pipeline.py path/to/micrograph.mrc 6BDF
```

---

## Configuration

All parameters are documented in the `CONFIGURATION` block at the top of the script. Key ones:

| Parameter | Description |
|---|---|
| `MRC_PATH` | Path to the input micrograph (.mrc) |
| `PDB_ID` | RCSB PDB identifier (e.g. `6BDF`) |
| `BIN_SIZE` | Spatial downsampling factor (e.g. 5 for 5× binning) |
| `PIXEL_SIZE_OVERRIDE` | Unbinned pixel size in Å — recommended if your MRC header is missing this value; set to `None` to read from header |
| `BLUR_SIGMA` | Gaussian blur applied to projection templates before correlation |
| `THRESHOLD_PCT` | Fraction of the NCC map maximum used as the detection threshold |
| `MIN_DISTANCE` | Minimum pixel separation between two detected peaks |
| `MARGIN` | Border region (pixels) excluded from peak detection |
| `DIMENSIONS` | Number of projection axes: `1` = XY, `2` = XY + XZ, `3` = XY + XZ + YZ |
| `OBLONG_PROJECTIONS` | Set of projection keys to rotate in-plane, e.g. `{'xz'}` for elongated particles |
| `ROTATION_STEP` | In-plane rotation step in degrees (used when `OBLONG_PROJECTIONS` is set) |

---

## Output

Results are written to:

```
output/<micrograph_name>_<PDB_ID>/
    particles.mrcs   — particle image stack, readable by RELION and cryoSPARC
    particles.star   — particle coordinates in RELION STAR format
    particles.pdf    — visual gallery for manual inspection
```

Coordinates in the STAR file are in **unbinned pixel space**, as expected by RELION for CTF correction and downstream processing.

---

## Caching

PDB files and computed projections are cached locally to avoid redundant downloads and recomputation:

```
PDB_cache/           — raw PDB files
Projection_cache/    — precomputed 2D projections (.npy)
```

Projection cache filenames encode the PDB ID and pixel size, so changing the binning factor or pixel size override automatically triggers recomputation while preserving cached projections at other scales.

---

## Notes

- The pipeline is designed for **single-particle cryo-EM** data. It is not suitable for tomography or subtomogram averaging without modification.
- Only 3 projection orientations (XY, XZ, YZ) are used by default. For a more exhaustive angular search, consider tools such as RELION autopicker or crYOLO.
- Output files should be written to a **local disk**. Writing to network drives or cloud-synced folders (iCloud, Dropbox, OneDrive) may cause I/O errors during PDF generation.
