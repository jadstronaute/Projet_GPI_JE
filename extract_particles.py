"""
Cryo-EM Template Matching Pipeline
------------------------------------
Performs 2D template matching between a cryo-EM micrograph (.mrc) and
projections derived from a PDB structure, extracts particle coordinates,
and exports results as an MRC stack, RELION STAR file, and PDF gallery.

Usage:
    python gpi_pipeline.py                        # uses CONFIG values
    python gpi_pipeline.py micrograph.mrc         # overrides MRC_PATH
    python gpi_pipeline.py micrograph.mrc 6BDF    # overrides MRC_PATH and PDB_ID

Output:
    output/<micrograph_name>_<PDB_ID>/
        particles.mrcs   — extracted particle stack (RELION/cryoSPARC compatible)
        particles.star   — particle coordinates in RELION STAR format
        particles.pdf    — visual gallery for manual inspection
"""

import os
import argparse
import urllib.request

import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import fft
from scipy.ndimage import uniform_filter, gaussian_filter, rotate as scipy_rotate
from skimage.feature import peak_local_max
from Bio.PDB import PDBParser


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MRC_PATH            = "./micrograph.mrc"  # path to input micrograph
PDB_ID              = "6BDF"             # RCSB PDB identifier
BIN_SIZE            = 5                  # spatial downsampling factor
PIXEL_SIZE_OVERRIDE = 0.66              # unbinned pixel size in Å; set None to read from header
PAD_WIDTH           = 100               # padding added to micrograph borders before correlation
BLUR_SIGMA          = 1.0               # Gaussian blur applied to projection templates
THRESHOLD_PCT       = 0.5               # minimum NCC score as fraction of map maximum
MIN_DISTANCE        = 20                # minimum separation between peaks in pixels
MARGIN              = 100               # border region excluded from peak detection (pixels)
DIMENSIONS          = 2                 # number of projection axes: 1=XY, 2=XY+XZ, 3=XY+XZ+YZ
OBLONG_PROJECTIONS  = set()            # projections to rotate in-plane, e.g. {'xz'}, set() for none
ROTATION_STEP       = 15               # in-plane rotation step in degrees (used if OBLONG_PROJECTIONS set)
PDB_CACHE_DIR       = "PDB_cache"
PROJ_CACHE_DIR      = "Projection_cache"
OUTPUT_DIR          = "output"


# ─────────────────────────────────────────────────────────────────────────────
# 1. MRC LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_mrc_micrograph(mrc_path: str) -> tuple[np.ndarray, float]:
    """
    Load a 2D cryo-EM micrograph from an MRC file.

    3D volumes are collapsed to 2D by averaging along Z. Pixel size is read
    from the MRC header unless PIXEL_SIZE_OVERRIDE is set.

    Returns
    -------
    image_data : np.ndarray  float32, shape (H, W)
    pixel_size : float       Ångströms per pixel (unbinned)
    """
    if not os.path.exists(mrc_path):
        raise FileNotFoundError(f"MRC file not found: {mrc_path}")

    print(f"-> Loading micrograph: {mrc_path}")

    with mrcfile.open(mrc_path, mode='r', permissive=True) as mrc:
        pixel_size = float(mrc.voxel_size.x)
        raw_data   = mrc.data

        if raw_data.ndim == 3:
            print(f"   Volume detected {raw_data.shape} — collapsing to 2D via Z mean")
            image_data = np.mean(raw_data, axis=0).astype(np.float32)
        elif raw_data.ndim == 2:
            image_data = raw_data.astype(np.float32)
        else:
            raise ValueError(f"Unsupported MRC dimensionality: {raw_data.ndim}D")

    if PIXEL_SIZE_OVERRIDE is not None:
        pixel_size = float(PIXEL_SIZE_OVERRIDE)
        print(f"   Pixel size (override): {pixel_size:.4f} Å/px")
    elif not np.isfinite(pixel_size) or pixel_size <= 0:
        print("   Warning: pixel size missing from header, defaulting to 1.0 Å/px")
        pixel_size = 1.0
    else:
        print(f"   Pixel size (header): {pixel_size:.4f} Å/px")

    print(f"   Micrograph shape: {image_data.shape}")
    return image_data, pixel_size


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def bin_image(image: np.ndarray, bin_size: int) -> np.ndarray:
    """
    Downsample an image by block-averaging non-overlapping bins.
    Trailing pixels that do not fill a complete bin are discarded.
    """
    if bin_size <= 1:
        return image.copy()

    h, w  = image.shape
    new_h = h // bin_size
    new_w = w // bin_size

    return (
        image[: new_h * bin_size, : new_w * bin_size]
        .reshape(new_h, bin_size, new_w, bin_size)
        .mean(axis=(1, 3))
        .astype(np.float32)
    )


def preprocess_image(image: np.ndarray, bin_size: int = 1, pad_width: int = 0) -> np.ndarray:
    """
    Prepare a micrograph for template matching:
      1. Bin (block-average downsampling)
      2. Invert contrast (cryo-EM high-density regions appear dark; inversion
         makes them bright, consistent with the projection templates)
      3. Normalise to [0, 1]
      4. Pad borders with zeros (avoids edge artefacts in FFT correlation)

    Returns float32 array.
    """
    processed = bin_image(image, bin_size)
    processed = -processed

    lo, hi = processed.min(), processed.max()
    processed = (processed - lo) / (hi - lo) if hi - lo > 1e-10 else np.zeros_like(processed)

    if pad_width > 0:
        processed = np.pad(
            processed,
            pad_width=((pad_width, pad_width), (pad_width, pad_width)),
            mode='constant',
            constant_values=0.0,
        )

    print(f"   Preprocessed shape: {processed.shape}")
    return processed


# ─────────────────────────────────────────────────────────────────────────────
# 3. PDB DOWNLOAD AND PROJECTION
# ─────────────────────────────────────────────────────────────────────────────

def get_pdb_projections(
    pdb_id: str,
    mrc_pixel_size: float,
    dimensions: int = 1,
    pdb_cache_dir: str = "PDB_cache",
    proj_cache_dir: str = "Projection_cache",
    padding: float = 5.0,
) -> dict[str, np.ndarray]:
    """
    Return 2D atom-density projections of a PDB structure scaled to the
    micrograph pixel size.

    PDB files and computed projections are cached on disk. Cached projections
    are keyed by PDB ID and pixel size, so changing the pixel size or binning
    factor automatically triggers recomputation.

    Parameters
    ----------
    pdb_id         : 4-character RCSB identifier, e.g. '6BDF'
    mrc_pixel_size : effective pixel size of the (binned) micrograph in Å/px
    dimensions     : 1 → XY only | 2 → XY + XZ | 3 → XY + XZ + YZ
    padding        : Å of empty space added around the atomic bounding box

    Returns
    -------
    dict mapping axis label ('xy', 'xz', 'yz') to normalised 2D float array
    """
    os.makedirs(pdb_cache_dir,  exist_ok=True)
    os.makedirs(proj_cache_dir, exist_ok=True)

    pdb_id       = pdb_id.upper()
    cache_suffix = f"{pdb_id}_px{mrc_pixel_size:.3f}_sq"
    requested    = {1: ['xy'], 2: ['xy', 'xz'], 3: ['xy', 'xz', 'yz']}.get(dimensions, ['xy'])

    # Load any projections already on disk
    projections = {}
    for key in requested:
        path = os.path.join(proj_cache_dir, f"{cache_suffix}_{key}.npy")
        if os.path.exists(path):
            projections[key] = np.load(path)

    if len(projections) == len(requested):
        print(f"-> Projections loaded from cache ({pdb_id}, {mrc_pixel_size:.3f} Å/px)")
        return projections

    # Download PDB file if not cached
    pdb_path = os.path.join(pdb_cache_dir, f"{pdb_id}.pdb")
    if not os.path.exists(pdb_path):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        print(f"-> Downloading {pdb_id} from RCSB PDB...")
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                pdb_content = response.read().decode('utf-8')
            with open(pdb_path, 'w') as f:
                f.write(pdb_content)
            print(f"   Saved to {pdb_path}")
        except Exception as exc:
            raise RuntimeError(f"Failed to download {pdb_id}: {exc}") from exc
    else:
        print(f"-> Using cached PDB: {pdb_path}")

    # Parse atomic coordinates and centre on centroid
    parser    = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)
    coords    = np.array([
        atom.get_coord()
        for chain   in structure[0].get_chains()
        for residue in chain.get_residues()
        for atom    in residue.get_atoms()
    ])

    if len(coords) == 0:
        raise ValueError(f"No atomic coordinates found in {pdb_path}")

    coords -= coords.mean(axis=0)
    cx, cy, cz = coords[:, 0], coords[:, 1], coords[:, 2]

    def _make_projection(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
        """
        Project atoms onto a 2D plane by histogramming coordinates.
        Output is a square array with pixel size matching the micrograph.
        """
        span1 = (c1.max() - c1.min()) + 2 * padding
        span2 = (c2.max() - c2.min()) + 2 * padding
        bins  = max(
            int(np.round(span1 / mrc_pixel_size)),
            int(np.round(span2 / mrc_pixel_size)),
        )
        expand1 = (bins * mrc_pixel_size - span1) / 2
        expand2 = (bins * mrc_pixel_size - span2) / 2
        r1 = [c1.min() - padding - expand1, c1.max() + padding + expand1]
        r2 = [c2.min() - padding - expand2, c2.max() + padding + expand2]

        H, _, _ = np.histogram2d(c1, c2, bins=[bins, bins], range=[r1, r2])
        return H / H.max() if H.max() > 0 else H

    axis_map = {'xy': (cx, cy), 'xz': (cx, cz), 'yz': (cy, cz)}

    for key in requested:
        if key not in projections:
            proj = _make_projection(*axis_map[key])
            np.save(os.path.join(proj_cache_dir, f"{cache_suffix}_{key}.npy"), proj)
            projections[key] = proj
            print(f"   {key.upper()} projection: {proj.shape}")

    print("-> Projections ready.")
    return projections


# ─────────────────────────────────────────────────────────────────────────────
# 4. FFT NORMALISED CROSS-CORRELATION AND PEAK DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def fft_ncc(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Compute the normalised cross-correlation (NCC) between an image and a
    template using FFT convolution.

    Both the image (via a sliding local window) and the template are
    zero-meaned and L2-normalised before correlation, making the score
    invariant to local brightness and contrast. Output values lie in [-1, 1],
    where 1 indicates a perfect match.

    The alignment convention used here (conjugate in frequency domain +
    half-template roll) places each score at the centre of the matched region,
    consistent with standard cryo-EM template-matching expectations.

    Returns float32 array with the same shape as image.
    """
    image    = np.asarray(image,    dtype=np.float32)
    template = np.asarray(template, dtype=np.float32)
    t_h, t_w = template.shape

    t_zm    = template - template.mean()
    t_sumsq = np.sum(t_zm ** 2)
    if t_sumsq == 0:
        raise ValueError("Template has zero variance — cannot correlate.")

    local_mean = uniform_filter(image, size=(t_h, t_w))
    img_zm     = image - local_mean

    # Pad to next power of 2 for FFT efficiency
    s1    = image.shape[0] + t_h - 1
    s2    = image.shape[1] + t_w - 1
    fsize = (int(2 ** np.ceil(np.log2(s1))), int(2 ** np.ceil(np.log2(s2))))

    corr = np.real(
        fft.ifft2(fft.fft2(img_zm, fsize) * np.conj(fft.fft2(t_zm, fsize)))
    )
    # Shift output so peak position corresponds to template centre in the image
    corr      = np.roll(corr, shift=(t_h // 2, t_w // 2), axis=(0, 1))
    numerator = corr[: image.shape[0], : image.shape[1]]

    local_var = np.maximum(
        uniform_filter(image ** 2, size=(t_h, t_w)) - local_mean ** 2, 0.0
    )
    denom = np.sqrt(local_var * (t_h * t_w) * t_sumsq)

    return np.where(denom > 1e-10, numerator / denom, 0.0).astype(np.float32)


def get_inplane_rotations(template: np.ndarray, step_deg: float) -> list[tuple[float, np.ndarray]]:
    """
    Generate in-plane rotations of a template at regular angular intervals.

    Covers 0–180° only; a 180° rotation produces the same NCC score as 0°
    due to the symmetry of cross-correlation, so 180–360° is redundant.

    Returns list of (angle_degrees, rotated_template) tuples.
    """
    rotations = []
    for angle in np.arange(0, 180, step_deg):
        rotated = scipy_rotate(template, angle=float(angle),
                               reshape=False, mode='constant', cval=0.0)
        rotations.append((float(angle), rotated))
    print(f"   {len(rotations)} in-plane rotations generated (0–180°, step={step_deg}°)")
    return rotations


def compute_ncc_maps(
    processed: np.ndarray,
    projections: dict[str, np.ndarray],
    blur_sigma: float,
    oblong_projections: set[str],
    rotation_step: float,
) -> np.ndarray:
    """
    Run FFT NCC for all projection templates and return a single merged score map.

    For projections listed in oblong_projections, in-plane rotations are also
    evaluated. The final map is the elementwise maximum across all templates
    and rotations tested, so each pixel reflects the best match achieved.

    Returns float32 array with the same shape as processed.
    """
    all_maps = []

    for key, proj in projections.items():
        blurred = gaussian_filter(proj, sigma=blur_sigma)

        rotations = (
            get_inplane_rotations(blurred, rotation_step)
            if key in oblong_projections
            else [(0.0, blurred)]
        )

        for angle, template in rotations:
            ncc = fft_ncc(processed, template)
            all_maps.append(ncc)
            print(f"   {key.upper()} @ {angle:>6.1f}°  max score: {ncc.max():.4f}")

    merged = np.maximum.reduce(all_maps)
    print(f"\n   Merged score range: [{merged.min():.4f}, {merged.max():.4f}]  "
          f"({len(all_maps)} maps)")
    return merged


def find_peaks(
    score_map: np.ndarray,
    threshold_pct: float = 0.5,
    min_distance: int = 20,
    margin: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect local maxima in an NCC score map.

    Parameters
    ----------
    threshold_pct : peaks below this fraction of the map maximum are rejected
    min_distance  : minimum pixel separation between accepted peaks
    margin        : border region (in pixels) zeroed before detection to
                    suppress padding and edge artefacts

    Returns
    -------
    coords : (N, 2) int array of (row, col) positions
    scores : (N,)  float array of NCC scores at each position
    """
    m = score_map.copy()
    if margin > 0:
        m[:margin,  :]  = 0.0
        m[-margin:, :]  = 0.0
        m[:,  :margin]  = 0.0
        m[:, -margin:]  = 0.0

    coords = peak_local_max(
        m,
        min_distance=min_distance,
        threshold_abs=threshold_pct * m.max(),
    )
    scores = score_map[coords[:, 0], coords[:, 1]]
    return coords, scores


# ─────────────────────────────────────────────────────────────────────────────
# 5. PARTICLE EXTRACTION AND OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def extract_patches(
    image: np.ndarray,
    coords: np.ndarray,
    scores: np.ndarray,
    box_size: int,
) -> list[dict]:
    """
    Extract square patches centred on each detected peak position.

    Particles whose bounding box extends beyond the image border are skipped.

    Returns list of dicts with keys:
        patch  : float32 array of shape (box_size, box_size)
        y, x   : integer centre coordinates in the processed image
        score  : NCC score at this position
        index  : 1-based particle index
    """
    half = box_size // 2
    h, w = image.shape
    particles = []

    for i, ((y, x), score) in enumerate(zip(coords, scores)):
        if y - half < 0 or y + half >= h or x - half < 0 or x + half >= w:
            continue
        particles.append({
            'patch' : image[y - half : y + half, x - half : x + half].copy().astype(np.float32),
            'y'     : int(y),
            'x'     : int(x),
            'score' : float(score),
            'index' : i + 1,
        })

    print(f"-> {len(particles)} particles extracted "
          f"({len(coords) - len(particles)} skipped — too close to border)")
    return particles


def save_mrcs(particles: list[dict], output_path: str, pixel_size: float) -> None:
    """
    Save extracted particle patches as an MRC image stack (.mrcs).
    Pixel size is written to the header for compatibility with RELION and cryoSPARC.
    Stack shape: (N, box_size, box_size), dtype float32.
    """
    if not particles:
        print("-> No particles to save (MRCS skipped).")
        return

    stack = np.stack([p['patch'] for p in particles], axis=0)
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(stack)
        mrc.voxel_size = pixel_size

    print(f"-> MRCS saved: {output_path}  ({len(particles)} particles)")


def save_star(
    particles: list[dict],
    micrograph_name: str,
    output_path: str,
    bin_size: int = 1,
) -> None:
    """
    Save particle coordinates as a RELION-compatible STAR file.

    Coordinates are written in unbinned pixel space (multiplied by bin_size),
    which is required by RELION for CTF correction and downstream processing.
    """
    if not particles:
        print("-> No particles to save (STAR skipped).")
        return

    with open(output_path, 'w') as f:
        f.write("data_particles\n\n")
        f.write("loop_\n")
        f.write("_rlnMicrographName\n")
        f.write("_rlnCoordinateX\n")
        f.write("_rlnCoordinateY\n")
        f.write("_rlnAutopickFigureOfMerit\n")
        for p in particles:
            f.write(
                f"{micrograph_name}\t"
                f"{p['x'] * bin_size:.2f}\t"
                f"{p['y'] * bin_size:.2f}\t"
                f"{p['score']:.6f}\n"
            )

    print(f"-> STAR saved: {output_path}  ({len(particles)} particles)")


def save_pdf(
    particles: list[dict],
    output_path: str,
    projections: dict[str, np.ndarray],
    pdb_id: str,
    blur_sigma: float,
    n_cols: int = 5,
) -> None:
    """
    Save a PDF gallery for manual inspection.

    Page 1: all projection templates used for matching (blurred, as correlated).
    Page 2+: all extracted patches in a grid, labelled with index,
             coordinates, and NCC score. Display contrast is set per-patch
             using the 2nd–98th percentile range.
    """
    if not particles:
        print("-> No particles to save (PDF skipped).")
        return

    n      = len(particles)
    n_rows = int(np.ceil(n / n_cols))

    with PdfPages(output_path) as pdf:
        # Page 1 — one panel per projection template
        n_proj = len(projections)
        fig, axes = plt.subplots(1, n_proj, figsize=(4 * n_proj, 4),
                                 squeeze=False)
        axes = axes.ravel()

        for i, (key, proj) in enumerate(projections.items()):
            axes[i].imshow(gaussian_filter(proj, sigma=blur_sigma), cmap='gray')
            axes[i].set_title(f"{key.upper()} projection\nPDB: {pdb_id}")
            axes[i].axis('off')

        fig.suptitle(f"{n} particles extracted", fontsize=11)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 2+ — particle gallery
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows),
                                 squeeze=False)
        axes = axes.ravel()

        for i, p in enumerate(particles):
            axes[i].imshow(
                p['patch'], cmap='gray',
                vmin=np.percentile(p['patch'], 2),
                vmax=np.percentile(p['patch'], 98),
            )
            axes[i].set_title(
                f"#{p['index']}  ({p['x']}, {p['y']})\n{p['score']:.3f}",
                fontsize=7,
            )
            axes[i].axis('off')

        for j in range(n, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f"-> PDF saved: {output_path}")


def save_all(
    particles: list[dict],
    micrograph_name: str,
    output_dir: str,
    pixel_size: float,
    projections: dict[str, np.ndarray],
    pdb_id: str,
    blur_sigma: float,
    bin_size: int = 1,
) -> None:
    """Write MRCS, STAR, and PDF outputs to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    save_mrcs(particles, os.path.join(output_dir, "particles.mrcs"), pixel_size)
    save_star(particles, micrograph_name, os.path.join(output_dir, "particles.star"), bin_size)
    save_pdf(particles,  os.path.join(output_dir, "particles.pdf"),
             projections, pdb_id, blur_sigma)
    print(f"-> All outputs written to: {output_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# 6. PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    print("=" * 60)
    print("  Cryo-EM Template Matching Pipeline")
    print(f"  MRC : {MRC_PATH}")
    print(f"  PDB : {PDB_ID}   Binning: {BIN_SIZE}x")
    print("=" * 60)

    # 1. Load
    image, pixel_size = load_mrc_micrograph(MRC_PATH)

    # 2. Preprocess
    print("\n[1/5] Preprocessing...")
    processed         = preprocess_image(image, bin_size=BIN_SIZE, pad_width=PAD_WIDTH)
    binned_pixel_size = pixel_size * BIN_SIZE
    print(f"   Effective pixel size after {BIN_SIZE}x binning: {binned_pixel_size:.3f} Å/px")

    # 3. Project
    print("\n[2/5] Loading PDB projections...")
    projections = get_pdb_projections(
        pdb_id         = PDB_ID,
        mrc_pixel_size = binned_pixel_size,
        dimensions     = DIMENSIONS,
        pdb_cache_dir  = PDB_CACHE_DIR,
        proj_cache_dir = PROJ_CACHE_DIR,
    )
    box_size = projections['xy'].shape[0] + 10

    # 4. Correlate
    print("\n[3/5] Running FFT NCC...")
    ncc_map = compute_ncc_maps(
        processed,
        projections,
        blur_sigma         = BLUR_SIGMA,
        oblong_projections = OBLONG_PROJECTIONS,
        rotation_step      = ROTATION_STEP,
    )

    # 5. Detect peaks
    print("\n[4/5] Detecting peaks...")
    coords, scores = find_peaks(
        ncc_map,
        threshold_pct = THRESHOLD_PCT,
        min_distance  = MIN_DISTANCE,
        margin        = MARGIN + PAD_WIDTH,
    )
    print(f"   {len(coords)} peaks detected")

    # 6. Extract and save
    print("\n[5/5] Extracting and saving...")
    particles       = extract_patches(processed, coords, scores, box_size)
    micrograph_name = os.path.basename(MRC_PATH)
    output_dir      = os.path.join(
        OUTPUT_DIR,
        f"{os.path.splitext(micrograph_name)[0]}_{PDB_ID}",
    )
    save_all(
        particles,
        micrograph_name = micrograph_name,
        output_dir      = output_dir,
        pixel_size      = binned_pixel_size,
        projections     = projections,
        pdb_id          = PDB_ID,
        blur_sigma      = BLUR_SIGMA,
        bin_size        = BIN_SIZE,
    )

    # Visual summary
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(ncc_map, cmap='hot')
    if len(coords):
        axes[0].plot(coords[:, 1], coords[:, 0], 'c+', markersize=8, markeredgewidth=1.5)
    axes[0].set_title(f"NCC score map — {len(coords)} detections")
    axes[0].axis('off')

    axes[1].imshow(
        processed, cmap='gray',
        vmin=np.percentile(processed, 2),
        vmax=np.percentile(processed, 98),
    )
    if len(coords):
        axes[1].plot(coords[:, 1], coords[:, 0], 'r+', markersize=8, markeredgewidth=1.5)
    axes[1].set_title("Preprocessed micrograph with detections")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cryo-EM template matching pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("mrc_path", nargs='?', default=None,
                        help="Path to input MRC micrograph (overrides MRC_PATH in config)")
    parser.add_argument("pdb_id",   nargs='?', default=None,
                        help="RCSB PDB ID (overrides PDB_ID in config)")
    args = parser.parse_args()

    if args.mrc_path:
        MRC_PATH = args.mrc_path
    if args.pdb_id:
        PDB_ID = args.pdb_id

    run_pipeline()
