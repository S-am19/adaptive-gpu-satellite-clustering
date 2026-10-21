"""Sentinel-2 L2A B02/B03/B04/B08 preprocessing; Rasterio is optional.

Raw product DN only: (DN + per-band BOA_ADD_OFFSET) / BOA quantification.
Do not feed already harmonized/scaled STAC assets to this conversion. Default
SCL exclusions: 0 no-data, 1 defective/saturated, 3 shadow, 8/9 cloud, 10 cirrus.
Snow (11), dark pixels (2), and unclassified pixels (7) are retained by default.
Array alignment supports north-up, same-CRS grids; file I/O uses windowed reprojection.
Rasterio is required for authoritative CRS parsing in both paths.
"""

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .contracts import validate_features, validate_integer

BANDS = ('B02', 'B03', 'B04', 'B08')
BAND_IDS = {'1': 'B02', '2': 'B03', '3': 'B04', '7': 'B08'}
EXCLUDED_SCL = (0, 1, 3, 8, 9, 10)


def _metric_crs(value):
    """Use GDAL/PROJ via Rasterio; never infer units from a CRS identifier."""
    try:
        from rasterio.crs import CRS
        from rasterio.errors import CRSError
    except ImportError as exc:
        raise ImportError('CRS validation requires rasterio; install only with approval') from exc
    try:
        crs = CRS.from_user_input(value)
        if not crs.is_projected:
            raise ValueError('Projected CRS with metre-based horizontal units required')
        unit, factor = crs.linear_units_factor
        # Check authoritative parsed unit name AND conversion, rejecting unknown
        # named units even if a caller's WKT assigns them conversion factor 1.
        if (unit.lower() not in ('metre', 'meter') or not np.isfinite(factor)
                or not np.isclose(factor, 1.0, rtol=0, atol=1e-12)):
            raise ValueError('Projected CRS with metre-based horizontal units required')
        return crs
    except CRSError as exc:
        raise ValueError('Invalid or unknown CRS definition') from exc


@dataclass(frozen=True)
class RasterGrid:
    height: int
    width: int
    transform: tuple  # GDAL-independent affine (a,b,c,d,e,f): x=a*col+b*row+c
    crs: str

    def __post_init__(self):
        validate_integer('height', self.height); validate_integer('width', self.width)
        t = tuple(float(x) for x in self.transform)
        if len(t) != 6 or not np.isfinite(t).all() or t[0] <= 0 or t[4] >= 0 or t[1] != 0 or t[3] != 0:
            raise ValueError('A finite north-up affine transform is required')
        if not isinstance(self.crs, str) or not self.crs:
            raise ValueError('An explicit CRS is required')
        crs = _metric_crs(self.crs)
        object.__setattr__(self, 'crs', crs.to_string())
        object.__setattr__(self, 'transform', t)

    def cropped(self, window):
        row, col, height, width = _window(window, self.height, self.width)
        a, b, c, d, e, f = self.transform
        return RasterGrid(height, width, (a, b, c + col*a, d, e, f + row*e), self.crs)


@dataclass(frozen=True)
class ProductCalibration:
    quantification: float
    offsets: tuple  # In BANDS order, never inferred from data values.
    processing_baseline: str
    product_id: str
    acquisition_time: str
    metadata_sha256: str
    nodata_dn: float = 0.0
    saturated_dn: float | None = None

    def __post_init__(self):
        if not np.isfinite(self.quantification) or self.quantification <= 0:
            raise ValueError('BOA quantification must be finite and positive')
        if len(self.offsets) != 4 or not np.isfinite(self.offsets).all():
            raise ValueError('Four finite BOA offsets are required')
        if not np.isfinite(self.nodata_dn) or (self.saturated_dn is not None and not np.isfinite(self.saturated_dn)):
            raise ValueError('Invalid special DN value')
        object.__setattr__(self, 'offsets', tuple(float(x) for x in self.offsets))


def parse_product_metadata(xml):
    """Parse original MTD_MSIL2A XML, rejecting missing/ambiguous calibration.

    PB >= 04.00 requires explicit offsets. Earlier products without offset
    entries use zero, following the pre-offset product convention.
    """
    data = xml.encode() if isinstance(xml, str) else xml
    root = ET.fromstring(data)
    def tag(node):
        return node.tag.rsplit('}', 1)[-1]
    def unique(name, required=True):
        values = [n.text.strip() for n in root.iter() if tag(n) == name and n.text and n.text.strip()]
        if len(set(values)) != 1:
            if not values and not required:
                return ''
            raise ValueError(f'Missing or ambiguous {name}')
        return values[0]
    if unique('PROCESSING_LEVEL').lower() != 'level-2a':
        raise ValueError('Only Level-2A product metadata is supported')
    baseline = unique('PROCESSING_BASELINE')
    try:
        baseline_value = float(baseline)
    except ValueError as exc:
        raise ValueError('Invalid processing baseline') from exc
    if not np.isfinite(baseline_value) or baseline_value < 0:
        raise ValueError('Invalid processing baseline')
    quantification = float(unique('BOA_QUANTIFICATION_VALUE'))
    found = {}
    for n in root.iter():
        if tag(n) == 'BOA_ADD_OFFSET' and n.attrib.get('band_id') in BAND_IDS:
            band = BAND_IDS[n.attrib['band_id']]
            if band in found:
                raise ValueError(f'Duplicate offset for {band}')
            found[band] = float(n.text)
    if found and set(found) != set(BANDS):
        raise ValueError('Missing offsets for selected bands')
    if baseline_value >= 4 and set(found) != set(BANDS):
        raise ValueError('PB >= 04.00 requires explicit BOA offsets')
    special = {}
    for n in root.iter():
        if tag(n) == 'Special_Values':
            entries = {tag(child): child.text for child in n}
            name = entries.get('SPECIAL_VALUE_TEXT', '').upper()
            if name in ('NODATA', 'SATURATED'):
                special[name] = float(entries['SPECIAL_VALUE_INDEX'])
    # DN=0 remains invalid even if additional raster no-data values exist.
    return ProductCalibration(quantification, tuple(found.get(b, 0) for b in BANDS),
                              baseline, unique('PRODUCT_URI'),
                              unique('PRODUCT_START_TIME'), hashlib.sha256(data).hexdigest(),
                              special.get('NODATA', 0), special.get('SATURATED'))


def read_product_metadata(path):
    return parse_product_metadata(Path(path).read_bytes())


def _window(window, height, width):
    if window is None:
        return 0, 0, height, width
    if len(window) != 4:
        raise ValueError('window must be (row, col, height, width)')
    row, col, h, w = window
    for name, value in zip(('row', 'col', 'height', 'width'), window):
        validate_integer(name, value, 0 if name in ('row', 'col') else 1)
    if row+h > height or col+w > width:
        raise ValueError('Crop lies outside image')
    return row, col, h, w


def align_scl(scl, source_grid, target_grid):
    """Nearest source cell at each target pixel center; outside coverage is 0.

    Resolution ratios need not be integers. CRS/alignment are explicit rather
    than inferred from array sizes; crop offsets are respected.
    """
    # A masked category means no-data regardless of its hidden stored value.
    # Fill only after validating dtype/shape; never mutate the caller's raster.
    scl = np.ma.asarray(scl)
    if scl.shape != (source_grid.height, source_grid.width) or not np.issubdtype(scl.dtype, np.integer):
        raise ValueError('SCL must be an integer raster matching its source grid')
    scl = scl.filled(0)
    if np.any((scl < 0) | (scl > 11)):
        raise ValueError('Unknown SCL category')
    if _metric_crs(source_grid.crs) != _metric_crs(target_grid.crs):
        raise ValueError('Array SCL alignment requires the same CRS; use the file reader for reprojection')
    a, _, c, _, e, f = target_grid.transform
    sa, _, sc, _, se, sf = source_grid.transform
    cols = np.floor((c + (np.arange(target_grid.width)+0.5)*a - sc)/sa).astype(np.int64)
    rows = np.floor((f + (np.arange(target_grid.height)+0.5)*e - sf)/se).astype(np.int64)
    inside = ((rows[:, None] >= 0) & (rows[:, None] < source_grid.height)
              & (cols[None, :] >= 0) & (cols[None, :] < source_grid.width))
    result = scl[np.clip(rows, 0, source_grid.height-1)[:, None],
                 np.clip(cols, 0, source_grid.width-1)[None, :]].copy()
    result[~inside] = 0
    return result


@dataclass
class PreparedScene:
    X: np.ndarray
    valid_mask: np.ndarray  # All eligible crop pixels, including unsampled ones.
    pixel_indices: np.ndarray  # Selected row locations in flattened crop.
    metadata: dict

    def standardized(self, statistics):
        """Apply frozen training statistics and retain their provenance in metadata."""
        if not isinstance(statistics, Standardization):
            raise TypeError('Frozen Standardization is required')
        X = statistics.transform(self.X)
        metadata = dict(self.metadata)
        metadata.update(standardization=json.loads(statistics.to_json()),
                        reflectance_X_sha256=self.metadata['X_sha256'],
                        X_sha256=hashlib.sha256(X.tobytes()).hexdigest())
        return PreparedScene(X, self.valid_mask.copy(), self.pixel_indices.copy(), metadata)

    def map_labels(self, labels, fill=-1):
        labels = np.asarray(labels)
        if labels.shape != (len(self.X),) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError('Integer labels must match selected feature rows')
        image = np.full(self.valid_mask.shape, fill, dtype=np.int64)
        image.ravel()[self.pixel_indices] = labels
        return image


def preprocess_arrays(bands, scl, grid, scl_grid, calibration, *, window=None,
                      sample_size=None, seed=0, exclude_snow=False, band_masks=None):
    """Prepare already loaded raw DN arrays; file reader bounds I/O separately.

    Bands must share the supplied 10m grid, ordered by canonical B02/B03/B04/B08
    names. Masks mark valid raster pixels. Sampling follows masking and does not
    reduce the memory needed for the crop; use a crop to bound that allocation.
    """
    _metric_crs(grid.crs)
    _metric_crs(scl_grid.crs)
    if set(bands) != set(BANDS):
        raise ValueError(f'Exactly these bands are required: {BANDS}')
    if not np.isclose(grid.transform[0], 10) or not np.isclose(grid.transform[4], -10):
        raise ValueError('Selected bands must use the common 10m grid')
    if not isinstance(calibration, ProductCalibration):
        raise TypeError('ProductCalibration from product metadata is required')
    if not isinstance(exclude_snow, (bool, np.bool_)):
        raise TypeError('exclude_snow must be boolean')
    validate_integer('seed', seed, 0)
    if sample_size is not None:
        validate_integer('sample_size', sample_size)
    if band_masks is not None and set(band_masks) != set(BANDS):
        raise ValueError('Masks must contain every selected band')
    row, col, h, w = _window(window, grid.height, grid.width)
    crop_grid = grid.cropped((row, col, h, w))
    aligned = align_scl(scl, scl_grid, crop_grid)
    excluded = EXCLUDED_SCL + ((11,) if exclude_snow else ())
    valid = ~np.isin(aligned, excluded)
    features = np.empty((h, w, 4), np.float64)
    for index, band in enumerate(BANDS):
        data = np.ma.asarray(bands[band])
        if data.shape != (grid.height, grid.width) or not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
            raise ValueError(f'{band} must be a real raster matching the common grid')
        cropped_data = data[row:row+h, col:col+w]
        supplied_valid = ~np.ma.getmaskarray(cropped_data)
        dn = cropped_data.astype(np.float64).filled(0)
        good = supplied_valid & np.isfinite(dn) & (dn > 0) & (dn != calibration.nodata_dn)
        if calibration.saturated_dn is not None:
            good &= dn != calibration.saturated_dn
        if band_masks is not None:
            mask = np.ma.asarray(band_masks[band])
            if mask.dtype != np.bool_ or mask.shape != data.shape:
                raise ValueError('Band masks must be boolean and match the band grid')
            good &= mask[row:row+h, col:col+w].filled(False)
        features[:, :, index] = (dn + calibration.offsets[index]) / calibration.quantification
        valid &= good & np.isfinite(features[:, :, index])
    valid &= np.all(np.abs(features) <= np.finfo(np.float32).max, axis=2)
    indices = np.flatnonzero(valid.ravel())
    if not len(indices):
        raise ValueError('No valid pixels remain')
    if sample_size is not None and sample_size < len(indices):
        indices = np.sort(np.random.default_rng(seed).choice(indices, sample_size, replace=False))
    X = np.ascontiguousarray(features.reshape(-1, 4)[indices], dtype=np.float32)
    validate_features('X', X)
    metadata = dict(bands=list(BANDS), source_grid=asdict(grid), crop_grid=asdict(crop_grid),
                    scl_source_grid=asdict(scl_grid), scl_resampling="nearest at pixel centers",
                    window=[row, col, h, w], calibration=asdict(calibration), excluded_scl=list(excluded),
                    valid_pixels=int(valid.sum()), selected_pixels=len(indices), total_pixels=h*w,
                    invalid_pixels=int((~valid).sum()), sampling_seed=int(seed),
                    sampling='sorted uniform sample without replacement' if len(indices) < valid.sum() else 'all valid',
                    scaling='product BOA reflectance; negative reflectance retained',
                    X_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
                    pixel_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest())
    return PreparedScene(X, np.ascontiguousarray(valid), indices, metadata)


@dataclass(frozen=True)
class Standardization:
    mean: tuple
    scale: tuple
    training_ids: tuple

    def __post_init__(self):
        if (len(self.mean) != 4 or len(self.scale) != 4 or not np.isfinite(self.mean).all()
                or not np.isfinite(self.scale).all() or np.any(np.asarray(self.scale) <= 0)
                or not self.training_ids):
            raise ValueError('Four finite means/positive scales and training provenance are required')
        if any(not isinstance(x, str) or not x for x in self.training_ids):
            raise ValueError('Training scene IDs must be nonempty strings')
        object.__setattr__(self, 'mean', tuple(float(x) for x in self.mean))
        object.__setattr__(self, 'scale', tuple(float(x) for x in self.scale))
        object.__setattr__(self, 'training_ids', tuple(self.training_ids))

    def transform(self, X):
        validate_features('X', X)
        if X.shape[1] != 4:
            raise ValueError('Expected four bands')
        values = (X.astype(np.float64) - self.mean) / self.scale
        if not np.isfinite(values).all() or np.any(np.abs(values) > np.finfo(np.float32).max):
            raise ValueError('Standardized values exceed finite float32 range')
        return np.ascontiguousarray(values, dtype=np.float32)

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text):
        values = json.loads(text)
        return cls(tuple(values['mean']), tuple(values['scale']), tuple(values['training_ids']))


def fit_standardization(training_X, *, training_ids, split):
    """Fit only explicitly designated training arrays; never auto-fit on transform.

    Pixel-weighted population variance, merged in float64 in input order. A
    constant band uses scale=1. Caller is responsible for truthful split labels.
    """
    if split != 'train':
        raise ValueError('Statistics may only be fitted on the training split')
    arrays = iter(training_X)
    ids = tuple(training_ids)
    count = 0; mean = np.zeros(4); m2 = np.zeros(4); scenes = 0
    for X in arrays:
        validate_features('training_X', X)
        if X.shape[1] != 4:
            raise ValueError('Expected four bands')
        values = X.astype(np.float64)
        n = len(values); local_mean = values.mean(axis=0)
        local_m2 = np.sum((values-local_mean)**2, axis=0)
        delta = local_mean-mean; total = count+n
        m2 += local_m2 + delta*delta*count*n/total
        mean += delta*n/total
        count = total; scenes += 1
    if not count or scenes != len(ids) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError('Each training array needs an explicit scene ID')
    scale = np.sqrt(m2/count); scale[scale == 0] = 1
    return Standardization(tuple(mean), tuple(scale), ids)


def load_sentinel2(band_paths, scl_path, metadata_path, *, window=None,
                   max_crop_pixels=1024*1024, sample_size=None, seed=0, exclude_snow=False):
    """Optional windowed Rasterio reader for original product band assets.

    No downloads. Band paths must belong to the specified product/granule.
    Raster masks and SCL coverage are respected. Only common north-up projected
    metre-based 10m band grids are accepted. SCL uses masked window reads and nearest-neighbor reprojection.
    A full tile is rejected by default; specify a crop to control memory.
    """
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import reproject, transform_bounds
        from rasterio.windows import Window, bounds, from_bounds
    except ImportError as exc:
        raise ImportError('File loading requires optional rasterio; install only with approval') from exc
    from contextlib import ExitStack
    if set(band_paths) != set(BANDS):
        raise ValueError(f'Expected {BANDS}')
    validate_integer('max_crop_pixels', max_crop_pixels)
    calibration = read_product_metadata(metadata_path)
    with ExitStack() as stack:
        sources = {b: stack.enter_context(rasterio.open(band_paths[b])) for b in BANDS}
        ref = sources[BANDS[0]]
        _metric_crs(ref.crs)
        grid = RasterGrid(ref.height, ref.width, tuple(ref.transform)[:6], ref.crs.to_string())
        for src in sources.values():
            if src.count != 1 or src.crs != ref.crs or src.transform != ref.transform or src.shape != ref.shape:
                raise ValueError('All four band assets must share shape, transform and CRS')
        crop = _window(window, grid.height, grid.width)
        row, col, h, w = crop
        if h*w > max_crop_pixels:
            raise MemoryError('Crop exceeds max_crop_pixels; choose a smaller window')
        cropped = grid.cropped(crop)
        read_window = Window(col, row, w, h)
        bands = {}; masks = {}
        for b, src in sources.items():
            values = src.read(1, window=read_window, masked=True)
            bands[b] = values.data; masks[b] = ~np.ma.getmaskarray(values)
        scl_source = stack.enter_context(rasterio.open(scl_path))
        if scl_source.count != 1:
            raise ValueError('SCL needs one band and explicit CRS')
        _metric_crs(scl_source.crs)
        source_grid = RasterGrid(scl_source.height, scl_source.width,
                                 tuple(scl_source.transform)[:6], scl_source.crs.to_string())
        # Read the SCL crop WITH its explicit mask first. A GDAL warper given
        # both a dataset mask and source no-data can ignore the dataset mask.
        # Filling masked categories with 0 before warping makes both invalid.
        source_bounds = transform_bounds(ref.crs, scl_source.crs,
                                         *bounds(read_window, ref.transform), densify_pts=21)
        fractional = from_bounds(*source_bounds, transform=scl_source.transform)
        c0 = max(0, int(np.floor(fractional.col_off))-1)
        r0 = max(0, int(np.floor(fractional.row_off))-1)
        c1 = min(source_grid.width, int(np.ceil(fractional.col_off+fractional.width))+1)
        r1 = min(source_grid.height, int(np.ceil(fractional.row_off+fractional.height))+1)
        scl = np.zeros((h, w), dtype=np.uint8)
        if c1 > c0 and r1 > r0:
            if (r1-r0)*(c1-c0) > max_crop_pixels:
                raise MemoryError('SCL source window exceeds max_crop_pixels')
            source_window = Window(c0, r0, c1-c0, r1-r0)
            source_values = scl_source.read(1, window=source_window, masked=True).filled(0)
            if (not np.issubdtype(source_values.dtype, np.integer)
                    or np.any((source_values < 0) | (source_values > 11))):
                raise ValueError('SCL must contain integer categories 0..11')
            reproject(source_values, scl,
                      src_transform=scl_source.window_transform(source_window), src_crs=scl_source.crs,
                      dst_transform=ref.window_transform(read_window), dst_crs=ref.crs,
                      src_nodata=0, dst_nodata=0, resampling=Resampling.nearest)
        scene = preprocess_arrays(bands, scl, cropped, cropped, calibration,
                                  sample_size=sample_size, seed=seed, exclude_snow=exclude_snow,
                                  band_masks=masks)
        scene.metadata.update(source_grid=asdict(grid), window=list(crop),
                              scl_source_grid=dict(height=scl_source.height, width=scl_source.width,
                                                   transform=list(scl_source.transform)[:6],
                                                   crs=scl_source.crs.to_string()),
                              band_paths={b: str(Path(path).resolve()) for b, path in band_paths.items()},
                              scl_path=str(Path(scl_path).resolve()),
                              scl_resampling='nearest georeferenced reprojection of masked source window')
        return scene
