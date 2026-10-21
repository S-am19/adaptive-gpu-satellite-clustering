"""Synthetic Sentinel-2 preprocessing tests; no imagery downloads."""

import importlib.util
import json

import numpy as np
import pytest

from pcap.contracts import validate_points_centroids
from pcap.preprocessing import (BANDS, ProductCalibration, RasterGrid, Standardization,
                                align_scl, fit_standardization, load_sentinel2,
                                parse_product_metadata, preprocess_arrays)


def xml(baseline='04.00', offsets=True, quantification='10000'):
    additions = ''.join(f'<BOA_ADD_OFFSET band_id="{i}">{value}</BOA_ADD_OFFSET>'
                        for i, value in zip((1, 2, 3, 7), (-1000, -900, -800, -700))) if offsets else ''
    return f'''<root xmlns="urn:test"><PROCESSING_LEVEL>Level-2A</PROCESSING_LEVEL>
    <PROCESSING_BASELINE>{baseline}</PROCESSING_BASELINE><PRODUCT_URI>TEST.SAFE</PRODUCT_URI>
    <PRODUCT_START_TIME>2026-10-01T00:00:00Z</PRODUCT_START_TIME>
    <BOA_QUANTIFICATION_VALUE>{quantification}</BOA_QUANTIFICATION_VALUE>{additions}
    <Special_Values><SPECIAL_VALUE_TEXT>NODATA</SPECIAL_VALUE_TEXT><SPECIAL_VALUE_INDEX>0</SPECIAL_VALUE_INDEX></Special_Values>
    <Special_Values><SPECIAL_VALUE_TEXT>SATURATED</SPECIAL_VALUE_TEXT><SPECIAL_VALUE_INDEX>65535</SPECIAL_VALUE_INDEX></Special_Values></root>'''


@pytest.fixture
def scene_arrays():
    grid = RasterGrid(4, 6, (10, 0, 500000, 0, -10, 4000000), 'EPSG:32643')
    scl_grid = RasterGrid(2, 3, (20, 0, 500000, 0, -20, 4000000), grid.crs)
    bands = {b: np.full((4, 6), 2000, np.uint16) for b in BANDS}
    scl = np.array([[4, 8, 6], [3, 11, 7]], np.uint8)
    return bands, scl, grid, scl_grid, parse_product_metadata(xml())


def test_product_metadata_and_band_specific_scaling(scene_arrays):
    *_, calibration = scene_arrays
    assert calibration.offsets == (-1000, -900, -800, -700)
    assert len(calibration.metadata_sha256) == 64
    scene = preprocess_arrays(*scene_arrays)
    assert scene.X.shape == (16, 4)
    np.testing.assert_allclose(scene.X, np.tile([0.10, 0.11, 0.12, 0.13], (16, 1)), rtol=1e-7)
    assert scene.X.dtype == np.float32 and scene.X.flags.c_contiguous
    validate_points_centroids(scene.X, scene.X[:2].copy())
    assert scene.metadata['valid_pixels'] == 16 and scene.metadata['invalid_pixels'] == 8


def test_older_product_without_offsets():
    assert parse_product_metadata(xml('03.01', False)).offsets == (0, 0, 0, 0)


@pytest.mark.parametrize('text', [xml(offsets=False), xml(quantification='0'),
                                xml(quantification='nan'), xml(baseline='nan'),
                                xml().replace('Level-2A', 'Level-1C'),
                                xml().replace('BOA_QUANTIFICATION_VALUE', 'OTHER'),
                                xml().replace('band_id="7"', 'band_id="8"'),
                                xml().replace('</root>', '<BOA_ADD_OFFSET band_id="1">0</BOA_ADD_OFFSET></root>')])
def test_invalid_product_metadata(text):
    with pytest.raises(ValueError):
        parse_product_metadata(text)


def test_nearest_scl_alignment_and_offset_crop(scene_arrays):
    _, scl, grid, source, _ = scene_arrays
    expected = np.repeat(np.repeat(scl, 2, axis=0), 2, axis=1)
    np.testing.assert_array_equal(align_scl(scl, source, grid), expected)
    crop = grid.cropped((1, 1, 3, 4))
    np.testing.assert_array_equal(align_scl(scl, source, crop), expected[1:4, 1:5])
    outside = RasterGrid(2, 2, (10, 0, 0, 0, -10, 0), grid.crs)
    assert not align_scl(scl, source, outside).any()


def test_all_scl_categories_and_snow_policy():
    grid = RasterGrid(1, 12, (10, 0, 0, 0, -10, 0), 'EPSG:32643')
    bands = {b: np.full((1, 12), 2000) for b in BANDS}
    scl = np.arange(12, dtype=np.uint8)[None, :]
    args = bands, scl, grid, grid, parse_product_metadata(xml())
    np.testing.assert_array_equal(preprocess_arrays(*args).pixel_indices, [2, 4, 5, 6, 7, 11])
    np.testing.assert_array_equal(preprocess_arrays(*args, exclude_snow=True).pixel_indices, [2, 4, 5, 6, 7])


def test_invalid_dn_raster_masks_and_negative_reflectance(scene_arrays):
    bands, _, grid, _, calibration = scene_arrays
    bands = {b: v.astype(np.float64) for b, v in bands.items()}
    bands['B02'][0, :5] = [0, np.nan, np.inf, 65535, -1]
    bands['B03'][1, 0] = 0
    bands['B04'][1, 1] = 500  # Valid negative BOA reflectance after offset.
    masks = {b: np.ones((4, 6), bool) for b in BANDS}
    masks['B08'][1, 2] = False
    scene = preprocess_arrays(bands, np.full((4, 6), 4, np.uint8), grid, grid, calibration, band_masks=masks)
    assert not scene.valid_mask[0, :5].any()
    assert not scene.valid_mask[1, 0] and not scene.valid_mask[1, 2]
    index = np.flatnonzero(scene.pixel_indices == 7)[0]
    assert scene.X[index, 2] == np.float32(-0.03)
    assert np.isfinite(scene.X).all()


def test_crop_sampling_reproducibility_and_mapping(scene_arrays):
    a = preprocess_arrays(*scene_arrays, window=(1, 1, 3, 4), sample_size=3, seed=27)
    b = preprocess_arrays(*scene_arrays, window=(1, 1, 3, 4), sample_size=3, seed=27)
    np.testing.assert_array_equal(a.pixel_indices, b.pixel_indices)
    np.testing.assert_array_equal(a.X, b.X)
    assert len(a.X) == 3 and np.all(a.valid_mask.ravel()[a.pixel_indices])
    image = a.map_labels(np.array([0, 1, 2], np.int32))
    np.testing.assert_array_equal(image.ravel()[a.pixel_indices], [0, 1, 2])
    assert np.sum(image == -1) == image.size - 3
    assert a.metadata['crop_grid']['transform'][2] == 500010
    assert a.metadata['crop_grid']['transform'][5] == 3999990
    json.dumps(a.metadata)
    with pytest.raises(ValueError):
        a.map_labels(np.zeros(4, np.int32))


@pytest.mark.parametrize('case', ['band_shape', 'missing_band', 'scl_shape', 'scl_category',
                                 'scl_float', 'crs', 'resolution', 'outside_crop', 'empty', 'mask'])
def test_input_mismatch_rejections(scene_arrays, case):
    bands, scl, grid, scl_grid, cal = scene_arrays
    kwargs = {}
    if case == 'band_shape': bands['B02'] = np.zeros((2, 2))
    if case == 'missing_band': del bands['B08']
    if case == 'scl_shape': scl = scl[:1]
    if case == 'scl_category': scl[0, 0] = 12
    if case == 'scl_float': scl = scl.astype(float)
    if case == 'crs': scl_grid = RasterGrid(2, 3, scl_grid.transform, 'EPSG:32644')
    if case == 'resolution': grid = RasterGrid(4, 6, (20, 0, 0, 0, -20, 0), grid.crs)
    if case == 'outside_crop': kwargs['window'] = (3, 4, 2, 2)
    if case == 'empty': scl[:] = 8
    if case == 'mask': kwargs['band_masks'] = {b: np.ones((4, 6), np.uint8) for b in BANDS}
    with pytest.raises((ValueError, MemoryError)):
        preprocess_arrays(bands, scl, grid, scl_grid, cal, **kwargs)


def test_training_only_standardization_serialization_and_no_leakage():
    X = np.array([[1, 2, 7, 4], [3, 6, 7, 8], [9, 4, 7, 2]], np.float32)
    stats = fit_standardization([X[:2], X[2:]], training_ids=['train-a', 'train-b'], split='train')
    np.testing.assert_allclose(stats.mean, X.astype(float).mean(axis=0))
    expected = X.astype(float).std(axis=0); expected[2] = 1
    np.testing.assert_allclose(stats.scale, expected)
    standardized = stats.transform(X)
    assert standardized.dtype == np.float32 and standardized.flags.c_contiguous
    np.testing.assert_allclose(standardized.mean(axis=0), 0, atol=1e-7)
    restored = Standardization.from_json(stats.to_json())
    heldout = np.full((2, 4), 100, np.float32)
    np.testing.assert_allclose(restored.transform(heldout), (heldout-stats.mean)/stats.scale, rtol=1e-7)
    assert restored == stats  # Held-out values never refit statistics.
    with pytest.raises(ValueError):
        fit_standardization([heldout], training_ids=['test-scene'], split='test')
    with pytest.raises(ValueError):
        fit_standardization([X], training_ids=[], split='train')


def test_optional_file_dependency_error(monkeypatch):
    import builtins
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == 'rasterio': raise ImportError('blocked')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', blocked)
    with pytest.raises(ImportError, match='optional rasterio'):
        load_sentinel2({}, 'missing', 'missing')


@pytest.mark.skipif(importlib.util.find_spec('rasterio') is None, reason='optional rasterio not installed')
def test_windowed_georeferenced_file_reader(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    paths = {}
    for b in BANDS:
        path = tmp_path / f'{b}.tif'; paths[b] = path
        with rasterio.open(path, 'w', driver='GTiff', height=4, width=6, count=1,
                           dtype='uint16', crs='EPSG:32643', transform=from_origin(500000, 4000000, 10, 10), nodata=0) as dst:
            dst.write(np.full((4, 6), 2000, np.uint16), 1)
    scl_path = tmp_path / 'SCL.tif'
    with rasterio.open(scl_path, 'w', driver='GTiff', height=2, width=3, count=1,
                       dtype='uint8', crs='EPSG:32643', transform=from_origin(500000, 4000000, 20, 20), nodata=0) as dst:
        dst.write(np.array([[4, 8, 6], [3, 11, 7]], np.uint8), 1)
    metadata = tmp_path / 'MTD_MSIL2A.xml'; metadata.write_text(xml())
    with pytest.raises(MemoryError):
        load_sentinel2(paths, scl_path, metadata, max_crop_pixels=4)
    scene = load_sentinel2(paths, scl_path, metadata, window=(1, 1, 3, 4))
    assert scene.valid_mask.shape == (3, 4) and scene.X.shape[1] == 4
    np.testing.assert_array_equal(scene.valid_mask, [[True, False, False, True],
                                                  [False, True, True, True],
                                                  [False, True, True, True]])
    assert scene.metadata['window'] == [1, 1, 3, 4]


def test_standardized_scene_retains_mapping_and_statistics(scene_arrays):
    scene = preprocess_arrays(*scene_arrays)
    stats = fit_standardization([scene.X], training_ids=['training-scene'], split='train')
    normalized = scene.standardized(stats)
    assert normalized.metadata['standardization']['training_ids'] == ['training-scene']
    assert normalized.metadata['reflectance_X_sha256'] == scene.metadata['X_sha256']
    assert normalized.metadata['X_sha256'] != scene.metadata['X_sha256']
    np.testing.assert_array_equal(normalized.pixel_indices, scene.pixel_indices)
    np.testing.assert_array_equal(normalized.valid_mask, scene.valid_mask)
    np.testing.assert_allclose(normalized.X, 0, atol=1e-7)
    assert not np.shares_memory(normalized.X, scene.X)
    validate_points_centroids(normalized.X, normalized.X[:1].copy())


@pytest.mark.parametrize('band', BANDS)
def test_masked_band_combines_with_explicit_masks_and_invalid_dn(scene_arrays, band):
    bands, _, grid, _, calibration = scene_arrays
    scl = np.full((4, 6), 4, np.uint8)
    hidden = np.zeros((4, 6), bool); hidden[1, 1] = True
    bands[band] = np.ma.array(bands[band], mask=hidden)
    bands['B02'][1, 2] = 0
    masks = {b: np.ones((4, 6), bool) for b in BANDS}
    masks['B08'][2, 1] = False
    # Even a True stored mask value must not be trusted when it is masked.
    mask_of_mask = np.zeros((4, 6), bool); mask_of_mask[2, 2] = True
    masks['B03'] = np.ma.array(masks['B03'], mask=mask_of_mask)
    scene = preprocess_arrays(bands, scl, grid, grid, calibration,
                              window=(1, 1, 2, 3), band_masks=masks)
    np.testing.assert_array_equal(scene.valid_mask, [[False, False, True],
                                                   [False, False, True]])
    assert scene.X.shape == (2, 4) and np.isfinite(scene.X).all()
    assert bands[band].mask[1, 1]  # Caller masks remain unchanged.
    assert bands[band].data[1, 1] == 2000


def test_masked_scl_resampling_and_nonaligned_crop(scene_arrays):
    bands, scl, grid, scl_grid, calibration = scene_arrays
    mask = np.zeros(scl.shape, bool); mask[0, 0] = True
    scl = np.ma.array(scl, mask=mask)
    scl.data[0, 0] = 255  # Unknown stored value is ignored only where masked.
    before_data, before_mask = scl.data.copy(), scl.mask.copy()
    aligned = align_scl(scl, scl_grid, grid)
    assert not aligned[:2, :2].any()
    expected = np.repeat(np.repeat(np.where(mask, 0, scl.data), 2, axis=0), 2, axis=1)
    np.testing.assert_array_equal(aligned, expected)
    scene = preprocess_arrays(bands, scl, grid, scl_grid, calibration,
                              window=(1, 1, 3, 4))
    np.testing.assert_array_equal(scene.valid_mask, [[False, False, False, True],
                                                   [False, True, True, True],
                                                   [False, True, True, True]])
    np.testing.assert_array_equal(scl.data, before_data)
    np.testing.assert_array_equal(scl.mask, before_mask)


@pytest.mark.parametrize('kind', ['band', 'scl'])
def test_fully_masked_raster_cannot_produce_valid_pixels(scene_arrays, kind):
    bands, scl, grid, scl_grid, calibration = scene_arrays
    if kind == 'band':
        bands['B04'] = np.ma.array(bands['B04'], mask=True)
    else:
        scl = np.ma.array(scl, mask=True)
    with pytest.raises(ValueError, match='No valid pixels'):
        preprocess_arrays(bands, scl, grid, scl_grid, calibration)


def test_unmasked_array_and_masked_array_without_mask_agree(scene_arrays):
    bands, scl, grid, scl_grid, calibration = scene_arrays
    reference = preprocess_arrays(*scene_arrays)
    actual = preprocess_arrays({b: np.ma.array(v, mask=np.ma.nomask) for b, v in bands.items()},
                               np.ma.array(scl, mask=np.ma.nomask), grid, scl_grid, calibration)
    np.testing.assert_array_equal(actual.X, reference.X)
    np.testing.assert_array_equal(actual.valid_mask, reference.valid_mask)
    np.testing.assert_array_equal(actual.pixel_indices, reference.pixel_indices)


@pytest.mark.parametrize('definition', ['EPSG:32643', 'EPSG:32736'])
def test_valid_projected_metre_grids_and_equivalent_wkt(definition):
    from rasterio.crs import CRS
    grid = RasterGrid(2, 2, (10, 0, 500000, 0, -10, 4000000), definition)
    other = RasterGrid(2, 2, grid.transform, CRS.from_user_input(definition).to_wkt())
    scl = np.full((2, 2), 4, np.uint8)
    np.testing.assert_array_equal(align_scl(scl, grid, other), scl)


@pytest.mark.parametrize('definition', ['EPSG:4326', 'EPSG:2263', 'EPSG:999999',
                                       'not-a-crs', '', None])
def test_grid_rejects_geographic_feet_invalid_and_missing_crs(definition):
    with pytest.raises(ValueError):
        RasterGrid(2, 2, (10, 0, 500000, 0, -10, 4000000), definition)


def test_unknown_unit_name_is_not_accepted_as_metres():
    from rasterio.crs import CRS
    wkt = CRS.from_epsg(32643).to_wkt()
    assert 'UNIT["metre",1' in wkt
    unknown = wkt.replace('UNIT["metre",1', 'UNIT["mystery",1')
    with pytest.raises(ValueError, match='metre-based'):
        RasterGrid(2, 2, (10, 0, 0, 0, -10, 0), unknown)


@pytest.mark.parametrize('transform', [(0, 0, 0, 0, -10, 0), (10, 1, 0, 0, -10, 0),
                                      (10, 0, 0, 1, -10, 0), (10, 0, 0, 0, 10, 0),
                                      (10, 0, float('nan'), 0, -10, 0), (10, 0)])
def test_invalid_grid_affines(transform):
    with pytest.raises(ValueError):
        RasterGrid(2, 2, transform, 'EPSG:32643')


@pytest.fixture
def raster_files(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    paths = {}
    for b in BANDS:
        paths[b] = tmp_path / (b + '.tif')
        with rasterio.open(paths[b], 'w', driver='GTiff', height=4, width=4, count=1,
                           dtype='uint16', crs='EPSG:32643', transform=from_origin(500000, 4000000, 10, 10), nodata=0) as dst:
            dst.write(np.full((4, 4), 2000, np.uint16), 1)
    scl = tmp_path / 'SCL.tif'
    with rasterio.open(scl, 'w', driver='GTiff', height=2, width=2, count=1,
                       dtype='uint8', crs='EPSG:32643', transform=from_origin(500000, 4000000, 20, 20), nodata=0) as dst:
        dst.write(np.full((2, 2), 4, np.uint8), 1)
    metadata = tmp_path / 'MTD_MSIL2A.xml'; metadata.write_text(xml())
    return paths, scl, metadata


@pytest.mark.parametrize('bad_crs', ['EPSG:4326', 'EPSG:2263'])
@pytest.mark.parametrize('asset', ['bands', 'scl'])
def test_file_reader_rejects_non_metric_crs(raster_files, bad_crs, asset):
    import rasterio
    paths, scl, metadata = raster_files
    targets = list(paths.values()) if asset == 'bands' else [scl]
    for path in targets:
        with rasterio.open(path, 'r+') as src:
            src.crs = bad_crs
    with pytest.raises(ValueError, match='metre-based'):
        load_sentinel2(paths, scl, metadata)


def test_disk_band_and_scl_masks_survive_warp_and_crop(raster_files):
    import rasterio
    paths, scl, metadata = raster_files
    with rasterio.open(paths['B03'], 'r+') as src:
        mask = np.full((4, 4), 255, np.uint8); mask[2, 2] = 0
        src.write_mask(mask)
    with rasterio.open(scl, 'r+') as src:
        mask = np.full((2, 2), 255, np.uint8); mask[0, 0] = 0
        src.write_mask(mask)
    scene = load_sentinel2(paths, scl, metadata, window=(1, 1, 3, 3))
    np.testing.assert_array_equal(scene.valid_mask, [[False, True, True],
                                                  [True, False, True],
                                                  [True, True, True]])
    assert scene.X.shape == (7, 4)
    image = scene.map_labels(np.arange(7, dtype=np.int32))
    assert image[0, 0] == image[1, 1] == -1
    assert scene.metadata['crop_grid']['transform'][2] == 500010
    assert scene.metadata['window'] == [1, 1, 3, 3]
