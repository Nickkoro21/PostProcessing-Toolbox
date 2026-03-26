# -*- coding: utf-8 -*-
"""
_postproc_common.py — Shared functions for PostProcessing Toolbox
=================================================================
Master Thesis: Semantic Segmentation of Multispectral Drone Imagery
Sensor: MicaSense Altum-PT  |  ArcGIS Pro 3.6.2

Developed by Nikolaos Koroniadis
MSc Geography and Applied Geoinformatics, University of the Aegean
Remote Sensing & GIS Research Group (https://rsgis.aegean.gr/)

This module is imported by PostProcessing.pyt and provides the
common pipeline phases shared across Vehicle, Building (and future)
analysis tools.

Phase map:
    0  validate_inputs        — CRS, cell size, AOI, extent
    1  extract_class_pixels   — Con(VALUE = target, 1, 0)
    2  run_majority_filter    — MajorityFilter (FOUR/EIGHT/Skip)
    3  vectorize_binary       — RasterToPolygon → remove gridcode
    5  compute_geometry_mbr   — Area, Perimeter, MBR, L/W, Compactness
    6  enrich_attributes      — ID, ZonalStats (nDSM + thermal)
    8  export_results         — CopyFeatures, centroids, JSON log

Phases 4 and 7 are tool-specific:
    4  Vehicle: SimplifyPolygon  |  Building: RegularizeBuildingFootprint
    7  Vehicle: classify by area + L/W  |  Building: classify by area + nDSM
"""

import arcpy
import os
import time
import math
import json


# ═══════════════════════════════════════════════════════════════════════════════
# CRS UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_crs(out_crs_text):
    """
    Resolve an output CRS from the parameter's valueAsText string.

    ArcGIS Pro's GPCoordinateSystem parameter returns either a factory
    code (e.g. '2100' for ΕΓΣΑ 87) or a CRS name string via valueAsText.
    The previous implementation used str() on the SpatialReference object,
    which produced WKT that often failed to round-trip.

    Returns
    -------
    arcpy.SpatialReference or None
    """
    if not out_crs_text:
        return None

    # Attempt 1: factory code (integer EPSG / WKID)
    try:
        return arcpy.SpatialReference(int(out_crs_text))
    except (ValueError, TypeError):
        pass

    # Attempt 2: CRS name string (e.g. 'Greek_Grid')
    try:
        sr = arcpy.SpatialReference(out_crs_text)
        if sr.name and sr.name != "Unknown":
            return sr
    except Exception:
        pass

    # Attempt 3: WKT string (last resort)
    try:
        sr = arcpy.SpatialReference()
        sr.loadFromString(out_crs_text)
        if sr.name and sr.name != "Unknown":
            return sr
    except Exception:
        pass

    return None


def crs_match(a, b):
    """Compare two SpatialReference objects (handles factoryCode=0)."""
    if a.factoryCode > 0 and a.factoryCode == b.factoryCode:
        return True
    return a.name == b.name


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0: INPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_inputs(in_cls, in_ndsm, in_therm, in_aoi,
                    out_crs_text, cs_opt, cs_custom,
                    class_val, class_label, run_log):
    """
    Validate inputs, resolve CRS, compute cell size and AOI extent.

    Parameters
    ----------
    in_cls : str
        Path to classified raster (single-band, integer).
    in_ndsm : str
        Path to nDSM raster.
    in_therm : str or None
        Path to thermal raster (optional).
    in_aoi : str or None
        Path to AOI feature/raster (optional).
    out_crs_text : str or None
        Output CRS (valueAsText from GPCoordinateSystem parameter).
    cs_opt : str
        Cell size option string.
    cs_custom : float or None
        Custom cell size in meters (used only if cs_opt == 'Custom').
    class_val : int
        Target class value to extract.
    class_label : str
        Human-readable label for messages (e.g. 'Vehicle', 'Building').
    run_log : dict
        Mutable log dictionary — cell_size_used_m is written here.

    Returns
    -------
    dict with keys:
        tgt_sr    : arcpy.SpatialReference
        cell_size : float
        aoi_ext   : arcpy.Extent
        aoi_m2    : float
    """
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage("PHASE 0: Input Validation")
    arcpy.AddMessage("=" * 60)

    cls_r = arcpy.Raster(in_cls)
    ndsm_r = arcpy.Raster(in_ndsm)
    cls_sr = cls_r.spatialReference
    ndsm_sr = ndsm_r.spatialReference

    arcpy.AddMessage(
        f"  Classified: {cls_r.width}x{cls_r.height}, "
        f"cell={cls_r.meanCellWidth:.4f} m, "
        f"CRS={cls_sr.name}")
    arcpy.AddMessage(
        f"  {class_label} class value: {class_val}")
    arcpy.AddMessage(
        f"  nDSM:       {ndsm_r.width}x{ndsm_r.height}, "
        f"cell={ndsm_r.meanCellWidth:.4f} m, "
        f"CRS={ndsm_sr.name}")

    # -- Thermal --
    therm_cell = 0.0
    if in_therm:
        tr = arcpy.Raster(in_therm)
        therm_sr = tr.spatialReference
        arcpy.AddMessage(
            f"  Thermal:    {tr.width}x{tr.height}, "
            f"cell={tr.meanCellWidth:.4f} m, "
            f"CRS={therm_sr.name}")
        therm_cell = tr.meanCellWidth
        if therm_sr.name != cls_sr.name:
            arcpy.AddWarning(
                "  Thermal CRS differs — "
                "on-the-fly reprojection applied.")
        del tr
    else:
        arcpy.AddMessage("  Thermal:    not provided")

    # -- Output CRS (BUG FIX: resolve from valueAsText) --
    tgt_sr = resolve_crs(out_crs_text)
    if tgt_sr is None or tgt_sr.name == "Unknown":
        tgt_sr = cls_sr
    arcpy.AddMessage(f"  Output CRS: {tgt_sr.name}")

    # FIX F1: Warn if output CRS is Geographic (degrees, not meters)
    if tgt_sr.type == "Geographic":
        arcpy.AddWarning(
            "  Output CRS is Geographic (units: degrees). "
            "Area, perimeter, and MBR calculations will be "
            "in square degrees — results may be incorrect. "
            "Consider using a Projected CRS (e.g. EPSG 2100).")

    if not (crs_match(cls_sr, tgt_sr)
            and crs_match(ndsm_sr, tgt_sr)):
        arcpy.AddWarning("  CRS mismatch detected.")
    else:
        arcpy.AddMessage("  CRS check: PASSED")

    # -- Cell size --
    cells = [cls_r.meanCellWidth, ndsm_r.meanCellWidth]
    if therm_cell > 0:
        cells.append(therm_cell)

    if cs_opt == "Classified raster cell size":
        cs = cls_r.meanCellWidth
    elif cs_opt == "Maximum of inputs (coarsest)":
        cs = max(cells)
    elif cs_opt == "nDSM cell size":
        cs = ndsm_r.meanCellWidth
    elif cs_opt == "Custom":
        cs = cs_custom
    else:
        cs = cls_r.meanCellWidth
    arcpy.AddMessage(f"  Cell size: {cs:.4f} m")
    run_log["parameters"]["cell_size_used_m"] = round(cs, 6)

    # -- AOI --
    if in_aoi:
        aoi_ext = arcpy.Describe(in_aoi).extent
        arcpy.AddMessage("  AOI: user-defined")
    else:
        ce = cls_r.extent
        ne = ndsm_r.extent
        xn = max(ce.XMin, ne.XMin)
        yn = max(ce.YMin, ne.YMin)
        xx = min(ce.XMax, ne.XMax)
        yx = min(ce.YMax, ne.YMax)
        if xn >= xx or yn >= yx:
            raise ValueError(
                "No spatial overlap between classified raster and nDSM!")
        aoi_ext = arcpy.Extent(xn, yn, xx, yx)
        arcpy.AddMessage("  AOI: intersection of inputs")

    aw = aoi_ext.XMax - aoi_ext.XMin
    ah = aoi_ext.YMax - aoi_ext.YMin
    aoi_m2 = aw * ah
    arcpy.AddMessage(
        f"  AOI: {aw:.1f}x{ah:.1f} m ({aoi_m2:,.1f} m2)")

    # -- Set environment --
    # NOTE: outputCoordinateSystem is NOT set here. Setting it during
    # raster phases (1-3) causes "no valid statistics" errors when the
    # target CRS differs from the input CRS. The caller must invoke
    # set_output_crs() AFTER all raster operations (after Phase 3).
    arcpy.env.cellSize = cs
    arcpy.env.extent = aoi_ext
    if in_aoi:
        arcpy.env.mask = in_aoi
    del cls_r, ndsm_r

    return {
        "tgt_sr": tgt_sr,
        "cell_size": cs,
        "aoi_ext": aoi_ext,
        "aoi_m2": aoi_m2,
    }


def set_output_crs(tgt_sr):
    """
    Set arcpy.env.outputCoordinateSystem for polygon operations.

    Must be called AFTER all raster phases (1-3) are complete.
    Setting it earlier causes 'no valid statistics' errors when the
    target CRS differs from the input rasters' CRS, because ArcGIS
    attempts on-the-fly reprojection during raster algebra.
    """
    arcpy.env.outputCoordinateSystem = tgt_sr
    arcpy.AddMessage(f"  Output CRS applied: {tgt_sr.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: EXTRACT CLASS PIXELS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_class_pixels(in_cls, class_val, class_label, tmp_dir):
    """
    Create binary raster: 1 where pixel == class_val, 0 elsewhere.

    Returns
    -------
    str : path to binary raster (TIF)
    """
    arcpy.AddMessage("")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage(
        f"PHASE 1: Extract {class_label} pixels (value = {class_val})")
    arcpy.AddMessage("=" * 60)

    cls_ras = arcpy.Raster(in_cls)
    vbin = arcpy.sa.Con(cls_ras, 1, 0, f"VALUE = {class_val}")
    bin_path = os.path.join(tmp_dir, "bin_extract.tif")
    vbin.save(bin_path)
    del cls_ras, vbin

    # Ensure statistics exist (needed by downstream SetNull)
    arcpy.management.CalculateStatistics(bin_path)

    bin_r = arcpy.Raster(bin_path)
    arcpy.AddMessage(f"  Binary: {bin_r.width}x{bin_r.height}")
    del bin_r

    return bin_path


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: MAJORITY FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def run_majority_filter(bin_path, neighborhood, passes, tmp_dir):
    """
    Apply MajorityFilter to binary raster.

    Parameters
    ----------
    bin_path : str
        Path to binary raster.
    neighborhood : str
        'FOUR', 'EIGHT', or 'Skip'.
    passes : int
        Number of filter passes (1-5).
    tmp_dir : str
        Temp directory for intermediate outputs.

    Returns
    -------
    str : path to filtered raster (or original if skipped)
    """
    arcpy.AddMessage("")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage("PHASE 2: Morphological Filtering")
    arcpy.AddMessage("=" * 60)

    if neighborhood == "Skip":
        arcpy.AddMessage("  Skipped")
        return bin_path

    arcpy.AddMessage(f"  {neighborhood}, {passes} pass(es)")
    filt_r = arcpy.Raster(bin_path)
    for i in range(passes):
        arcpy.AddMessage(f"  Pass {i+1}/{passes}")
        filt_r = arcpy.sa.MajorityFilter(
            filt_r, neighborhood, "MAJORITY")
    out_path = os.path.join(tmp_dir, "majority.tif")
    filt_r.save(out_path)
    del filt_r

    # Statistics required by downstream SetNull — not always
    # auto-calculated after MajorityFilter save-to-disk.
    arcpy.management.CalculateStatistics(out_path)

    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: VECTORIZE
# ═══════════════════════════════════════════════════════════════════════════════

def vectorize_binary(raster_path, tmp_dir):
    """
    Convert binary raster to polygons (value=1 only).

    Returns
    -------
    tuple (str, int) : (path to raw polygon shapefile, feature count)
    """
    arcpy.AddMessage("")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage("PHASE 3: Vectorization")
    arcpy.AddMessage("=" * 60)

    vec_r = arcpy.Raster(raster_path)
    # Con with no false branch -> 0-pixels become NoData automatically.
    # Avoids SetNull which requires pre-computed raster statistics.
    vec_clean = arcpy.sa.Con(vec_r, vec_r, where_clause="VALUE > 0")
    raw_path = os.path.join(tmp_dir, "raw_poly.shp")
    arcpy.conversion.RasterToPolygon(
        vec_clean, raw_path, "NO_SIMPLIFY", "VALUE",
        "SINGLE_OUTER_PART")
    del vec_r, vec_clean

    n_raw = int(arcpy.management.GetCount(raw_path)[0])
    arcpy.AddMessage(f"  Raw polygons: {n_raw}")

    # Remove gridcode field if present
    if "gridcode" in [f.name for f in arcpy.ListFields(raw_path)]:
        arcpy.management.DeleteField(raw_path, ["gridcode"])

    return raw_path, n_raw


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: GEOMETRY & MBR
# ═══════════════════════════════════════════════════════════════════════════════

def compute_geometry_mbr(fc_path, tmp_dir):
    """
    Compute geometry attributes (Area, Perimeter) and Minimum Bounding
    Rectangle (Length, Width, Orientation, L/W ratio) for all features.

    Operates in-place on fc_path.
    """
    arcpy.AddMessage("")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage("PHASE 5: Geometry & MBR")
    arcpy.AddMessage("=" * 60)

    # Area + Perimeter
    arcpy.management.AddField(fc_path, "Area_m2", "DOUBLE")
    arcpy.management.AddField(fc_path, "Perim_m", "DOUBLE")
    arcpy.management.CalculateGeometryAttributes(
        fc_path,
        [["Area_m2", "AREA"],
         ["Perim_m", "PERIMETER_LENGTH"]],
        length_unit="METERS",
        area_unit="SQUARE_METERS")

    # Temp ID for MBR join
    arcpy.management.AddField(fc_path, "TmpID", "LONG")
    with arcpy.da.UpdateCursor(fc_path, ["TmpID", "OID@"]) as ucur:
        for row in ucur:
            row[0] = row[1]
            ucur.updateRow(row)

    # Minimum Bounding Rectangle
    mbr_path = os.path.join(tmp_dir, "mbr.shp")
    arcpy.management.MinimumBoundingGeometry(
        fc_path, mbr_path,
        "RECTANGLE_BY_WIDTH", "NONE", "", "MBG_FIELDS")
    arcpy.management.JoinField(
        fc_path, "TmpID", mbr_path, "TmpID",
        ["MBG_Width", "MBG_Length", "MBG_Orient"])

    # L/W ratio
    arcpy.management.AddField(fc_path, "LW_Ratio", "DOUBLE")
    with arcpy.da.UpdateCursor(
        fc_path, ["MBG_Length", "MBG_Width", "LW_Ratio"]
    ) as ucur:
        for row in ucur:
            lg = row[0] if row[0] else 0
            wd = row[1] if row[1] else 0
            row[2] = round(lg / wd, 2) if wd > 0 else 0
            ucur.updateRow(row)

    # Rename MBG_ fields -> cleaner names
    arcpy.management.AddField(fc_path, "MBR_W", "DOUBLE")
    arcpy.management.AddField(fc_path, "MBR_L", "DOUBLE")
    arcpy.management.AddField(fc_path, "MBR_Azim", "DOUBLE")
    arcpy.management.CalculateField(
        fc_path, "MBR_W", "!MBG_Width!", "PYTHON3")
    arcpy.management.CalculateField(
        fc_path, "MBR_L", "!MBG_Length!", "PYTHON3")
    arcpy.management.CalculateField(
        fc_path, "MBR_Azim", "!MBG_Orient!", "PYTHON3")
    arcpy.management.DeleteField(
        fc_path,
        ["MBG_Width", "MBG_Length", "MBG_Orient", "TmpID"])

    arcpy.AddMessage("  Phase 5 complete")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: ATTRIBUTE ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_join_zonal_mean(fc_path, id_field, zonal_dbf,
                          target_field):
    """
    Join the MEAN column from a ZonalStatisticsAsTable output
    into fc_path, renaming it to target_field.

    FIX BUG 2: If a previous MEAN field was not fully cleaned up
    (schema lock, race condition), JoinField may create MEAN_1
    instead.  This helper detects the collision and handles it
    gracefully.

    Parameters
    ----------
    fc_path : str
        Feature class to enrich (modified in-place).
    id_field : str
        Join key (e.g. 'VehicleID').
    zonal_dbf : str
        DBF output from ZonalStatisticsAsTable.
    target_field : str
        Destination field name (e.g. 'MEAN_nDSM', 'MEAN_Therm').
    """
    arcpy.management.AddField(fc_path, target_field, "DOUBLE")

    # Join — ArcGIS may append _1 if MEAN already exists
    arcpy.management.JoinField(
        fc_path, id_field,
        zonal_dbf, id_field.upper(), ["MEAN"])

    # Detect which field was actually added
    current_fields = [f.name for f in arcpy.ListFields(fc_path)]

    if "MEAN" in current_fields:
        source_field = "MEAN"
    elif "MEAN_1" in current_fields:
        arcpy.AddWarning(
            f"  MEAN field collision detected — "
            f"using MEAN_1 for {target_field}")
        source_field = "MEAN_1"
    elif "MEAN_12" in current_fields:
        source_field = "MEAN_12"
    else:
        arcpy.AddWarning(
            f"  Could not find MEAN field for {target_field} "
            f"— values will be NULL")
        return

    arcpy.management.CalculateField(
        fc_path, target_field, f"!{source_field}!", "PYTHON3")
    try:
        arcpy.management.DeleteField(fc_path, [source_field])
    except Exception:
        arcpy.AddWarning(
            f"  Could not delete {source_field} field — "
            f"will remain in attribute table")


def enrich_attributes(fc_path, in_ndsm, in_therm,
                      id_field, tmp_dir):
    """
    Enrich feature class with sequential ID, Compactness,
    and zonal statistics (mean nDSM, mean thermal).

    Parameters
    ----------
    fc_path : str
        Path to feature class (modified in-place).
    in_ndsm : str
        Path to nDSM raster.
    in_therm : str or None
        Path to thermal raster (optional).
    id_field : str
        Name for the sequential ID field (e.g. 'VehicleID', 'BuildingID').
    tmp_dir : str
        Temp directory for zonal stats tables.

    Returns
    -------
    int : count of features with NULL nDSM values
    """
    arcpy.AddMessage("")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage("PHASE 6: Attribute Enrichment")
    arcpy.AddMessage("=" * 60)

    # -- Sequential ID --
    arcpy.management.AddField(fc_path, id_field, "LONG")
    fid = 1
    with arcpy.da.UpdateCursor(fc_path, [id_field]) as ucur:
        for row in ucur:
            row[0] = fid
            fid += 1
            ucur.updateRow(row)

    # -- Compactness (Polsby-Popper: 4piA/P^2) --
    arcpy.management.AddField(fc_path, "Compact", "DOUBLE")
    with arcpy.da.UpdateCursor(
        fc_path, ["Area_m2", "Perim_m", "Compact"]
    ) as ucur:
        for row in ucur:
            a = row[0] if row[0] else 0
            p = row[1] if row[1] else 0
            row[2] = (round((4 * math.pi * a) / (p ** 2), 4)
                      if p > 0 else 0)
            ucur.updateRow(row)

    # -- Zonal Statistics: nDSM --
    # FIX BUG 2: Use _safe_join_zonal_mean to handle MEAN field
    # collisions between nDSM and thermal joins.
    arcpy.AddMessage("  Computing nDSM zonal stats...")
    zh_path = os.path.join(tmp_dir, "zonal_ndsm.dbf")
    arcpy.sa.ZonalStatisticsAsTable(
        fc_path, id_field,
        arcpy.Raster(in_ndsm), zh_path, "DATA", "MEAN")
    _safe_join_zonal_mean(fc_path, id_field, zh_path, "MEAN_nDSM")

    # Count NULLs
    n_null_h = 0
    with arcpy.da.SearchCursor(fc_path, ["MEAN_nDSM"]) as scur:
        for row in scur:
            if row[0] is None:
                n_null_h += 1
    if n_null_h > 0:
        arcpy.AddWarning(f"  {n_null_h} feature(s) with NULL nDSM")
    else:
        arcpy.AddMessage("  nDSM: OK")

    # -- Zonal Statistics: Thermal --
    if in_therm:
        arcpy.AddMessage("  Computing thermal zonal stats...")
        zt_path = os.path.join(tmp_dir, "zonal_therm.dbf")
        arcpy.sa.ZonalStatisticsAsTable(
            fc_path, id_field,
            arcpy.Raster(in_therm), zt_path, "DATA", "MEAN")
        _safe_join_zonal_mean(
            fc_path, id_field, zt_path, "MEAN_Therm")
        arcpy.AddMessage("  Thermal: complete")
    else:
        arcpy.AddMessage("  Thermal: skipped")

    return n_null_h


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 8: EXPORT & SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def export_results(enriched_path, output_dir, prefix, do_centroids,
                   run_log, in_aoi, aoi_m2, class_val, class_label,
                   n_raw, lost_simp, n_total, n_accepted,
                   reject_counts, n_null_h, tgt_sr, t0):
    """
    Copy final features to output folder, generate centroids,
    print summary, and write JSON log.

    Parameters
    ----------
    enriched_path : str
        Path to enriched feature class with Class/Reason fields.
    output_dir : str
        Root output folder (subfolder '{prefix}s' is created inside).
    prefix : str
        Output name prefix: 'vehicle' or 'building'.
    do_centroids : bool
        Whether to generate centroid point shapefile.
    run_log : dict
        Mutable log dictionary.
    in_aoi : str or None
        AOI input path (for study area computation).
    aoi_m2 : float
        Bounding-box area in m2 (fallback if no AOI polygon).
    class_val : int
        Target class value.
    class_label : str
        Human-readable label ('Vehicle', 'Building').
    n_raw : int
        Number of raw polygons before simplification/regularization.
    lost_simp : int
        Features lost during simplification/regularization.
    n_total : int
        Total features after simplification/regularization.
    n_accepted : int
        Accepted features count.
    reject_counts : dict
        Keys are reason strings, values are counts.
    n_null_h : int
        Features with NULL nDSM.
    tgt_sr : arcpy.SpatialReference
        Output coordinate system.
    t0 : float
        Start time (from time.time()).

    Returns
    -------
    tuple (str, str or None) : (results_path, points_path)
    """
    arcpy.AddMessage("")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage("PHASE 8: Export & Summary")
    arcpy.AddMessage("=" * 60)

    # Subfolder: 'vehicles' or 'buildings'
    sub_dir = os.path.join(output_dir, f"{prefix}s")
    os.makedirs(sub_dir, exist_ok=True)

    results_name = f"{prefix}_results.shp"
    points_name = f"{prefix}_points.shp"
    log_name = f"{prefix}_log.json"

    final_path = os.path.join(sub_dir, results_name)
    arcpy.management.CopyFeatures(enriched_path, final_path)

    pts_path = None
    if do_centroids and n_total > 0:
        pts_path = os.path.join(sub_dir, points_name)
        arcpy.management.FeatureToPoint(
            final_path, pts_path, "CENTROID")
        arcpy.AddMessage(
            f"  Centroids: "
            f"{int(arcpy.management.GetCount(pts_path)[0])}")

    # -- Study area --
    total_t = time.time() - t0
    if in_aoi:
        desc = arcpy.Describe(in_aoi)
        if hasattr(desc, "shapeType"):
            with arcpy.da.SearchCursor(
                in_aoi, ["SHAPE@AREA"]
            ) as scur:
                study_a = sum(r[0] for r in scur)
        else:
            study_a = aoi_m2
    else:
        study_a = aoi_m2

    # -- Summary messages --
    n_rej = sum(reject_counts.values())
    n_all = n_accepted + n_rej  # total classified features
    arcpy.AddMessage("")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage(f"{class_label.upper()} ANALYSIS COMPLETE")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage(
        f"  Study area:   {study_a:,.1f} m2 "
        f"({study_a / 10000:,.4f} ha)")
    arcpy.AddMessage(f"  Class value:  {class_val}")
    arcpy.AddMessage(f"  Raw:          {n_raw}")
    arcpy.AddMessage(f"  Processed:    {n_total}")
    arcpy.AddMessage("")
    arcpy.AddMessage("  ── Classification Results ──")
    pct_acc = (100.0 * n_accepted / n_all) if n_all > 0 else 0.0
    pct_rej = (100.0 * n_rej / n_all) if n_all > 0 else 0.0
    arcpy.AddMessage(
        f"  {class_label}:      {n_accepted:>5}  "
        f"({pct_acc:5.1f}%)")
    arcpy.AddMessage(
        f"  Rejected:       {n_rej:>5}  "
        f"({pct_rej:5.1f}%)")
    if n_rej > 0:
        arcpy.AddMessage("    Breakdown:")
        for reason, count in reject_counts.items():
            if count > 0:
                pct_r = (100.0 * count / n_all)
                arcpy.AddMessage(
                    f"      {reason:<20} {count:>5}  "
                    f"({pct_r:5.1f}%)")
    arcpy.AddMessage(
        f"  Total:          {n_all:>5}  (100.0%)")
    arcpy.AddMessage("")
    if n_accepted > 0 and study_a > 0:
        arcpy.AddMessage(
            f"  Density:      "
            f"{n_accepted / (study_a / 10000):.1f} {prefix}/ha")
    arcpy.AddMessage(f"  Output:       {final_path}")
    arcpy.AddMessage(f"  Time:         {total_t:.1f} s")
    arcpy.AddMessage("=" * 60)

    # -- Update log --
    run_log["results"].update({
        "study_area_m2": round(study_a, 2),
        "study_area_ha": round(study_a / 10000, 4),
        "raw_polygons": n_raw,
        "lost_at_processing": lost_simp,
        "total_features": n_total,
        f"accepted_{prefix}s": n_accepted,
        "rejected_total": n_rej,
        "null_ndsm_count": n_null_h,
        "output_crs": tgt_sr.name,
        "elapsed_seconds": round(total_t, 1),
    })
    for reason, count in reject_counts.items():
        run_log["results"][f"rejected_{reason}"] = count

    run_log["outputs"] = {
        f"{prefix}_results": final_path,
        f"{prefix}_points": pts_path,
    }

    write_log(run_log, sub_dir, log_name)

    return final_path, pts_path


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def write_log(log_dict, output_dir, filename):
    """Write run log as JSON."""
    log_path = os.path.join(output_dir, filename)
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_dict, f, indent=2, ensure_ascii=False)
        arcpy.AddMessage(f"  Log: {log_path}")
    except Exception as e:
        arcpy.AddWarning(f"  Could not write log: {e}")


def init_run(out_folder, prefix, in_cls, class_val, in_ndsm,
             in_therm, in_aoi, params_dict):
    """
    Initialize a tool run: create output directories, clean old files,
    set up temp folder and run_log dictionary.

    Parameters
    ----------
    out_folder : str
        User-specified output folder.
    prefix : str
        'vehicle' or 'building'.
    in_cls, class_val, in_ndsm, in_therm, in_aoi :
        Input paths / values for the log.
    params_dict : dict
        Tool parameters for the log.

    Returns
    -------
    tuple (str, str, dict, float) :
        (sub_dir, tmp_dir, run_log, t0)
    """
    t0 = time.time()
    run_id = time.strftime("%Y%m%d_%H%M%S")

    sub_dir = os.path.join(out_folder, f"{prefix}s")
    os.makedirs(sub_dir, exist_ok=True)

    # Clean old temp directories
    for item in os.listdir(sub_dir):
        ip = os.path.join(sub_dir, item)
        if item.startswith("_temp") and os.path.isdir(ip):
            try:
                for f in os.listdir(ip):
                    fp = os.path.join(ip, f)
                    if arcpy.Exists(fp):
                        arcpy.management.Delete(fp)
                os.rmdir(ip)
            except Exception:
                pass

    # Clean old outputs
    for old in [f"{prefix}_results.shp", f"{prefix}_points.shp",
                f"{prefix}_log.json"]:
        op = os.path.join(sub_dir, old)
        if old.endswith(".json"):
            if os.path.exists(op):
                os.remove(op)
        elif arcpy.Exists(op):
            try:
                arcpy.management.Delete(op)
            except Exception:
                pass

    tmp_dir = os.path.join(sub_dir, f"_temp_{run_id}")
    os.makedirs(tmp_dir)

    run_log = {
        "run_id": run_id,
        "tool": f"{prefix}_analysis",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {
            "classified_raster": in_cls,
            f"{prefix}_class_value": class_val,
            "ndsm_raster": in_ndsm,
            "thermal_raster": in_therm,
            "aoi": in_aoi,
        },
        "parameters": dict(params_dict),
        "results": {},
    }

    return sub_dir, tmp_dir, run_log, t0


def cleanup_temp(tmp_dir):
    """Remove temporary directory and all files inside."""
    arcpy.AddMessage("\nCleaning temp...")
    try:
        for f in os.listdir(tmp_dir):
            fp = os.path.join(tmp_dir, f)
            if arcpy.Exists(fp):
                arcpy.management.Delete(fp)
        os.rmdir(tmp_dir)
        arcpy.AddMessage("  Done")
    except Exception as e:
        arcpy.AddMessage(f"  Partial: {e}")


def reset_environment():
    """Reset arcpy environment settings after a tool run."""
    arcpy.CheckInExtension("Spatial")
    arcpy.CheckInExtension("3D")
    arcpy.env.outputCoordinateSystem = None
    arcpy.env.cellSize = None
    arcpy.env.extent = None
    arcpy.env.mask = None
