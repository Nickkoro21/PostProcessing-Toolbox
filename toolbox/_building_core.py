# -*- coding: utf-8 -*-
"""
_building_core.py — Building Analysis pipeline
================================================
Master Thesis: Semantic Segmentation of Multispectral Drone Imagery
Sensor: MicaSense Altum-PT  |  ArcGIS Pro 3.6.2

Developed by Nikolaos Koroniadis
MSc Geography and Applied Geoinformatics, University of the Aegean
Remote Sensing & GIS Research Group (https://rsgis.aegean.gr/)

Building-specific phases:
    4  regularize_polygons    — RegularizeBuildingFootprint
    7  classify_buildings     — Accept/reject by area + min nDSM

Imported by PostProcessing.pyt → BuildingAnalysis.execute()
"""

import arcpy
import os

import _postproc_common as common


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: REGULARIZE (Building-specific)
# ═══════════════════════════════════════════════════════════════════════════════

def regularize_polygons(raw_path, method, tolerance, tmp_dir):
    """
    Regularize raw building polygons to enforce straight edges
    and right angles, producing clean building footprints.

    Parameters
    ----------
    raw_path : str
        Path to raw polygon shapefile from vectorization.
    method : str
        Regularization method:
        'RIGHT_ANGLES', 'RIGHT_ANGLES_AND_DIAGONALS', or 'ANY_ANGLE'.
    tolerance : float
        Regularization tolerance in meters.
    tmp_dir : str
        Temp directory for intermediate outputs.

    Returns
    -------
    tuple (str, int, int) : (regularized_path, n_after, n_lost)
        Returns (None, 0, n_raw) if all polygons collapse.
    """
    arcpy.AddMessage("")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage(
        f"PHASE 4: Regularize Building Footprints "
        f"({method}, tol = {tolerance} m)")
    arcpy.AddMessage("=" * 60)

    reg_path = os.path.join(tmp_dir, "b_reg.shp")
    arcpy.ddd.RegularizeBuildingFootprint(
        raw_path, reg_path,
        method=method,
        tolerance=tolerance,
        densification=tolerance * 0.8,
        precision=0.25)

    n_raw = int(arcpy.management.GetCount(raw_path)[0])
    n_reg = int(arcpy.management.GetCount(reg_path)[0])
    lost = n_raw - n_reg
    arcpy.AddMessage(
        f"  After regularize: {n_reg} ({lost} collapsed)")

    if n_reg == 0:
        arcpy.AddWarning("All polygons collapsed during regularization.")
        return None, 0, lost

    return reg_path, n_reg, lost


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7: CLASSIFICATION (Building-specific)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_buildings(fc_path, min_area, max_area,
                       min_ndsm, reject_null_ndsm):
    """
    Classify features as 'Building' or 'Rejected possible building'
    based on area and nDSM height thresholds.

    Adds 'Class' and 'Reason' fields. Operates in-place.

    Parameters
    ----------
    fc_path : str
        Path to enriched feature class.
    min_area : float
        Minimum building area (m²).
    max_area : float
        Maximum building area (m²).
    min_ndsm : float
        Minimum mean nDSM height (m). Features below are rejected.
    reject_null_ndsm : bool
        If True, features with NULL nDSM are rejected.
        If False, they are kept as Building with a warning.

    Returns
    -------
    tuple (int, dict) : (n_accepted, reject_counts)
    """
    arcpy.AddMessage("")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage("PHASE 7: Classification")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage(
        f"  Area: {min_area}–{max_area} m², "
        f"min nDSM >= {min_ndsm} m")
    arcpy.AddMessage(
        f"  NULL nDSM policy: "
        f"{'reject' if reject_null_ndsm else 'keep with warning'}")

    arcpy.management.AddField(
        fc_path, "Class", "TEXT", field_length=30)
    arcpy.management.AddField(
        fc_path, "Reason", "TEXT", field_length=30)

    n_accepted = 0
    n_small = 0
    n_large = 0
    n_low_ndsm = 0
    n_null_ndsm = 0
    n_null_kept = 0

    with arcpy.da.UpdateCursor(
        fc_path,
        ["Area_m2", "MEAN_nDSM", "Class", "Reason"]
    ) as ucur:
        for row in ucur:
            area = row[0] if row[0] else 0
            ndsm = row[1]  # can be None

            if area < min_area:
                row[2] = "Rejected possible building"
                row[3] = "Area_below_min"
                n_small += 1
            elif area > max_area:
                row[2] = "Rejected possible building"
                row[3] = "Area_above_max"
                n_large += 1
            elif ndsm is None:
                if reject_null_ndsm:
                    row[2] = "Rejected possible building"
                    row[3] = "nDSM_null"
                    n_null_ndsm += 1
                else:
                    row[2] = "Building"
                    row[3] = "nDSM_null_kept"
                    n_accepted += 1
                    n_null_kept += 1
            elif ndsm < min_ndsm:
                row[2] = "Rejected possible building"
                row[3] = "nDSM_below_min"
                n_low_ndsm += 1
            else:
                row[2] = "Building"
                row[3] = ""
                n_accepted += 1
            ucur.updateRow(row)

    reject_counts = {
        "area_small": n_small,
        "area_large": n_large,
        "nDSM_below_min": n_low_ndsm,
        "nDSM_null": n_null_ndsm,
    }
    n_rej = n_small + n_large + n_low_ndsm + n_null_ndsm

    arcpy.AddMessage(f"  Building: {n_accepted}")
    if n_null_kept > 0:
        arcpy.AddWarning(
            f"  ({n_null_kept} accepted with NULL nDSM)")
    arcpy.AddMessage(
        f"  Rejected: {n_rej} "
        f"(area:{n_small + n_large}, "
        f"nDSM_low:{n_low_ndsm}, nDSM_null:{n_null_ndsm})")

    return n_accepted, reject_counts


# ═══════════════════════════════════════════════════════════════════════════════
# FULL BUILDING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_building_pipeline(in_cls, bld_val, in_ndsm, in_therm,
                          in_aoi, out_crs_text, cs_opt, cs_custom,
                          min_area, max_area, min_ndsm,
                          reject_null_ndsm,
                          maj_nbr, maj_passes,
                          reg_method, reg_tolerance,
                          do_centroids, out_folder):
    """
    Execute the complete building post-processing pipeline.

    Called by BuildingAnalysis.execute() in PostProcessing.pyt.

    Parameters
    ----------
    in_cls : str          — Classified raster path
    bld_val : int         — Building class value
    in_ndsm : str         — nDSM raster path
    in_therm : str|None   — Thermal raster path (optional)
    in_aoi : str|None     — AOI feature/raster path (optional)
    out_crs_text : str|None — Output CRS (valueAsText)
    cs_opt : str          — Cell size option
    cs_custom : float|None — Custom cell size
    min_area : float      — Min building area (m²)
    max_area : float      — Max building area (m²)
    min_ndsm : float      — Min mean nDSM height (m)
    reject_null_ndsm : bool — Reject features with NULL nDSM
    maj_nbr : str         — Majority neighborhood (FOUR/EIGHT/Skip)
    maj_passes : int      — Majority filter passes
    reg_method : str      — Regularize method
    reg_tolerance : float — Regularize tolerance (m)
    do_centroids : bool   — Generate centroid points
    out_folder : str      — Output folder

    Returns
    -------
    tuple (str, str|None) : (results_shapefile, points_shapefile)
    """
    PREFIX = "building"
    LABEL = "Building"
    ID_FIELD = "BuildingID"

    # ── INIT ──────────────────────────────────────────────────────
    arcpy.env.overwriteOutput = True
    arcpy.CheckOutExtension("Spatial")
    arcpy.CheckOutExtension("3D")

    params_dict = {
        "cell_size_option": cs_opt,
        "cell_size_custom": cs_custom,
        "min_area_m2": min_area,
        "max_area_m2": max_area,
        "min_ndsm_m": min_ndsm,
        "reject_null_ndsm": reject_null_ndsm,
        "majority_neighborhood": maj_nbr,
        "majority_passes": (maj_passes if maj_nbr != "Skip" else 0),
        "regularize_method": reg_method,
        "regularize_tolerance_m": reg_tolerance,
        "generate_centroids": do_centroids,
        "output_folder": out_folder,
    }

    sub_dir, tmp_dir, run_log, t0 = common.init_run(
        out_folder, PREFIX, in_cls, bld_val,
        in_ndsm, in_therm, in_aoi, params_dict)

    try:
        # ── PHASE 0: VALIDATE ────────────────────────────────────
        env = common.validate_inputs(
            in_cls, in_ndsm, in_therm, in_aoi,
            out_crs_text, cs_opt, cs_custom,
            bld_val, LABEL, run_log)

        # ── PHASE 1: EXTRACT ─────────────────────────────────────
        bin_path = common.extract_class_pixels(
            in_cls, bld_val, LABEL, tmp_dir)

        # ── PHASE 2: MAJORITY FILTER ─────────────────────────────
        filt_path = common.run_majority_filter(
            bin_path, maj_nbr, maj_passes, tmp_dir)

        # ── PHASE 3: VECTORIZE ───────────────────────────────────
        raw_path, n_raw = common.vectorize_binary(filt_path, tmp_dir)

        if n_raw == 0:
            arcpy.AddWarning(
                f"No pixels with value {bld_val} found.")
            empty_path = os.path.join(sub_dir, "building_results.shp")
            arcpy.management.CopyFeatures(raw_path, empty_path)
            common.write_log(run_log, sub_dir, "building_log.json")
            return empty_path, None

        # ── PHASE 4: REGULARIZE (building-specific) ──────────────
        reg_path, n_reg, lost_reg = regularize_polygons(
            raw_path, reg_method, reg_tolerance, tmp_dir)

        if reg_path is None:
            empty_path = os.path.join(sub_dir, "building_results.shp")
            arcpy.management.CopyFeatures(raw_path, empty_path)
            common.write_log(run_log, sub_dir, "building_log.json")
            return empty_path, None

        # ── PHASE 5: GEOMETRY & MBR ──────────────────────────────
        common.compute_geometry_mbr(reg_path, tmp_dir)

        # ── PHASE 6: ENRICHMENT ──────────────────────────────────
        enriched_path = os.path.join(tmp_dir, "b_enriched.shp")
        arcpy.management.CopyFeatures(reg_path, enriched_path)

        n_null_h = common.enrich_attributes(
            enriched_path, in_ndsm, in_therm, ID_FIELD, tmp_dir)

        # ── PHASE 7: CLASSIFY (building-specific) ────────────────
        n_accepted, reject_counts = classify_buildings(
            enriched_path, min_area, max_area,
            min_ndsm, reject_null_ndsm)

        # ── Set output CRS (all raster operations complete) ──────
        common.set_output_crs(env["tgt_sr"])

        # ── PHASE 8: EXPORT ──────────────────────────────────────
        final_path, pts_path = common.export_results(
            enriched_path, out_folder, PREFIX, do_centroids,
            run_log, in_aoi, env["aoi_m2"],
            bld_val, LABEL,
            n_raw, lost_reg, n_reg, n_accepted,
            reject_counts, n_null_h, env["tgt_sr"], t0)

        return final_path, pts_path

    except Exception as e:
        arcpy.AddError(f"Failed: {str(e)}")
        import traceback
        arcpy.AddError(traceback.format_exc())
        raise

    finally:
        common.cleanup_temp(tmp_dir)
        common.reset_environment()
