# -*- coding: utf-8 -*-
"""
_vehicle_core.py — Vehicle Analysis pipeline
=============================================
Master Thesis: Semantic Segmentation of Multispectral Drone Imagery
Sensor: MicaSense Altum-PT  |  ArcGIS Pro 3.6.2

Developed by Nikolaos Koroniadis
MSc Geography and Applied Geoinformatics, University of the Aegean
Remote Sensing & GIS Research Group (https://rsgis.aegean.gr/)

Vehicle-specific phases:
    4  simplify_polygons  — SimplifyPolygon (POINT_REMOVE)
    7  classify_vehicles  — Accept/reject by area + L/W ratio

Imported by PostProcessing.pyt → VehicleAnalysis.execute()
"""

import arcpy
import os

import _postproc_common as common


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: SIMPLIFY (Vehicle-specific)
# ═══════════════════════════════════════════════════════════════════════════════

def simplify_polygons(raw_path, simp_tol, tmp_dir):
    """
    Simplify raw vehicle polygons using POINT_REMOVE algorithm.

    Parameters
    ----------
    raw_path : str
        Path to raw polygon shapefile from vectorization.
    simp_tol : float
        Simplification tolerance in meters.
    tmp_dir : str
        Temp directory for intermediate outputs.

    Returns
    -------
    tuple (str, int, int) : (simplified_path, n_after, n_lost)
        Returns (None, 0, n_raw) if all polygons collapse.
    """
    arcpy.AddMessage("")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage(f"PHASE 4: Simplify (tol = {simp_tol} m)")
    arcpy.AddMessage("=" * 60)

    simp_path = os.path.join(tmp_dir, "v_simp.shp")
    arcpy.cartography.SimplifyPolygon(
        raw_path, simp_path, "POINT_REMOVE", simp_tol,
        minimum_area="0 SquareMeters",
        error_option="RESOLVE_ERRORS",
        collapsed_point_option="NO_KEEP")

    n_raw = int(arcpy.management.GetCount(raw_path)[0])
    n_simp = int(arcpy.management.GetCount(simp_path)[0])
    lost = n_raw - n_simp
    arcpy.AddMessage(
        f"  After simplify: {n_simp} ({lost} collapsed)")

    if n_simp == 0:
        arcpy.AddWarning("All polygons collapsed.")
        return None, 0, lost

    return simp_path, n_simp, lost


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7: CLASSIFICATION (Vehicle-specific)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_vehicles(fc_path, min_area, max_area, min_lw):
    """
    Classify features as 'Vehicle' or 'Rejected possible vehicle'
    based on area and L/W ratio thresholds.

    Adds 'Class' and 'Reason' fields. Operates in-place.

    Parameters
    ----------
    fc_path : str
        Path to enriched feature class.
    min_area : float
        Minimum vehicle area (m²).
    max_area : float
        Maximum vehicle area (m²).
    min_lw : float
        Minimum Length/Width ratio.

    Returns
    -------
    tuple (int, dict) : (n_accepted, reject_counts)
        reject_counts = {'area_small': n, 'area_large': n, 'lw': n}
    """
    arcpy.AddMessage("")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage("PHASE 7: Classification")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage(
        f"  Area: {min_area}–{max_area} m², L/W >= {min_lw}")

    arcpy.management.AddField(
        fc_path, "Class", "TEXT", field_length=30)
    arcpy.management.AddField(
        fc_path, "Reason", "TEXT", field_length=30)

    n_accepted = n_small = n_large = n_rej_lw = 0

    with arcpy.da.UpdateCursor(
        fc_path,
        ["Area_m2", "LW_Ratio", "Class", "Reason"]
    ) as ucur:
        for row in ucur:
            area = row[0] if row[0] else 0
            lw = row[1] if row[1] else 0

            if area < min_area:
                row[2] = "Rejected possible vehicle"
                row[3] = "Area_below_min"
                n_small += 1
            elif area > max_area:
                row[2] = "Rejected possible vehicle"
                row[3] = "Area_above_max"
                n_large += 1
            elif lw < min_lw:
                row[2] = "Rejected possible vehicle"
                row[3] = "LW_below_min"
                n_rej_lw += 1
            else:
                row[2] = "Vehicle"
                row[3] = ""
                n_accepted += 1
            ucur.updateRow(row)

    reject_counts = {
        "area_small": n_small,
        "area_large": n_large,
        "lw": n_rej_lw,
    }
    n_rej = n_small + n_large + n_rej_lw

    arcpy.AddMessage(f"  Vehicle:  {n_accepted}")
    arcpy.AddMessage(
        f"  Rejected: {n_rej} "
        f"(area:{n_small + n_large}, L/W:{n_rej_lw})")

    return n_accepted, reject_counts


# ═══════════════════════════════════════════════════════════════════════════════
# FULL VEHICLE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_vehicle_pipeline(in_cls, veh_val, in_ndsm, in_therm,
                         in_aoi, out_crs_text, cs_opt, cs_custom,
                         min_area, max_area, min_lw,
                         maj_nbr, maj_passes, simp_tol,
                         do_centroids, out_folder):
    """
    Execute the complete vehicle post-processing pipeline.

    Called by VehicleAnalysis.execute() in PostProcessing.pyt.

    Parameters
    ----------
    in_cls : str          — Classified raster path
    veh_val : int         — Vehicle class value
    in_ndsm : str         — nDSM raster path
    in_therm : str|None   — Thermal raster path (optional)
    in_aoi : str|None     — AOI feature/raster path (optional)
    out_crs_text : str|None — Output CRS (valueAsText)
    cs_opt : str          — Cell size option
    cs_custom : float|None — Custom cell size
    min_area : float      — Min vehicle area (m²)
    max_area : float      — Max vehicle area (m²)
    min_lw : float        — Min L/W ratio
    maj_nbr : str         — Majority neighborhood (FOUR/EIGHT/Skip)
    maj_passes : int      — Majority filter passes
    simp_tol : float      — Simplify tolerance (m)
    do_centroids : bool   — Generate centroid points
    out_folder : str      — Output folder

    Returns
    -------
    tuple (str, str|None) : (results_shapefile, points_shapefile)
    """
    PREFIX = "vehicle"
    LABEL = "Vehicle"
    ID_FIELD = "VehicleID"

    # ── INIT ──────────────────────────────────────────────────────
    arcpy.env.overwriteOutput = True
    arcpy.CheckOutExtension("Spatial")
    arcpy.CheckOutExtension("3D")

    params_dict = {
        "cell_size_option": cs_opt,
        "cell_size_custom": cs_custom,
        "min_area_m2": min_area,
        "max_area_m2": max_area,
        "min_lw_ratio": min_lw,
        "majority_neighborhood": maj_nbr,
        "majority_passes": (maj_passes if maj_nbr != "Skip" else 0),
        "simplify_tolerance_m": simp_tol,
        "generate_centroids": do_centroids,
        "output_folder": out_folder,
    }

    sub_dir, tmp_dir, run_log, t0 = common.init_run(
        out_folder, PREFIX, in_cls, veh_val,
        in_ndsm, in_therm, in_aoi, params_dict)

    try:
        # ── PHASE 0: VALIDATE ────────────────────────────────────
        env = common.validate_inputs(
            in_cls, in_ndsm, in_therm, in_aoi,
            out_crs_text, cs_opt, cs_custom,
            veh_val, LABEL, run_log)

        # ── PHASE 1: EXTRACT ─────────────────────────────────────
        bin_path = common.extract_class_pixels(
            in_cls, veh_val, LABEL, tmp_dir)

        # ── PHASE 2: MAJORITY FILTER ─────────────────────────────
        filt_path = common.run_majority_filter(
            bin_path, maj_nbr, maj_passes, tmp_dir)

        # ── PHASE 3: VECTORIZE ───────────────────────────────────
        raw_path, n_raw = common.vectorize_binary(filt_path, tmp_dir)

        if n_raw == 0:
            arcpy.AddWarning(
                f"No pixels with value {veh_val} found.")
            empty_path = os.path.join(sub_dir, "vehicle_results.shp")
            arcpy.management.CopyFeatures(raw_path, empty_path)
            common.write_log(run_log, sub_dir, "vehicle_log.json")
            return empty_path, None

        # ── PHASE 4: SIMPLIFY (vehicle-specific) ─────────────────
        simp_path, n_simp, lost_simp = simplify_polygons(
            raw_path, simp_tol, tmp_dir)

        if simp_path is None:
            empty_path = os.path.join(sub_dir, "vehicle_results.shp")
            arcpy.management.CopyFeatures(raw_path, empty_path)
            common.write_log(run_log, sub_dir, "vehicle_log.json")
            return empty_path, None

        # ── PHASE 5: GEOMETRY & MBR ──────────────────────────────
        common.compute_geometry_mbr(simp_path, tmp_dir)

        # ── PHASE 6: ENRICHMENT ──────────────────────────────────
        enriched_path = os.path.join(tmp_dir, "v_enriched.shp")
        arcpy.management.CopyFeatures(simp_path, enriched_path)

        n_null_h = common.enrich_attributes(
            enriched_path, in_ndsm, in_therm, ID_FIELD, tmp_dir)

        # ── PHASE 7: CLASSIFY (vehicle-specific) ─────────────────
        n_accepted, reject_counts = classify_vehicles(
            enriched_path, min_area, max_area, min_lw)

        # ── Set output CRS now (all raster operations complete) ──
        common.set_output_crs(env["tgt_sr"])

        # ── PHASE 8: EXPORT ──────────────────────────────────────
        final_path, pts_path = common.export_results(
            enriched_path, out_folder, PREFIX, do_centroids,
            run_log, in_aoi, env["aoi_m2"],
            veh_val, LABEL,
            n_raw, lost_simp, n_simp, n_accepted,
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
