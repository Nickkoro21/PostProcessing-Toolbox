# -*- coding: utf-8 -*-
"""
_tree_core.py — Tree Analysis pipeline
========================================
Master Thesis: Semantic Segmentation of Multispectral Drone Imagery
Sensor: MicaSense Altum-PT  |  ArcGIS Pro 3.6.2

Developed by Nikolaos Koroniadis
MSc Geography and Applied Geoinformatics, University of the Aegean
Remote Sensing & GIS Research Group (https://rsgis.aegean.gr/)

Tree-specific phases:
    4    smooth_polygons             — SmoothPolygon (PAEK)
    4.5  crown_separation            — Watershed OR Grid (mutually exclusive)
    7    classify_trees              — Accept/reject by area + nDSM height

Crown separation methods (mutually exclusive):
    Watershed — Runs ONLY on polygons > ws_min_area. Uses nDSM local maxima
               with prominence filter. Polygons with < 3 peaks stay intact.
               Total area of each original polygon is preserved.
    Grid      — Fishnet overlay + Intersect. Simple, deterministic.
    None      — No crown separation (default).

Imported by PostProcessing.pyt -> TreeAnalysis.execute()
"""

import arcpy
import os

import _postproc_common as common


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: SMOOTH (Tree-specific)
# ═══════════════════════════════════════════════════════════════════════════════

def smooth_polygons(raw_path, smooth_tol, tmp_dir):
    """
    Smooth raw tree polygons using the PAEK algorithm to produce
    organic crown shapes.

    Returns
    -------
    tuple (str, int, int) : (smoothed_path, n_after, n_lost)
        Returns (None, 0, n_raw) if all polygons collapse.
    """
    arcpy.AddMessage("")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage(
        f"PHASE 4: Smooth Polygons (PAEK, tol = {smooth_tol} m)")
    arcpy.AddMessage("=" * 60)

    smooth_path = os.path.join(tmp_dir, "t_smooth.shp")
    arcpy.cartography.SmoothPolygon(
        raw_path, smooth_path,
        "PAEK", smooth_tol,
        "FIXED_ENDPOINT", "NO_CHECK")

    n_raw = int(arcpy.management.GetCount(raw_path)[0])
    n_smooth = int(arcpy.management.GetCount(smooth_path)[0])
    lost = n_raw - n_smooth
    arcpy.AddMessage(
        f"  After smooth: {n_smooth} ({lost} collapsed)")

    if n_smooth == 0:
        arcpy.AddWarning("All polygons collapsed during smoothing.")
        return None, 0, lost

    return smooth_path, n_smooth, lost


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4.5 — OPTION A: WATERSHED CROWN SEPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def watershed_crown_separation(poly_path, in_ndsm,
                               ws_min_area, search_radius_m,
                               tmp_dir):
    """
    Separate merged tree canopies using marker-controlled Watershed.

    KEY DESIGN:
        1. Only polygons > ws_min_area are processed.
        2. Prominence filter: peak must rise >= 2 m above local min.
        3. Only polygons with 3+ peaks are split.
        4. Split pieces clipped to original polygon → area preserved.

    Parameters
    ----------
    poly_path : str         — Smoothed tree polygons
    in_ndsm : str           — nDSM raster path
    ws_min_area : float     — Min polygon area (m2) to apply watershed
    search_radius_m : float — Local maxima search radius in meters
    tmp_dir : str           — Temp directory

    Returns
    -------
    tuple (str, int) : (result_path, feature_count)
    """
    MIN_PEAKS_TO_SPLIT = 3
    PROMINENCE_MIN = 2.0
    SMOOTH_RADIUS_M = 2.5

    arcpy.AddMessage("")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage("PHASE 4.5: Watershed Crown Separation")
    arcpy.AddMessage("=" * 60)

    n_input = int(arcpy.management.GetCount(poly_path)[0])
    arcpy.AddMessage(f"  Input polygons: {n_input}")
    arcpy.AddMessage(f"  Min area to process: {ws_min_area} m2")
    arcpy.AddMessage(f"  Search radius: {search_radius_m} m")
    arcpy.AddMessage(f"  Prominence >= {PROMINENCE_MIN} m")
    arcpy.AddMessage(f"  Split threshold: {MIN_PEAKS_TO_SPLIT}+ peaks")

    # ── 0. Tag originals + separate large vs normal ──────────────
    arcpy.management.AddField(poly_path, "_OrigID", "LONG")
    arcpy.management.AddField(poly_path, "_Area", "DOUBLE")
    arcpy.management.CalculateGeometryAttributes(
        poly_path, [["_Area", "AREA"]], area_unit="SQUARE_METERS")
    with arcpy.da.UpdateCursor(poly_path, ["_OrigID", "OID@"]) as uc:
        for row in uc:
            row[0] = row[1]
            uc.updateRow(row)

    large_lyr = arcpy.management.MakeFeatureLayer(
        poly_path, "ws_large_lyr", f"_Area > {ws_min_area}")
    normal_lyr = arcpy.management.MakeFeatureLayer(
        poly_path, "ws_normal_lyr", f"_Area <= {ws_min_area}")

    n_large = int(arcpy.management.GetCount(large_lyr)[0])
    n_normal = int(arcpy.management.GetCount(normal_lyr)[0])
    arcpy.AddMessage(
        f"  Large (> {ws_min_area} m2): {n_large}, "
        f"Normal: {n_normal}")

    large_path = os.path.join(tmp_dir, "ws_large.shp")
    normal_path = os.path.join(tmp_dir, "ws_normal.shp")
    arcpy.management.CopyFeatures(large_lyr, large_path)
    arcpy.management.CopyFeatures(normal_lyr, normal_path)
    arcpy.management.Delete(large_lyr)
    arcpy.management.Delete(normal_lyr)

    if n_large == 0:
        arcpy.AddMessage("  No oversized polygons — skipping.")
        _cleanup_ws_fields(poly_path)
        return poly_path, n_input

    # ── 1. Mask nDSM to large polygons ───────────────────────────
    ndsm_r = arcpy.Raster(in_ndsm)
    tree_mask = arcpy.sa.ExtractByMask(ndsm_r, large_path)
    mask_path = os.path.join(tmp_dir, "ws_ndsm_mask.tif")
    tree_mask.save(mask_path)
    del ndsm_r, tree_mask

    # ── 2. Smooth nDSM ───────────────────────────────────────────
    arcpy.AddMessage("  Smoothing nDSM...")
    mask_r = arcpy.Raster(mask_path)
    cell_size = mask_r.meanCellWidth
    smooth_cells = max(5, int(round(SMOOTH_RADIUS_M / cell_size)))
    smoothed = arcpy.sa.FocalStatistics(
        mask_r, arcpy.sa.NbrCircle(smooth_cells, "CELL"), "MEAN")
    smooth_ndsm_path = os.path.join(tmp_dir, "ws_ndsm_smooth.tif")
    smoothed.save(smooth_ndsm_path)
    del mask_r, smoothed

    # ── 3. Invert ────────────────────────────────────────────────
    inv_ndsm = arcpy.sa.Negate(arcpy.Raster(smooth_ndsm_path))
    inv_path = os.path.join(tmp_dir, "ws_ndsm_inv.tif")
    inv_ndsm.save(inv_path)
    del inv_ndsm

    # ── 4. Local maxima detection ────────────────────────────────
    inv_r = arcpy.Raster(inv_path)
    nbr_cells = max(7, int(round(search_radius_m / cell_size)))
    nbr = arcpy.sa.NbrCircle(nbr_cells, "CELL")
    arcpy.AddMessage(
        f"  Local maxima: radius {nbr_cells} cells "
        f"(~{nbr_cells * cell_size:.1f} m)")

    focal_min = arcpy.sa.FocalStatistics(inv_r, nbr, "MINIMUM")
    raw_maxima = arcpy.sa.Con(inv_r == focal_min, 1)
    del inv_r, focal_min

    # ── 4b. Prominence filter ────────────────────────────────────
    sm_r = arcpy.Raster(smooth_ndsm_path)
    prom_cells = max(7, int(round(search_radius_m * 1.2 / cell_size)))
    local_min_ndsm = arcpy.sa.FocalStatistics(
        sm_r, arcpy.sa.NbrCircle(prom_cells, "CELL"), "MINIMUM")
    prominence = sm_r - local_min_ndsm
    del sm_r, local_min_ndsm

    maxima = arcpy.sa.Con(
        (raw_maxima == 1) & (prominence >= PROMINENCE_MIN), 1)
    maxima_path = os.path.join(tmp_dir, "ws_maxima.tif")
    maxima.save(maxima_path)
    arcpy.management.CalculateStatistics(maxima_path)
    del raw_maxima, prominence, maxima

    # ── 5. Label seeds ───────────────────────────────────────────
    seed_regions = arcpy.sa.RegionGroup(
        arcpy.Raster(maxima_path), "EIGHT", "WITHIN", "NO_LINK")
    seed_path = os.path.join(tmp_dir, "ws_seeds.tif")
    seed_regions.save(seed_path)
    arcpy.management.CalculateStatistics(seed_path)
    del seed_regions

    try:
        seed_r = arcpy.Raster(seed_path)
        if seed_r.maximum is None or seed_r.maximum == 0:
            arcpy.AddWarning("  No prominent peaks — skipping.")
            del seed_r
            return _merge_and_return(large_path, normal_path,
                                     n_normal, poly_path, tmp_dir)
        n_seeds = int(seed_r.maximum)
        arcpy.AddMessage(f"  Prominent seeds: {n_seeds}")
        del seed_r
    except Exception:
        arcpy.AddWarning("  Seed raster error — skipping.")
        return _merge_and_return(large_path, normal_path,
                                 n_normal, poly_path, tmp_dir)

    # ── 6. Flow Direction ────────────────────────────────────────
    arcpy.AddMessage("  Computing flow direction...")
    flow_dir = arcpy.sa.FlowDirection(arcpy.Raster(inv_path))
    fd_path = os.path.join(tmp_dir, "ws_flowdir.tif")
    flow_dir.save(fd_path)
    arcpy.management.CalculateStatistics(fd_path)
    del flow_dir

    # ── 7. Watershed ─────────────────────────────────────────────
    ws_result = arcpy.sa.Watershed(
        arcpy.Raster(fd_path), arcpy.Raster(seed_path))
    ws_path = os.path.join(tmp_dir, "ws_zones.tif")
    ws_result.save(ws_path)
    arcpy.management.CalculateStatistics(ws_path)
    del ws_result

    # ── 8. Vectorize zones ───────────────────────────────────────
    ws_poly_path = os.path.join(tmp_dir, "ws_zones_poly.shp")
    arcpy.conversion.RasterToPolygon(
        ws_path, ws_poly_path,
        "NO_SIMPLIFY", "VALUE", "SINGLE_OUTER_PART")
    n_zones = int(arcpy.management.GetCount(ws_poly_path)[0])
    arcpy.AddMessage(f"  Watershed zones: {n_zones}")

    if n_zones == 0:
        arcpy.AddWarning("  No zones produced.")
        return _merge_and_return(large_path, normal_path,
                                 n_normal, poly_path, tmp_dir)

    # ── 9. Intersect zones with large polygons ───────────────────
    ws_clip_path = os.path.join(tmp_dir, "t_ws_intersect.shp")
    arcpy.analysis.Intersect(
        [large_path, ws_poly_path],
        ws_clip_path, "ALL", "", "INPUT")

    n_intersected = int(arcpy.management.GetCount(ws_clip_path)[0])
    arcpy.AddMessage(f"  Intersected pieces: {n_intersected}")

    if n_intersected == 0:
        arcpy.AddWarning("  No intersect results.")
        return _merge_and_return(large_path, normal_path,
                                 n_normal, poly_path, tmp_dir)

    # ── 10. Count pieces per original polygon ────────────────────
    zone_counts = {}
    with arcpy.da.SearchCursor(ws_clip_path, ["_OrigID"]) as sc:
        for row in sc:
            oid = row[0]
            zone_counts[oid] = zone_counts.get(oid, 0) + 1

    split_ids = {k for k, v in zone_counts.items()
                 if v >= MIN_PEAKS_TO_SPLIT}
    keep_ids = {k for k, v in zone_counts.items()
                if v < MIN_PEAKS_TO_SPLIT}

    all_large_ids = set()
    with arcpy.da.SearchCursor(large_path, ["_OrigID"]) as sc:
        for row in sc:
            all_large_ids.add(row[0])
    keep_ids = keep_ids | (all_large_ids - set(zone_counts.keys()))

    arcpy.AddMessage(
        f"  Polygons split ({MIN_PEAKS_TO_SPLIT}+ zones): "
        f"{len(split_ids)}")
    arcpy.AddMessage(
        f"  Large kept intact: {len(keep_ids)}")

    # ── 11. Collect split pieces ─────────────────────────────────
    merge_inputs = []

    if len(split_ids) > 0:
        split_path = os.path.join(tmp_dir, "t_ws_split.shp")
        id_list = ",".join(str(i) for i in split_ids)
        lyr = arcpy.management.MakeFeatureLayer(
            ws_clip_path, "ws_sp_lyr",
            f"_OrigID IN ({id_list})")
        arcpy.management.CopyFeatures(lyr, split_path)
        arcpy.management.Delete(lyr)
        merge_inputs.append(split_path)

    # ── 12. Collect kept large originals ─────────────────────────
    if len(keep_ids) > 0:
        kept_path = os.path.join(tmp_dir, "t_ws_kept_large.shp")
        id_list = ",".join(str(i) for i in keep_ids)
        lyr = arcpy.management.MakeFeatureLayer(
            large_path, "ws_kp_lyr",
            f"_OrigID IN ({id_list})")
        arcpy.management.CopyFeatures(lyr, kept_path)
        arcpy.management.Delete(lyr)
        merge_inputs.append(kept_path)

    # ── 13. Merge: split + kept-large + normal ───────────────────
    if n_normal > 0:
        merge_inputs.append(normal_path)

    merged_path = os.path.join(tmp_dir, "t_ws_merged.shp")
    arcpy.management.Merge(merge_inputs, merged_path)

    _cleanup_ws_fields(poly_path)
    try:
        arcpy.management.DeleteField(merged_path, ["_OrigID", "_Area"])
    except Exception:
        pass

    n_after = int(arcpy.management.GetCount(merged_path)[0])
    arcpy.AddMessage(
        f"  Final: {n_after} polygons (was {n_input})")

    return merged_path, n_after


def _merge_and_return(large_path, normal_path, n_normal,
                      poly_path, tmp_dir):
    """Fallback: merge large + normal back and return."""
    _cleanup_ws_fields(poly_path)
    merged = os.path.join(tmp_dir, "t_ws_merged.shp")
    inputs = [large_path]
    if n_normal > 0:
        inputs.append(normal_path)
    arcpy.management.Merge(inputs, merged)
    try:
        arcpy.management.DeleteField(merged, ["_OrigID", "_Area"])
    except Exception:
        pass
    return merged, int(arcpy.management.GetCount(merged)[0])


def _cleanup_ws_fields(poly_path):
    """Remove temporary watershed fields."""
    try:
        arcpy.management.DeleteField(poly_path, ["_OrigID", "_Area"])
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4.5 — OPTION B: GRID SPLITTING
# ═══════════════════════════════════════════════════════════════════════════════

def grid_split_polygons(poly_path, grid_size, tmp_dir):
    """
    Split polygons using a regular fishnet grid overlay.

    Returns
    -------
    tuple (str, int) : (result_path, feature_count)
    """
    arcpy.AddMessage("")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage(
        f"PHASE 4.5: Grid Splitting (cell = {grid_size} m)")
    arcpy.AddMessage("=" * 60)

    n_input = int(arcpy.management.GetCount(poly_path)[0])
    arcpy.AddMessage(f"  Input polygons: {n_input}")

    desc = arcpy.Describe(poly_path)
    ext = desc.extent

    fish_path = os.path.join(tmp_dir, "t_fishnet.shp")
    arcpy.management.CreateFishnet(
        fish_path,
        origin_coord=f"{ext.XMin} {ext.YMin}",
        y_axis_coord=f"{ext.XMin} {ext.YMax}",
        cell_width=grid_size,
        cell_height=grid_size,
        number_rows=0, number_columns=0,
        corner_coord=f"{ext.XMax} {ext.YMax}",
        labels="NO_LABELS",
        template=poly_path,
        geometry_type="POLYGON")

    n_cells = int(arcpy.management.GetCount(fish_path)[0])
    arcpy.AddMessage(f"  Grid cells: {n_cells}")

    if n_cells == 0:
        arcpy.AddWarning("  No grid cells — keeping originals.")
        return poly_path, n_input

    grid_clip_path = os.path.join(tmp_dir, "t_grid_split.shp")
    arcpy.analysis.Intersect(
        [poly_path, fish_path],
        grid_clip_path, "ALL", "", "INPUT")

    n_after = int(arcpy.management.GetCount(grid_clip_path)[0])
    arcpy.AddMessage(f"  After grid split: {n_after} (was {n_input})")

    if n_after == 0:
        arcpy.AddWarning("  Grid intersect empty — keeping originals.")
        return poly_path, n_input

    return grid_clip_path, n_after


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7: CLASSIFICATION (Tree-specific)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_trees(fc_path, min_area, max_area,
                   min_ndsm, reject_null_ndsm):
    """
    Classify features as 'Tree' or 'Rejected possible tree'.

    Returns
    -------
    tuple (int, dict) : (n_accepted, reject_counts)
    """
    arcpy.AddMessage("")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage("PHASE 7: Classification")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage(
        f"  Area: {min_area}--{max_area} m2, "
        f"min nDSM >= {min_ndsm} m")
    arcpy.AddMessage(
        f"  NULL nDSM policy: "
        f"{'reject' if reject_null_ndsm else 'keep with warning'}")

    arcpy.management.AddField(fc_path, "Class", "TEXT", field_length=30)
    arcpy.management.AddField(fc_path, "Reason", "TEXT", field_length=30)

    n_accepted = n_small = n_large = n_low_ndsm = 0
    n_null_ndsm = n_null_kept = 0

    with arcpy.da.UpdateCursor(
        fc_path, ["Area_m2", "MEAN_nDSM", "Class", "Reason"]
    ) as ucur:
        for row in ucur:
            area = row[0] if row[0] else 0
            ndsm = row[1]

            if area < min_area:
                row[2], row[3] = "Rejected possible tree", "Area_below_min"
                n_small += 1
            elif area > max_area:
                row[2], row[3] = "Rejected possible tree", "Area_above_max"
                n_large += 1
            elif ndsm is None:
                if reject_null_ndsm:
                    row[2], row[3] = "Rejected possible tree", "nDSM_null"
                    n_null_ndsm += 1
                else:
                    row[2], row[3] = "Tree", "nDSM_null_kept"
                    n_accepted += 1
                    n_null_kept += 1
            elif ndsm < min_ndsm:
                row[2], row[3] = "Rejected possible tree", "nDSM_below_min"
                n_low_ndsm += 1
            else:
                row[2], row[3] = "Tree", ""
                n_accepted += 1
            ucur.updateRow(row)

    reject_counts = {
        "area_small": n_small, "area_large": n_large,
        "nDSM_below_min": n_low_ndsm, "nDSM_null": n_null_ndsm,
    }
    n_rej = n_small + n_large + n_low_ndsm + n_null_ndsm

    arcpy.AddMessage(f"  Tree: {n_accepted}")
    if n_null_kept > 0:
        arcpy.AddWarning(f"  ({n_null_kept} accepted with NULL nDSM)")
    arcpy.AddMessage(
        f"  Rejected: {n_rej} "
        f"(area:{n_small + n_large}, "
        f"nDSM_low:{n_low_ndsm}, nDSM_null:{n_null_ndsm})")

    return n_accepted, reject_counts


# ═══════════════════════════════════════════════════════════════════════════════
# FULL TREE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_tree_pipeline(in_cls, tree_val, in_ndsm, in_therm,
                      in_aoi, out_crs_text, cs_opt, cs_custom,
                      min_area, max_area, min_ndsm,
                      reject_null_ndsm,
                      maj_nbr, maj_passes, smooth_tol,
                      crown_method, ws_min_area,
                      ws_search_radius, grid_size,
                      do_centroids, out_folder):
    """
    Execute the complete tree post-processing pipeline.
    """
    PREFIX = "tree"
    LABEL = "Tree"
    ID_FIELD = "TreeID"

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
        "smooth_tolerance_m": smooth_tol,
        "crown_separation": crown_method,
        "ws_min_area_m2": ws_min_area if crown_method == "Watershed" else None,
        "ws_search_radius_m": ws_search_radius if crown_method == "Watershed" else None,
        "grid_cell_size_m": grid_size if crown_method == "Grid" else None,
        "generate_centroids": do_centroids,
        "output_folder": out_folder,
    }

    sub_dir, tmp_dir, run_log, t0 = common.init_run(
        out_folder, PREFIX, in_cls, tree_val,
        in_ndsm, in_therm, in_aoi, params_dict)

    try:
        env = common.validate_inputs(
            in_cls, in_ndsm, in_therm, in_aoi,
            out_crs_text, cs_opt, cs_custom,
            tree_val, LABEL, run_log)

        bin_path = common.extract_class_pixels(
            in_cls, tree_val, LABEL, tmp_dir)

        filt_path = common.run_majority_filter(
            bin_path, maj_nbr, maj_passes, tmp_dir)

        raw_path, n_raw = common.vectorize_binary(filt_path, tmp_dir)

        if n_raw == 0:
            arcpy.AddWarning(f"No pixels with value {tree_val} found.")
            empty_path = os.path.join(sub_dir, "tree_results.shp")
            arcpy.management.CopyFeatures(raw_path, empty_path)
            common.write_log(run_log, sub_dir, "tree_log.json")
            return empty_path, None

        smooth_path, n_smooth, lost_smooth = smooth_polygons(
            raw_path, smooth_tol, tmp_dir)

        if smooth_path is None:
            empty_path = os.path.join(sub_dir, "tree_results.shp")
            arcpy.management.CopyFeatures(raw_path, empty_path)
            common.write_log(run_log, sub_dir, "tree_log.json")
            return empty_path, None

        work_path = smooth_path
        n_work = n_smooth

        # ── PHASE 4.5: CROWN SEPARATION (mutually exclusive) ────
        if crown_method == "Watershed":
            work_path, n_work = watershed_crown_separation(
                work_path, in_ndsm,
                ws_min_area, ws_search_radius, tmp_dir)
            run_log["results"]["n_after_watershed"] = n_work
        elif crown_method == "Grid":
            work_path, n_work = grid_split_polygons(
                work_path, grid_size, tmp_dir)
            run_log["results"]["n_after_grid_split"] = n_work

        common.compute_geometry_mbr(work_path, tmp_dir)

        enriched_path = os.path.join(tmp_dir, "t_enriched.shp")
        arcpy.management.CopyFeatures(work_path, enriched_path)

        n_null_h = common.enrich_attributes(
            enriched_path, in_ndsm, in_therm, ID_FIELD, tmp_dir)

        n_accepted, reject_counts = classify_trees(
            enriched_path, min_area, max_area,
            min_ndsm, reject_null_ndsm)

        common.set_output_crs(env["tgt_sr"])

        final_path, pts_path = common.export_results(
            enriched_path, out_folder, PREFIX, do_centroids,
            run_log, in_aoi, env["aoi_m2"],
            tree_val, LABEL,
            n_raw, lost_smooth, n_work, n_accepted,
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
