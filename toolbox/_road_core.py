# -*- coding: utf-8 -*-
"""
_road_core.py — Road Analysis pipeline
========================================
Master Thesis: Semantic Segmentation of Multispectral Drone Imagery
Sensor: MicaSense Altum-PT  |  ArcGIS Pro 3.6.2

Developed by Nikolaos Koroniadis
MSc Geography and Applied Geoinformatics, University of the Aegean
Remote Sensing & GIS Research Group (https://rsgis.aegean.gr/)

Road-specific phases:
    4a   simplify_and_dissolve    — SimplifyPolygon + Dissolve adjacent
    7    classify_roads            — Accept/reject by nDSM + length + width
    7.5  extract_centerlines       — Voronoi medial axis → polyline centerlines
                                    (fallback: morphological Thin if deps missing)

Roads differ from discrete objects (vehicles, buildings, trees) in that
they are linear/network features.  Adjacent road polygons are dissolved
into contiguous segments before classification, and an optional
centerline extraction phase produces a polyline network from the
accepted road polygons.

Centerline method (v2 — Voronoi):
    The original Thin-based approach produced excessive spurs ("καρδιογράφημα")
    and failed at junctions and variable-width sections.  The Voronoi medial
    axis approach densifies the polygon boundary, computes the Voronoi diagram,
    retains only interior edges, builds a networkx graph, and iteratively prunes
    short terminal branches.  This produces geometrically correct centerlines
    that handle junctions, bends, and variable widths natively.

    Dependencies: shapely ≥ 2.0, scipy, networkx, numpy
    Fallback:     morphological Thin (no extra deps, lower quality)

Imported by PostProcessing.pyt → RoadAnalysis.execute()
"""

import arcpy
import os
import traceback

import _postproc_common as common

# ── Optional dependencies for Voronoi centerlines ───────────────────────────
try:
    import numpy as np
    from shapely.geometry import (
        Polygon as ShapelyPolygon,
        MultiPolygon as ShapelyMultiPolygon,
        LineString as ShapelyLineString,
        MultiLineString as ShapelyMultiLineString,
        Point as ShapelyPoint,
    )
    from shapely.ops import linemerge
    from shapely.prepared import prep
    from shapely import wkt as shapely_wkt
    from scipy.spatial import Voronoi
    import networkx as nx

    HAS_VORONOI = True
except ImportError as _imp_err:
    HAS_VORONOI = False
    _VORONOI_IMPORT_MSG = str(_imp_err)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4a: SIMPLIFY + DISSOLVE (Road-specific)  — UNCHANGED
# ═══════════════════════════════════════════════════════════════════════════════

def simplify_and_dissolve(raw_path, simp_tol, tmp_dir):
    """
    Simplify raw road polygons and dissolve adjacent segments.

    Step 1: SimplifyPolygon (POINT_REMOVE) to remove the raster
            staircase effect.
    Step 2: Dissolve with no dissolve field + SINGLE_PART to merge
            all touching/overlapping polygons into contiguous road
            segments while keeping disconnected segments separate.

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
    tuple (str, int, int, int) :
        (dissolved_path, n_dissolved, n_lost_simplify, n_before_dissolve)
        Returns (None, 0, n_raw, 0) if all polygons collapse.
    """
    arcpy.AddMessage("")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage(
        f"PHASE 4a: Simplify + Dissolve (tol = {simp_tol} m)")
    arcpy.AddMessage("=" * 60)

    n_raw = int(arcpy.management.GetCount(raw_path)[0])

    # ── Step 1: Simplify ─────────────────────────────────────────
    simp_path = os.path.join(tmp_dir, "r_simp.shp")
    arcpy.cartography.SimplifyPolygon(
        raw_path, simp_path, "POINT_REMOVE", simp_tol,
        minimum_area="0 SquareMeters",
        error_option="RESOLVE_ERRORS",
        collapsed_point_option="NO_KEEP")

    n_simp = int(arcpy.management.GetCount(simp_path)[0])
    lost_simp = n_raw - n_simp
    arcpy.AddMessage(
        f"  After simplify: {n_simp} ({lost_simp} collapsed)")

    if n_simp == 0:
        arcpy.AddWarning("All polygons collapsed during simplification.")
        return None, 0, lost_simp, 0

    # ── Step 2: Dissolve adjacent ────────────────────────────────
    diss_path = os.path.join(tmp_dir, "r_dissolved.shp")
    arcpy.management.Dissolve(
        simp_path, diss_path,
        dissolve_field=None,
        statistics_fields=None,
        multi_part="SINGLE_PART",
        unsplit_lines="DISSOLVE_LINES")

    n_diss = int(arcpy.management.GetCount(diss_path)[0])
    arcpy.AddMessage(
        f"  After dissolve: {n_diss} "
        f"(merged {n_simp - n_diss} adjacent segments)")

    if n_diss == 0:
        arcpy.AddWarning("All polygons lost during dissolve.")
        return None, 0, lost_simp, n_simp

    return diss_path, n_diss, lost_simp, n_simp


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7: CLASSIFICATION (Road-specific)  — UNCHANGED
# ═══════════════════════════════════════════════════════════════════════════════

def classify_roads(fc_path, min_area, max_ndsm,
                   min_length, min_width, reject_null_ndsm):
    """
    Classify features as 'Road' or 'Rejected possible road'
    based on area, nDSM height ceiling, and MBR dimensions.

    Roads must be ground-level (nDSM below a ceiling), long enough,
    and wide enough to be accepted.  This is the inverse logic of
    buildings/trees which require a minimum nDSM.

    Adds 'Class' and 'Reason' fields.  Operates in-place.

    Parameters
    ----------
    fc_path : str
        Path to enriched feature class.
    min_area : float
        Minimum road segment area (m²).
    max_ndsm : float
        Maximum mean nDSM height (m).  Features above are rejected
        (likely misclassified elevated structures).
    min_length : float
        Minimum MBR length (m).
    min_width : float
        Minimum MBR width (m).
    reject_null_ndsm : bool
        If True, features with NULL nDSM are rejected.
        If False, they are kept as Road with a warning.

    Returns
    -------
    tuple (int, dict) : (n_accepted, reject_counts)
    """
    arcpy.AddMessage("")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage("PHASE 7: Classification")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage(
        f"  Area >= {min_area} m², nDSM <= {max_ndsm} m")
    arcpy.AddMessage(
        f"  MBR_L >= {min_length} m, MBR_W >= {min_width} m")
    arcpy.AddMessage(
        f"  NULL nDSM policy: "
        f"{'reject' if reject_null_ndsm else 'keep with warning'}")

    arcpy.management.AddField(
        fc_path, "Class", "TEXT", field_length=30)
    arcpy.management.AddField(
        fc_path, "Reason", "TEXT", field_length=30)

    n_accepted = 0
    n_small = 0
    n_high_ndsm = 0
    n_short = 0
    n_narrow = 0
    n_null_ndsm = 0
    n_null_kept = 0

    with arcpy.da.UpdateCursor(
        fc_path,
        ["Area_m2", "MEAN_nDSM", "MBR_L", "MBR_W",
         "Class", "Reason"]
    ) as ucur:
        for row in ucur:
            area = row[0] if row[0] else 0
            ndsm = row[1]       # can be None
            mbr_l = row[2] if row[2] else 0
            mbr_w = row[3] if row[3] else 0

            # ── Rejection cascade (most common first) ────────────
            if area < min_area:
                row[4] = "Rejected possible road"
                row[5] = "Area_below_min"
                n_small += 1
            elif ndsm is None:
                if reject_null_ndsm:
                    row[4] = "Rejected possible road"
                    row[5] = "nDSM_null"
                    n_null_ndsm += 1
                else:
                    row[4] = "Road"
                    row[5] = "nDSM_null_kept"
                    n_accepted += 1
                    n_null_kept += 1
            elif ndsm > max_ndsm:
                row[4] = "Rejected possible road"
                row[5] = "nDSM_above_max"
                n_high_ndsm += 1
            elif mbr_l < min_length:
                row[4] = "Rejected possible road"
                row[5] = "Length_below_min"
                n_short += 1
            elif mbr_w < min_width:
                row[4] = "Rejected possible road"
                row[5] = "Width_below_min"
                n_narrow += 1
            else:
                row[4] = "Road"
                row[5] = ""
                n_accepted += 1
            ucur.updateRow(row)

    reject_counts = {
        "area_small": n_small,
        "nDSM_above_max": n_high_ndsm,
        "length_short": n_short,
        "width_narrow": n_narrow,
        "nDSM_null": n_null_ndsm,
    }
    n_rej = n_small + n_high_ndsm + n_short + n_narrow + n_null_ndsm

    arcpy.AddMessage(f"  Road:     {n_accepted}")
    if n_null_kept > 0:
        arcpy.AddWarning(
            f"  ({n_null_kept} accepted with NULL nDSM)")
    arcpy.AddMessage(
        f"  Rejected: {n_rej} "
        f"(area:{n_small}, nDSM_high:{n_high_ndsm}, "
        f"short:{n_short}, narrow:{n_narrow}, "
        f"nDSM_null:{n_null_ndsm})")

    return n_accepted, reject_counts


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7.5: CENTERLINE EXTRACTION (Road-specific)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Primary method  : Voronoi medial axis  (shapely + scipy + networkx)
# Fallback method : Morphological Thin   (arcpy only)
# ═══════════════════════════════════════════════════════════════════════════════


def extract_centerlines(fc_path, in_cls, cell_size,
                        min_cl_length, tmp_dir,
                        max_hole_area=20.0,
                        prune_factor=2.0,
                        densify_factor=5.0):
    """
    Extract polyline centerlines from accepted road polygons.

    Dispatches to Voronoi method (preferred) or Thin fallback.
    """
    arcpy.AddMessage("")
    arcpy.AddMessage("=" * 60)
    arcpy.AddMessage("PHASE 7.5: Centerline Extraction")
    arcpy.AddMessage("=" * 60)

    if HAS_VORONOI:
        arcpy.AddMessage("  Method: Voronoi medial axis")
        return _extract_centerlines_voronoi(
            fc_path, cell_size, min_cl_length, tmp_dir,
            max_hole_area, prune_factor, densify_factor)
    else:
        arcpy.AddWarning(
            f"  Voronoi deps missing ({_VORONOI_IMPORT_MSG})")
        arcpy.AddWarning(
            "  Falling back to morphological Thin (lower quality)")
        return _extract_centerlines_thin(
            fc_path, in_cls, cell_size, min_cl_length, tmp_dir)


# ─── VORONOI METHOD (PRIMARY) ────────────────────────────────────────────────

def _extract_centerlines_voronoi(fc_path, cell_size,
                                 min_cl_length, tmp_dir,
                                 max_hole_area=20.0,
                                 prune_factor=2.0,
                                 densify_factor=5.0):
    """
    Voronoi medial axis centerline extraction.

    Workflow per polygon:
        1. Convert arcpy polygon → shapely
        2. Densify boundary at regular interval
        3. Compute Voronoi diagram (scipy)
        4. Retain only edges fully interior to polygon
        5. Build networkx graph
        6. Iteratively prune short terminal branches
        7. Extract merged LineStrings (shapely linemerge)

    Then:
        8. Write all centerlines to temp shapefile
        9. SpatialJoin to inherit parent polygon attributes
       10. Compute CL_Len_m + filter by min_cl_length
       11. SmoothLine (PAEK)
    """

    # ── 1. Select accepted roads ─────────────────────────────────
    acc_lyr = arcpy.management.MakeFeatureLayer(
        fc_path, "road_accepted_lyr_v", "Class = 'Road'")
    n_acc = int(arcpy.management.GetCount(acc_lyr)[0])
    arcpy.AddMessage(f"  Accepted road polygons: {n_acc}")

    if n_acc == 0:
        arcpy.AddWarning(
            "  No accepted roads — skipping centerline extraction.")
        arcpy.management.Delete(acc_lyr)
        return None

    acc_path = os.path.join(tmp_dir, "r_accepted_v.shp")
    arcpy.management.CopyFeatures(acc_lyr, acc_path)
    arcpy.management.Delete(acc_lyr)

    sr = arcpy.Describe(acc_path).spatialReference

    # ── 2. Compute median width for prune threshold ──────────────
    widths = []
    with arcpy.da.SearchCursor(acc_path, ["MBR_W"]) as sc:
        for row in sc:
            if row[0] is not None and row[0] > 0:
                widths.append(row[0])
    if widths:
        widths.sort()
        median_w = widths[len(widths) // 2]
    else:
        median_w = 3.0
    arcpy.AddMessage(f"  Median road width: {median_w:.2f} m")

    # Prune threshold: branches shorter than this are removed
    prune_length = median_w * prune_factor
    arcpy.AddMessage(f"  Prune threshold: {prune_length:.2f} m "
                     f"({prune_factor}× median width)")

    # Densify interval: densify_factor × cell_size but at least 0.15m
    densify_interval = max(cell_size * densify_factor, 0.15)
    arcpy.AddMessage(
        f"  Densify interval: {densify_interval:.3f} m "
        f"({densify_factor}× cell size)")

    # ── 3. Process each polygon ──────────────────────────────────
    all_lines = []    # list of shapely LineStrings
    n_success = 0
    n_skip = 0
    n_holes_filled = 0

    arcpy.AddMessage(
        f"  Max hole area to fill: {max_hole_area:.1f} m²")

    with arcpy.da.SearchCursor(
        acc_path, ["OID@", "SHAPE@"]
    ) as scur:
        for oid, shape in scur:
            try:
                # Convert arcpy → shapely via WKT
                shp_poly = shapely_wkt.loads(shape.WKT)

                # Handle MultiPolygon (shouldn't happen after
                # SINGLE_PART dissolve, but be safe)
                if isinstance(shp_poly, ShapelyMultiPolygon):
                    polys = list(shp_poly.geoms)
                elif isinstance(shp_poly, ShapelyPolygon):
                    polys = [shp_poly]
                else:
                    arcpy.AddMessage(
                        f"    OID {oid}: unexpected geom type "
                        f"'{shp_poly.geom_type}' — skipping")
                    n_skip += 1
                    continue

                for poly in polys:
                    # Validate / repair
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    if poly.is_empty or poly.area < 0.5:
                        continue

                    # Count holes that will be filled
                    if poly.interiors:
                        small = sum(
                            1 for r in poly.interiors
                            if abs(ShapelyPolygon(r).area)
                            <= max_hole_area)
                        n_holes_filled += small

                    result = _voronoi_centerline(
                        poly, densify_interval, prune_length,
                        max_hole_area)
                    if result:
                        all_lines.extend(result)
                        n_success += 1

            except Exception as e:
                arcpy.AddMessage(
                    f"    OID {oid}: Voronoi failed ({e}) — skipping")
                n_skip += 1

    if n_holes_filled > 0:
        arcpy.AddMessage(
            f"  Holes filled (vehicles/shadows): {n_holes_filled}")
    arcpy.AddMessage(
        f"  Voronoi: {n_success} polygons produced centerlines, "
        f"{n_skip} skipped")
    arcpy.AddMessage(
        f"  Raw centerline segments: {len(all_lines)}")

    if len(all_lines) == 0:
        arcpy.AddWarning(
            "  No centerlines generated — skipping.")
        return None

    # ── 4. Write centerlines to temp shapefile ───────────────────
    cl_raw_path = os.path.join(tmp_dir, "r_cl_voronoi.shp")
    arcpy.management.CreateFeatureclass(
        tmp_dir, "r_cl_voronoi.shp",
        "POLYLINE", spatial_reference=sr)

    with arcpy.da.InsertCursor(cl_raw_path, ["SHAPE@"]) as icur:
        for line in all_lines:
            try:
                arcpy_line = arcpy.FromWKT(line.wkt, sr)
                icur.insertRow([arcpy_line])
            except Exception:
                pass

    n_written = int(arcpy.management.GetCount(cl_raw_path)[0])
    arcpy.AddMessage(f"  Written to shapefile: {n_written}")

    if n_written == 0:
        arcpy.AddWarning(
            "  No centerlines written — skipping.")
        return None

    # ── 5. SpatialJoin — inherit parent polygon attributes ───────
    arcpy.AddMessage("  Joining parent polygon attributes...")
    cl_joined_path = os.path.join(tmp_dir, "r_cl_joined.shp")
    arcpy.analysis.SpatialJoin(
        cl_raw_path, acc_path, cl_joined_path,
        join_operation="JOIN_ONE_TO_ONE",
        join_type="KEEP_ALL",
        match_option="INTERSECT")

    # ── 6. Compute CL_Len_m ─────────────────────────────────────
    arcpy.management.AddField(cl_joined_path, "CL_Len_m", "DOUBLE")
    arcpy.management.CalculateGeometryAttributes(
        cl_joined_path,
        [["CL_Len_m", "LENGTH"]],
        length_unit="METERS")

    # ── 7. Filter by minimum length ──────────────────────────────
    arcpy.AddMessage(
        f"  Length filter: min = {min_cl_length} m")

    n_short_removed = 0
    with arcpy.da.UpdateCursor(
        cl_joined_path, ["CL_Len_m"]
    ) as ucur:
        for row in ucur:
            if row[0] is None or row[0] < min_cl_length:
                ucur.deleteRow()
                n_short_removed += 1

    n_after_len = int(
        arcpy.management.GetCount(cl_joined_path)[0])
    arcpy.AddMessage(
        f"  Removed {n_short_removed} short segments "
        f"({n_after_len} remaining)")

    if n_after_len == 0:
        arcpy.AddWarning(
            "  All centerlines below minimum length — skipping.")
        return None

    # ── 8. Smooth final centerlines ──────────────────────────────
    arcpy.AddMessage("  Smoothing centerlines (PAEK)...")
    smooth_path = os.path.join(tmp_dir, "r_cl_smooth.shp")
    arcpy.cartography.SmoothLine(
        cl_joined_path, smooth_path,
        "PAEK", cell_size * 20)

    n_smooth = int(arcpy.management.GetCount(smooth_path)[0])
    if n_smooth > 0:
        cl_final_path = smooth_path
        arcpy.AddMessage(f"  Smoothed: {n_smooth} centerlines")
    else:
        arcpy.AddMessage(
            "  Smooth produced no output — using unsmoothed.")
        cl_final_path = cl_joined_path

    # ── Clean up extra fields ────────────────────────────────────
    keep_fields = {
        "FID", "Shape", "OID",
        "RoadID", "Area_m2", "Perim_m",
        "MBR_W", "MBR_L", "MBR_Azim", "LW_Ratio", "Compact",
        "MEAN_nDSM", "MEAN_Therm", "Class", "Reason",
        "CL_Len_m",
    }
    drop_fields = []
    for f in arcpy.ListFields(cl_final_path):
        if (f.name not in keep_fields
                and not f.required
                and f.name.lower() not in ("shape_leng",
                                            "smoothline")):
            drop_fields.append(f.name)
    if drop_fields:
        try:
            arcpy.management.DeleteField(
                cl_final_path, drop_fields)
        except Exception:
            pass

    n_final = int(arcpy.management.GetCount(cl_final_path)[0])
    arcpy.AddMessage(f"  Final centerlines: {n_final}")
    arcpy.AddMessage(
        f"  Pipeline: {len(all_lines)} voronoi > "
        f"{n_written} written > {n_after_len} filtered > "
        f"{n_final} smoothed")

    return cl_final_path


# ─── VORONOI HELPER FUNCTIONS ────────────────────────────────────────────────

def _voronoi_centerline(polygon, densify_interval, prune_length,
                        max_hole_area=20.0):
    """
    Compute the Voronoi medial axis centerline of a single polygon.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Input road polygon.
    densify_interval : float
        Distance between sampled boundary points (meters).
    prune_length : float
        Terminal branches shorter than this are removed.
    max_hole_area : float
        Interior holes (vehicles, trees, shadows) smaller than this
        area (m²) are filled before computing the medial axis.
        Default 20 m² covers most vehicles (~12 m²) and small
        misclassified patches.  Larger holes (e.g. actual buildings)
        are preserved.

    Returns
    -------
    list of shapely LineStrings, or None
    """
    # ── 0. Fill small interior holes ─────────────────────────────
    #    Vehicles, shadows, and small misclassified objects sitting
    #    inside the road polygon create holes that force the Voronoi
    #    medial axis to branch around them.  Filling small holes
    #    produces a clean, straight centerline.
    if polygon.interiors:
        big_holes = [
            ring for ring in polygon.interiors
            if abs(ShapelyPolygon(ring).area) > max_hole_area
        ]
        n_filled = len(polygon.interiors) - len(big_holes)
        if n_filled > 0:
            polygon = ShapelyPolygon(polygon.exterior, big_holes)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
                # FIX F3: buffer(0) can return MultiPolygon on
                # degenerate geometries — keep the largest part
                if isinstance(polygon, ShapelyMultiPolygon):
                    polygon = max(polygon.geoms, key=lambda g: g.area)
                if polygon.is_empty:
                    return None

    exterior = polygon.exterior
    perimeter = exterior.length

    # Skip tiny polygons
    if perimeter < densify_interval * 8:
        return None

    # Cap boundary points to prevent excessive computation
    max_pts = 15000
    actual_interval = max(densify_interval, perimeter / max_pts)
    n_pts = int(perimeter / actual_interval)
    if n_pts < 20:
        return None

    # ── 1. Densify boundary ──────────────────────────────────────
    # Sample exterior ring
    distances = np.linspace(0, perimeter, n_pts, endpoint=False)
    coords_list = []
    for d in distances:
        pt = exterior.interpolate(d)
        coords_list.append((pt.x, pt.y))

    # Also densify remaining interior rings (large holes only)
    for interior in polygon.interiors:
        int_perim = interior.length
        int_interval = max(actual_interval, int_perim / 2000)
        n_int = max(int(int_perim / int_interval), 10)
        int_distances = np.linspace(
            0, int_perim, n_int, endpoint=False)
        for d in int_distances:
            pt = interior.interpolate(d)
            coords_list.append((pt.x, pt.y))

    coords = np.array(coords_list)

    if len(coords) < 4:
        return None

    # ── 2. Voronoi diagram ───────────────────────────────────────
    try:
        vor = Voronoi(coords)
    except Exception:
        return None

    # ── 3. Filter edges — keep only interior ones ────────────────
    prepped = prep(polygon)

    G = nx.Graph()

    for v1_idx, v2_idx in vor.ridge_vertices:
        # Skip infinite edges (index = -1)
        if v1_idx < 0 or v2_idx < 0:
            continue

        v1 = vor.vertices[v1_idx]
        v2 = vor.vertices[v2_idx]

        # Check both endpoints AND midpoint are inside polygon.
        # This catches edges that straddle the boundary.
        mid = ((v1[0] + v2[0]) * 0.5, (v1[1] + v2[1]) * 0.5)

        p1 = ShapelyPoint(v1[0], v1[1])
        p2 = ShapelyPoint(v2[0], v2[1])
        pm = ShapelyPoint(mid[0], mid[1])

        if (prepped.contains(p1)
                and prepped.contains(p2)
                and prepped.contains(pm)):
            edge_len = float(np.hypot(
                v1[0] - v2[0], v1[1] - v2[1]))
            n1 = (float(v1[0]), float(v1[1]))
            n2 = (float(v2[0]), float(v2[1]))
            G.add_edge(n1, n2, weight=edge_len)

    if G.number_of_edges() == 0:
        return None

    # ── 4. Prune short terminal branches ─────────────────────────
    G = _prune_graph(G, prune_length)

    if G.number_of_edges() == 0:
        return None

    # ── 5. Extract LineStrings via linemerge ──────────────────────
    edge_lines = [
        ShapelyLineString([n1, n2])
        for n1, n2 in G.edges()
    ]
    merged = linemerge(edge_lines)

    if merged.is_empty:
        return None

    # Normalize output to list of LineStrings
    if isinstance(merged, ShapelyLineString):
        return [merged]
    elif isinstance(merged, ShapelyMultiLineString):
        return list(merged.geoms)
    else:
        return [merged]


def _prune_graph(G, min_branch_length):
    """
    Iteratively remove terminal branches shorter than min_branch_length.

    A terminal branch is the chain from a degree-1 node (leaf) to the
    nearest junction (degree ≥ 3).  If both ends are leaves (an isolated
    short segment), the entire segment is removed if shorter than the
    threshold.

    Parameters
    ----------
    G : networkx.Graph
        Medial axis graph (modified in-place).
    min_branch_length : float
        Minimum branch length to survive (meters).

    Returns
    -------
    networkx.Graph (same object, pruned)
    """
    changed = True
    while changed:
        changed = False

        # Collect current leaves (degree-1 nodes)
        leaves = [n for n in G.nodes() if G.degree(n) == 1]

        for leaf in leaves:
            if not G.has_node(leaf) or G.degree(leaf) != 1:
                continue  # may have been removed in this iteration

            # Walk from leaf along the chain
            path = [leaf]
            path_length = 0.0
            current = leaf
            keep_branch = False

            while True:
                neighbors = [
                    n for n in G.neighbors(current)
                    if n not in path
                ]

                if len(neighbors) == 0:
                    # Dead end (isolated node or completed chain)
                    break

                next_node = neighbors[0]
                edge_w = G.edges[current, next_node].get(
                    "weight", 0.0)
                path_length += edge_w
                path.append(next_node)

                # Check if we've exceeded the threshold
                if path_length >= min_branch_length:
                    keep_branch = True
                    break

                # Check if next_node is a junction (degree ≥ 3)
                if G.degree(next_node) >= 3:
                    # Reached a junction — stop
                    break

                # Continue walking (next_node has degree 2 or 1)
                current = next_node

            if not keep_branch:
                # Remove short branch
                last_node = path[-1]
                if G.has_node(last_node) and G.degree(last_node) >= 3:
                    # Don't remove the junction itself
                    to_remove = path[:-1]
                else:
                    # Both ends are leaves or degree ≤ 2 → remove all
                    to_remove = path

                for node in to_remove:
                    if G.has_node(node):
                        G.remove_node(node)
                        changed = True

    # Clean up isolated nodes (degree 0)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    return G


# ─── THIN FALLBACK METHOD ────────────────────────────────────────────────────

def _extract_centerlines_thin(fc_path, in_cls, cell_size,
                              min_cl_length, tmp_dir):
    """
    Fallback centerline extraction using morphological Thin.

    Used only when shapely/scipy/networkx are not available.
    Produces lower-quality centerlines with possible spurs.

    Workflow:
        1. Select accepted roads
        2. Rasterize → Thin → RasterToPolyline
        3. Clip to original polygons
        4. TrimLine (dangle = 2 × median MBR_W)
        5. Dissolve UNSPLIT_LINES
        6. SpatialJoin (inherit parent polygon attributes)
        7. Compute CL_Len_m + filter by min_cl_length
        8. SmoothLine (PAEK)
    """

    # ── 1. Select accepted roads ─────────────────────────────────
    acc_lyr = arcpy.management.MakeFeatureLayer(
        fc_path, "road_accepted_lyr", "Class = 'Road'")
    n_acc = int(arcpy.management.GetCount(acc_lyr)[0])
    arcpy.AddMessage(f"  Accepted road polygons: {n_acc}")

    if n_acc == 0:
        arcpy.AddWarning(
            "  No accepted roads — skipping centerline extraction.")
        arcpy.management.Delete(acc_lyr)
        return None

    acc_path = os.path.join(tmp_dir, "r_accepted.shp")
    arcpy.management.CopyFeatures(acc_lyr, acc_path)
    arcpy.management.Delete(acc_lyr)

    # Compute median road width for TrimLine threshold
    widths = []
    with arcpy.da.SearchCursor(acc_path, ["MBR_W"]) as sc:
        for row in sc:
            if row[0] is not None and row[0] > 0:
                widths.append(row[0])
    if widths:
        widths.sort()
        median_w = widths[len(widths) // 2]
    else:
        median_w = 5.0
    arcpy.AddMessage(f"  Median road width: {median_w:.2f} m")

    # ── 2. Rasterize → Thin → Polyline ───────────────────────────
    arcpy.management.AddField(acc_path, "_BIN", "SHORT")
    arcpy.management.CalculateField(
        acc_path, "_BIN", "1", "PYTHON3")

    bin_ras_path = os.path.join(tmp_dir, "r_bin_cl.tif")
    arcpy.conversion.PolygonToRaster(
        acc_path, "_BIN", bin_ras_path,
        "CELL_CENTER", "", cell_size)
    arcpy.AddMessage(
        f"  Rasterized at {cell_size:.4f} m cell size")

    arcpy.AddMessage("  Thinning...")
    bin_r = arcpy.Raster(bin_ras_path)
    thinned = arcpy.sa.Thin(
        bin_r,
        background_value="ZERO",
        filter="FILTER",
        corners="ROUND",
        maximum_thickness=0)
    thin_path = os.path.join(tmp_dir, "r_thin.tif")
    thinned.save(thin_path)
    del bin_r, thinned
    arcpy.management.CalculateStatistics(thin_path)

    thin_r = arcpy.Raster(thin_path)
    if thin_r.maximum is None or thin_r.maximum == 0:
        arcpy.AddWarning(
            "  Thinning produced no foreground — skipping.")
        del thin_r
        return None
    del thin_r

    arcpy.AddMessage("  Converting skeleton to polylines...")
    raw_cl_path = os.path.join(tmp_dir, "r_cl_raw.shp")
    arcpy.conversion.RasterToPolyline(
        thin_path, raw_cl_path,
        "ZERO",
        minimum_dangle_length=0,
        simplify="SIMPLIFY",
        raster_field="Value")

    n_raw = int(arcpy.management.GetCount(raw_cl_path)[0])
    arcpy.AddMessage(f"  Raw centerlines: {n_raw}")

    if n_raw == 0:
        arcpy.AddWarning(
            "  No centerlines generated — skipping.")
        return None

    # ── 3. Clip to original polygons ─────────────────────────────
    clipped_path = os.path.join(tmp_dir, "r_cl_clipped.shp")
    arcpy.analysis.Clip(raw_cl_path, acc_path, clipped_path)
    n_clipped = int(arcpy.management.GetCount(clipped_path)[0])
    arcpy.AddMessage(f"  After clip: {n_clipped}")

    if n_clipped == 0:
        arcpy.AddWarning(
            "  No centerlines after clip — skipping.")
        return None

    # ── 4. TrimLine — remove spurs ───────────────────────────────
    trim_len = median_w * 2.0
    arcpy.AddMessage(
        f"  TrimLine: removing dangles < {trim_len:.1f} m "
        f"(2 x median width)...")
    arcpy.edit.TrimLine(
        clipped_path,
        dangle_length=f"{trim_len} Meters",
        delete_shorts="DELETE_SHORT")

    n_trimmed = int(arcpy.management.GetCount(clipped_path)[0])
    arcpy.AddMessage(
        f"  After TrimLine: {n_trimmed} "
        f"(removed {n_clipped - n_trimmed})")

    if n_trimmed == 0:
        arcpy.AddWarning(
            "  All centerlines trimmed — skipping.")
        return None

    # ── 5. Dissolve UNSPLIT — merge connected segments ───────────
    arcpy.AddMessage("  Dissolving connected segments...")
    dissolved_cl_path = os.path.join(tmp_dir, "r_cl_dissolved.shp")
    arcpy.management.Dissolve(
        clipped_path, dissolved_cl_path,
        dissolve_field=None,
        statistics_fields=None,
        multi_part="SINGLE_PART",
        unsplit_lines="UNSPLIT_LINES")

    n_dissolved = int(
        arcpy.management.GetCount(dissolved_cl_path)[0])
    arcpy.AddMessage(
        f"  After dissolve: {n_dissolved} continuous lines "
        f"(from {n_trimmed} segments)")

    if n_dissolved == 0:
        arcpy.AddWarning(
            "  No centerlines after dissolve — skipping.")
        return None

    # ── 6. SpatialJoin — inherit parent polygon attributes ───────
    arcpy.AddMessage("  Joining parent polygon attributes...")
    cl_joined_path = os.path.join(tmp_dir, "r_cl_joined.shp")

    arcpy.analysis.SpatialJoin(
        dissolved_cl_path, acc_path, cl_joined_path,
        join_operation="JOIN_ONE_TO_ONE",
        join_type="KEEP_ALL",
        match_option="INTERSECT")

    # ── 7. Compute CL_Len_m ─────────────────────────────────────
    arcpy.management.AddField(cl_joined_path, "CL_Len_m", "DOUBLE")
    arcpy.management.CalculateGeometryAttributes(
        cl_joined_path,
        [["CL_Len_m", "LENGTH"]],
        length_unit="METERS")

    # ── 8. Filter by minimum length ──────────────────────────────
    arcpy.AddMessage(
        f"  Length filter: min = {min_cl_length} m")

    n_before = int(arcpy.management.GetCount(cl_joined_path)[0])
    n_short_removed = 0
    with arcpy.da.UpdateCursor(
        cl_joined_path, ["CL_Len_m"]
    ) as ucur:
        for row in ucur:
            if row[0] is None or row[0] < min_cl_length:
                ucur.deleteRow()
                n_short_removed += 1

    n_after_len = int(
        arcpy.management.GetCount(cl_joined_path)[0])
    arcpy.AddMessage(
        f"  Removed {n_short_removed} short segments "
        f"({n_after_len} remaining)")

    if n_after_len == 0:
        arcpy.AddWarning(
            "  All centerlines below minimum length — skipping.")
        return None

    # ── 9. Smooth final centerlines ──────────────────────────────
    arcpy.AddMessage("  Smoothing centerlines (PAEK)...")
    smooth_path = os.path.join(tmp_dir, "r_cl_smooth.shp")
    arcpy.cartography.SmoothLine(
        cl_joined_path, smooth_path,
        "PAEK", cell_size * 20)

    n_smooth = int(arcpy.management.GetCount(smooth_path)[0])
    if n_smooth > 0:
        cl_joined_path = smooth_path
        arcpy.AddMessage(f"  Smoothed: {n_smooth} centerlines")
    else:
        arcpy.AddMessage("  Smooth produced no output — "
                         "using unsmoothed.")

    # ── Clean up — match road attribute table ────────────────────
    keep_fields = {
        "FID", "Shape", "OID",
        "RoadID", "Area_m2", "Perim_m",
        "MBR_W", "MBR_L", "MBR_Azim", "LW_Ratio", "Compact",
        "MEAN_nDSM", "MEAN_Therm", "Class", "Reason",
        "CL_Len_m",
    }
    drop_fields = []
    for f in arcpy.ListFields(cl_joined_path):
        if (f.name not in keep_fields
                and not f.required
                and f.name.lower() not in ("shape_leng",
                                            "smoothline")):
            drop_fields.append(f.name)
    if drop_fields:
        try:
            arcpy.management.DeleteField(
                cl_joined_path, drop_fields)
        except Exception:
            pass

    n_final = int(arcpy.management.GetCount(cl_joined_path)[0])
    arcpy.AddMessage(f"  Final centerlines (Thin): {n_final}")

    return cl_joined_path


# ═══════════════════════════════════════════════════════════════════════════════
# FULL ROAD PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_road_pipeline(in_cls, road_val, in_ndsm, in_therm,
                      in_aoi, out_crs_text, cs_opt, cs_custom,
                      min_area, max_ndsm, reject_null_ndsm,
                      maj_nbr, maj_passes, simp_tol,
                      min_length, min_width,
                      do_centerlines, min_cl_length,
                      max_hole_area=20.0, prune_factor=2.0,
                      densify_factor=5.0,
                      do_centroids=True, out_folder=""):
    """
    Execute the complete road post-processing pipeline.

    Called by RoadAnalysis.execute() in PostProcessing.pyt.

    Parameters
    ----------
    in_cls : str          — Classified raster path
    road_val : int        — Road class value
    in_ndsm : str         — nDSM raster path
    in_therm : str|None   — Thermal raster path (optional)
    in_aoi : str|None     — AOI feature/raster path (optional)
    out_crs_text : str|None — Output CRS (valueAsText)
    cs_opt : str          — Cell size option
    cs_custom : float|None — Custom cell size
    min_area : float      — Min road segment area (m²)
    max_ndsm : float      — Max mean nDSM height (m)
    reject_null_ndsm : bool — Reject features with NULL nDSM
    maj_nbr : str         — Majority neighborhood (FOUR/EIGHT/Skip)
    maj_passes : int      — Majority filter passes
    simp_tol : float      — Simplify tolerance (m)
    min_length : float    — Min MBR length (m)
    min_width : float     — Min MBR width (m)
    do_centerlines : bool — Generate centerline polylines
    min_cl_length : float — Min centerline segment length (m)
    max_hole_area : float — Fill holes smaller than this (m²)
    prune_factor : float  — Prune branches < factor × median width
    densify_factor : float — Densify boundary at factor × cell size
    do_centroids : bool   — Generate centroid points
    out_folder : str      — Output folder

    Returns
    -------
    tuple (str, str|None, str|None) :
        (results_shapefile, centerlines_shapefile, points_shapefile)
    """
    PREFIX = "road"
    LABEL = "Road"
    ID_FIELD = "RoadID"

    # ── INIT ──────────────────────────────────────────────────────
    arcpy.env.overwriteOutput = True
    arcpy.CheckOutExtension("Spatial")
    arcpy.CheckOutExtension("3D")

    params_dict = {
        "cell_size_option": cs_opt,
        "cell_size_custom": cs_custom,
        "min_area_m2": min_area,
        "max_ndsm_m": max_ndsm,
        "reject_null_ndsm": reject_null_ndsm,
        "majority_neighborhood": maj_nbr,
        "majority_passes": (maj_passes if maj_nbr != "Skip" else 0),
        "simplify_tolerance_m": simp_tol,
        "min_mbr_length_m": min_length,
        "min_mbr_width_m": min_width,
        "generate_centerlines": do_centerlines,
        "min_centerline_length_m": min_cl_length,
        "max_hole_area_m2": max_hole_area,
        "prune_factor": prune_factor,
        "densify_factor": densify_factor,
        "generate_centroids": do_centroids,
        "output_folder": out_folder,
    }

    sub_dir, tmp_dir, run_log, t0 = common.init_run(
        out_folder, PREFIX, in_cls, road_val,
        in_ndsm, in_therm, in_aoi, params_dict)

    # Clean old centerline output too
    cl_old = os.path.join(sub_dir, "road_centerlines.shp")
    if arcpy.Exists(cl_old):
        try:
            arcpy.management.Delete(cl_old)
        except Exception:
            pass

    try:
        # ── PHASE 0: VALIDATE ────────────────────────────────────
        env = common.validate_inputs(
            in_cls, in_ndsm, in_therm, in_aoi,
            out_crs_text, cs_opt, cs_custom,
            road_val, LABEL, run_log)

        # ── PHASE 1: EXTRACT ─────────────────────────────────────
        bin_path = common.extract_class_pixels(
            in_cls, road_val, LABEL, tmp_dir)

        # ── PHASE 2: MAJORITY FILTER ─────────────────────────────
        filt_path = common.run_majority_filter(
            bin_path, maj_nbr, maj_passes, tmp_dir)

        # ── PHASE 3: VECTORIZE ───────────────────────────────────
        raw_path, n_raw = common.vectorize_binary(filt_path, tmp_dir)

        if n_raw == 0:
            arcpy.AddWarning(
                f"No pixels with value {road_val} found.")
            empty_path = os.path.join(sub_dir, "road_results.shp")
            arcpy.management.CopyFeatures(raw_path, empty_path)
            common.write_log(run_log, sub_dir, "road_log.json")
            return empty_path, None, None

        # ── PHASE 4a: SIMPLIFY + DISSOLVE (road-specific) ────────
        diss_path, n_diss, lost_simp, n_pre_dissolve = \
            simplify_and_dissolve(raw_path, simp_tol, tmp_dir)

        if diss_path is None:
            empty_path = os.path.join(sub_dir, "road_results.shp")
            arcpy.management.CopyFeatures(raw_path, empty_path)
            common.write_log(run_log, sub_dir, "road_log.json")
            return empty_path, None, None

        run_log["results"]["n_pre_dissolve"] = n_pre_dissolve
        run_log["results"]["n_after_dissolve"] = n_diss

        # ── PHASE 5: GEOMETRY & MBR ──────────────────────────────
        common.compute_geometry_mbr(diss_path, tmp_dir)

        # ── PHASE 6: ENRICHMENT ──────────────────────────────────
        enriched_path = os.path.join(tmp_dir, "r_enriched.shp")
        arcpy.management.CopyFeatures(diss_path, enriched_path)

        n_null_h = common.enrich_attributes(
            enriched_path, in_ndsm, in_therm, ID_FIELD, tmp_dir)

        # ── PHASE 7: CLASSIFY (road-specific) ────────────────────
        n_accepted, reject_counts = classify_roads(
            enriched_path, min_area, max_ndsm,
            min_length, min_width, reject_null_ndsm)

        # ── PHASE 7.5: CENTERLINE (road-specific) ────────────────
        cl_tmp_path = None
        if do_centerlines and n_accepted > 0:
            cl_tmp_path = extract_centerlines(
                enriched_path, in_cls,
                env["cell_size"], min_cl_length, tmp_dir,
                max_hole_area, prune_factor, densify_factor)
            if cl_tmp_path:
                run_log["results"]["centerline_segments"] = \
                    int(arcpy.management.GetCount(cl_tmp_path)[0])
                run_log["results"]["centerline_method"] = (
                    "voronoi" if HAS_VORONOI else "thin")

        # ── Set output CRS (all raster operations complete) ──────
        common.set_output_crs(env["tgt_sr"])

        # ── PHASE 8: EXPORT ──────────────────────────────────────
        final_path, pts_path = common.export_results(
            enriched_path, out_folder, PREFIX, do_centroids,
            run_log, in_aoi, env["aoi_m2"],
            road_val, LABEL,
            n_raw, lost_simp, n_diss, n_accepted,
            reject_counts, n_null_h, env["tgt_sr"], t0)

        # ── Export centerlines ───────────────────────────────────
        cl_path = None
        if cl_tmp_path:
            cl_path = os.path.join(sub_dir, "road_centerlines.shp")
            arcpy.management.CopyFeatures(cl_tmp_path, cl_path)
            n_cl = int(arcpy.management.GetCount(cl_path)[0])
            arcpy.AddMessage(f"  Centerlines: {cl_path}")
            arcpy.AddMessage(f"  Centerline segments: {n_cl}")
            run_log["outputs"]["road_centerlines"] = cl_path
            common.write_log(
                run_log, sub_dir, "road_log.json")

        return final_path, cl_path, pts_path

    except Exception as e:
        arcpy.AddError(f"Failed: {str(e)}")
        arcpy.AddError(traceback.format_exc())
        raise

    finally:
        common.cleanup_temp(tmp_dir)
        common.reset_environment()
