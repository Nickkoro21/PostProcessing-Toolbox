# -*- coding: utf-8 -*-
"""
PostProcessing.pyt — Land Cover Post-Processing Toolbox
========================================================
Master Thesis: Semantic Segmentation of Multispectral Drone Imagery
Sensor: MicaSense Altum-PT  |  ArcGIS Pro 3.6.2

Developed by Nikolaos Koroniadis
MSc Geography and Applied Geoinformatics, University of the Aegean
Remote Sensing & GIS Research Group (https://rsgis.aegean.gr/)

Architecture:
    PostProcessing.pyt          ← this file (UI + parameter logic)
    _postproc_common.py         ← shared pipeline phases
    _vehicle_core.py            ← vehicle-specific phases + orchestrator
    _building_core.py           ← building-specific phases + orchestrator
    _tree_core.py               ← tree-specific phases + orchestrator
    _road_core.py               ← road-specific phases + orchestrator

Companion metadata:
    PostProcessing.VehicleAnalysis.pyt.xml
    PostProcessing.BuildingAnalysis.pyt.xml
    PostProcessing.TreeAnalysis.pyt.xml
    PostProcessing.RoadAnalysis.pyt.xml
"""

import arcpy
import os
import sys

# ── Ensure sibling modules are importable ─────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


class Toolbox(object):
    def __init__(self):
        self.label = "Post-Processing Toolbox"
        self.alias = "PostProc"
        self.tools = [VehicleAnalysis, BuildingAnalysis, TreeAnalysis,
                      RoadAnalysis]


# ═══════════════════════════════════════════════════════════════════════════════
#  1. VEHICLE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class VehicleAnalysis(object):

    def __init__(self):
        self.label = "1. Vehicle Analysis"
        self.description = (
            "Vehicle detection post-processing pipeline. "
            "See the help page (?) for full documentation."
        )
        self.canRunInBackground = False
        self.category = "Vehicle"

    # ----------------------------------------------------------
    # PARAMETERS
    # ----------------------------------------------------------
    def getParameterInfo(self):

        # 0: Classified raster (single-band, integer classes)
        p0 = arcpy.Parameter(
            displayName="Classified Raster",
            name="in_classified",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input"
        )

        # 1: Vehicle class value (populated from raster)
        p1 = arcpy.Parameter(
            displayName="Vehicle Class Value",
            name="vehicle_class",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        p1.enabled = False

        # 2: nDSM
        p2 = arcpy.Parameter(
            displayName="nDSM Raster (m)",
            name="in_ndsm",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input"
        )

        # 3: Thermal
        p3 = arcpy.Parameter(
            displayName="Thermal Raster (\u00b0C)",
            name="in_thermal",
            datatype="GPRasterLayer",
            parameterType="Optional",
            direction="Input"
        )

        # 4: AOI
        p4 = arcpy.Parameter(
            displayName="Area of Interest",
            name="in_aoi",
            datatype=["GPFeatureLayer", "GPRasterLayer",
                       "DEFeatureClass", "DERasterDataset",
                       "DEShapefile"],
            parameterType="Optional",
            direction="Input"
        )

        # 5: Output CRS
        p5 = arcpy.Parameter(
            displayName="Output Coordinate System",
            name="out_crs",
            datatype="GPCoordinateSystem",
            parameterType="Optional",
            direction="Input"
        )

        # 6: Cell size option
        p6 = arcpy.Parameter(
            displayName="Processing Cell Size",
            name="cell_size_option",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p6.filter.type = "ValueList"
        p6.filter.list = [
            "Classified raster cell size",
            "Maximum of inputs (coarsest)",
            "nDSM cell size",
            "Custom"
        ]
        p6.value = "Classified raster cell size"

        # 7: Custom cell size
        p7 = arcpy.Parameter(
            displayName="Custom Cell Size (m)",
            name="cell_size_custom",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )
        p7.enabled = False

        # 8: Min area
        p8 = arcpy.Parameter(
            displayName="Minimum Vehicle Area (m\u00b2)",
            name="min_area",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p8.value = 1.5
        p8.filter.type = "Range"
        p8.filter.list = [0.1, 100.0]

        # 9: Max area
        p9 = arcpy.Parameter(
            displayName="Maximum Vehicle Area (m\u00b2)",
            name="max_area",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p9.value = 12.0
        p9.filter.type = "Range"
        p9.filter.list = [1.0, 500.0]

        # 10: Min L/W
        p10 = arcpy.Parameter(
            displayName="Minimum L/W Ratio",
            name="min_lw_ratio",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p10.value = 1.5
        p10.filter.type = "Range"
        p10.filter.list = [1.0, 10.0]

        # 11: Majority neighborhood
        p11 = arcpy.Parameter(
            displayName="Majority Filter Neighborhood",
            name="majority_neighborhood",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p11.filter.type = "ValueList"
        p11.filter.list = ["FOUR", "EIGHT", "Skip"]
        p11.value = "FOUR"

        # 12: Majority passes
        p12 = arcpy.Parameter(
            displayName="Majority Filter Passes",
            name="majority_passes",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        p12.value = 1
        p12.filter.type = "Range"
        p12.filter.list = [1, 5]

        # 13: Simplify tolerance
        p13 = arcpy.Parameter(
            displayName="Simplify Tolerance (m)",
            name="simplify_tol",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p13.value = 0.10
        p13.filter.type = "Range"
        p13.filter.list = [0.01, 5.0]

        # 14: Centroids
        p14 = arcpy.Parameter(
            displayName="Generate Centroid Points",
            name="do_centroids",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input"
        )
        p14.value = True

        # 15: Output folder
        p15 = arcpy.Parameter(
            displayName="Output Folder",
            name="out_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )

        # 16-17: Derived outputs
        p16 = arcpy.Parameter(
            displayName="Output Vehicle Results",
            name="out_results",
            datatype="DEFeatureClass",
            parameterType="Derived",
            direction="Output"
        )
        p17 = arcpy.Parameter(
            displayName="Output Vehicle Points",
            name="out_pts",
            datatype="DEFeatureClass",
            parameterType="Derived",
            direction="Output"
        )

        return [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                p10, p11, p12, p13, p14, p15, p16, p17]

    # ----------------------------------------------------------
    # DYNAMIC UPDATES
    # ----------------------------------------------------------
    def updateParameters(self, parameters):
        # Populate vehicle class dropdown from raster unique values
        if parameters[0].altered and parameters[0].value:
            try:
                r = arcpy.Raster(str(parameters[0].value))
                if r.isInteger and r.bandCount == 1:
                    vals = []
                    ras_path = str(parameters[0].valueAsText)
                    try:
                        with arcpy.da.SearchCursor(
                            ras_path, ["Value"]
                        ) as scur:
                            for row in scur:
                                if row[0] is not None and row[0] > 0:
                                    vals.append(int(row[0]))
                    except Exception:
                        vals = []

                    if len(vals) >= 2:
                        parameters[1].enabled = True
                        parameters[1].filter.type = "ValueList"
                        parameters[1].filter.list = sorted(vals)
                        if (parameters[1].value is not None and
                                int(parameters[1].value) not in vals):
                            parameters[1].value = None
                    elif len(vals) == 1:
                        parameters[1].enabled = True
                        parameters[1].filter.type = "ValueList"
                        parameters[1].filter.list = vals
                    else:
                        parameters[1].enabled = True
                        parameters[1].filter.type = "Range"
                        parameters[1].filter.list = [1, 9999]
                else:
                    parameters[1].enabled = False
                    parameters[1].value = None
                del r
            except Exception:
                parameters[1].enabled = True
                parameters[1].filter.type = "Range"
                parameters[1].filter.list = [1, 9999]
        elif not parameters[0].value:
            parameters[1].enabled = False
            parameters[1].value = None

        # Custom cell size toggle
        if parameters[6].altered:
            if str(parameters[6].value) == "Custom":
                parameters[7].enabled = True
            else:
                parameters[7].enabled = False
                parameters[7].value = None

        # Majority skip toggle
        if parameters[11].altered:
            if str(parameters[11].value) == "Skip":
                parameters[12].enabled = False
                parameters[12].value = 1
            else:
                parameters[12].enabled = True
        return

    # ----------------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------------
    def updateMessages(self, parameters):
        # Single band check
        if parameters[0].altered and parameters[0].value:
            try:
                r = arcpy.Raster(str(parameters[0].value))
                if r.bandCount != 1:
                    parameters[0].setErrorMessage(
                        f"Expected single-band raster, "
                        f"got {r.bandCount} bands.")
                elif not r.isInteger:
                    parameters[0].setErrorMessage(
                        "Raster must have integer pixel type "
                        "(class values).")
                del r
            except Exception:
                pass

        # Vehicle class required
        if (parameters[0].value and
                parameters[1].enabled and
                parameters[1].value is None):
            parameters[1].setErrorMessage(
                "Select the class value that represents vehicles.")

        # Area check
        if (parameters[8].value is not None and
                parameters[9].value is not None):
            if float(parameters[8].value) >= float(parameters[9].value):
                parameters[9].setErrorMessage(
                    "Max area must be greater than min area.")

        # Custom cell size
        if (parameters[6].value and
                str(parameters[6].value) == "Custom"):
            if (not parameters[7].value or
                    float(parameters[7].value) <= 0):
                parameters[7].setErrorMessage(
                    "Enter a positive cell size in meters.")

        # EIGHT + many passes warning
        if (parameters[11].value and
                str(parameters[11].value) == "EIGHT" and
                parameters[12].value is not None and
                int(parameters[12].value) > 1):
            parameters[12].setWarningMessage(
                "EIGHT neighborhood with multiple passes "
                "may erode small vehicles.")

        # Overwrite warning
        if parameters[15].altered and parameters[15].value:
            rp = os.path.join(
                str(parameters[15].valueAsText),
                "vehicles", "vehicle_results.shp")
            if os.path.exists(rp):
                parameters[15].setWarningMessage(
                    "Existing results will be OVERWRITTEN.")
        return

    # ----------------------------------------------------------
    # LICENSE
    # ----------------------------------------------------------
    def isLicensed(self):
        try:
            return (arcpy.CheckExtension("Spatial") == "Available"
                    and arcpy.CheckExtension("3D") == "Available")
        except Exception:
            return False

    # ----------------------------------------------------------
    # EXECUTE
    # ----------------------------------------------------------
    def execute(self, parameters, messages):
        import _vehicle_core

        final_path, pts_path = _vehicle_core.run_vehicle_pipeline(
            in_cls=str(parameters[0].valueAsText),
            veh_val=int(parameters[1].value),
            in_ndsm=str(parameters[2].valueAsText),
            in_therm=(str(parameters[3].valueAsText)
                      if parameters[3].value else None),
            in_aoi=(str(parameters[4].valueAsText)
                    if parameters[4].value else None),
            out_crs_text=(parameters[5].valueAsText
                          if parameters[5].value else None),
            cs_opt=str(parameters[6].value),
            cs_custom=(float(parameters[7].value)
                       if parameters[7].value else None),
            min_area=float(parameters[8].value),
            max_area=float(parameters[9].value),
            min_lw=float(parameters[10].value),
            maj_nbr=str(parameters[11].value),
            maj_passes=int(parameters[12].value),
            simp_tol=float(parameters[13].value),
            do_centroids=bool(parameters[14].value),
            out_folder=str(parameters[15].valueAsText),
        )

        parameters[16].value = final_path
        if pts_path:
            parameters[17].value = pts_path

    def postExecute(self, parameters):
        return

# ═══════════════════════════════════════════════════════════════════════════════
#  2. BUILDING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class BuildingAnalysis(object):

    def __init__(self):
        self.label = "2. Building Analysis"
        self.description = (
            "Building footprint post-processing pipeline. "
            "See the help page (?) for full documentation."
        )
        self.canRunInBackground = False
        self.category = "Building"

    # ----------------------------------------------------------
    # PARAMETERS
    # ----------------------------------------------------------
    def getParameterInfo(self):

        # 0: Classified raster
        p0 = arcpy.Parameter(
            displayName="Classified Raster",
            name="in_classified",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input"
        )

        # 1: Building class value
        p1 = arcpy.Parameter(
            displayName="Building Class Value",
            name="building_class",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        p1.enabled = False

        # 2: nDSM
        p2 = arcpy.Parameter(
            displayName="nDSM Raster (m)",
            name="in_ndsm",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input"
        )

        # 3: Thermal
        p3 = arcpy.Parameter(
            displayName="Thermal Raster (\u00b0C)",
            name="in_thermal",
            datatype="GPRasterLayer",
            parameterType="Optional",
            direction="Input"
        )

        # 4: AOI
        p4 = arcpy.Parameter(
            displayName="Area of Interest",
            name="in_aoi",
            datatype=["GPFeatureLayer", "GPRasterLayer",
                       "DEFeatureClass", "DERasterDataset",
                       "DEShapefile"],
            parameterType="Optional",
            direction="Input"
        )

        # 5: Output CRS
        p5 = arcpy.Parameter(
            displayName="Output Coordinate System",
            name="out_crs",
            datatype="GPCoordinateSystem",
            parameterType="Optional",
            direction="Input"
        )

        # 6: Cell size option
        p6 = arcpy.Parameter(
            displayName="Processing Cell Size",
            name="cell_size_option",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p6.filter.type = "ValueList"
        p6.filter.list = [
            "Classified raster cell size",
            "Maximum of inputs (coarsest)",
            "nDSM cell size",
            "Custom"
        ]
        p6.value = "Classified raster cell size"

        # 7: Custom cell size
        p7 = arcpy.Parameter(
            displayName="Custom Cell Size (m)",
            name="cell_size_custom",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )
        p7.enabled = False

        # 8: Min area
        p8 = arcpy.Parameter(
            displayName="Minimum Building Area (m\u00b2)",
            name="min_area",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p8.value = 5.0
        p8.filter.type = "Range"
        p8.filter.list = [0.1, 5000.0]

        # 9: Max area
        p9 = arcpy.Parameter(
            displayName="Maximum Building Area (m\u00b2)",
            name="max_area",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p9.value = 1000.0
        p9.filter.type = "Range"
        p9.filter.list = [1.0, 10000.0]

        # 10: Min nDSM
        p10 = arcpy.Parameter(
            displayName="Minimum nDSM Height (m)",
            name="min_ndsm",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p10.value = 1.0
        p10.filter.type = "Range"
        p10.filter.list = [0.0, 100.0]

        # 11: Reject NULL nDSM
        p11 = arcpy.Parameter(
            displayName="Reject Features with NULL nDSM",
            name="reject_null_ndsm",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input"
        )
        p11.value = False

        # 12: Majority neighborhood
        p12 = arcpy.Parameter(
            displayName="Majority Filter Neighborhood",
            name="majority_neighborhood",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p12.filter.type = "ValueList"
        p12.filter.list = ["FOUR", "EIGHT", "Skip"]
        p12.value = "FOUR"

        # 13: Majority passes
        p13 = arcpy.Parameter(
            displayName="Majority Filter Passes",
            name="majority_passes",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        p13.value = 1
        p13.filter.type = "Range"
        p13.filter.list = [1, 5]

        # 14: Regularize method
        p14 = arcpy.Parameter(
            displayName="Regularize Method",
            name="regularize_method",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p14.filter.type = "ValueList"
        p14.filter.list = [
            "RIGHT_ANGLES",
            "RIGHT_ANGLES_AND_DIAGONALS",
            "ANY_ANGLE"
        ]
        p14.value = "RIGHT_ANGLES"

        # 15: Regularize tolerance
        p15 = arcpy.Parameter(
            displayName="Regularize Tolerance (m)",
            name="regularize_tol",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p15.value = 0.20
        p15.filter.type = "Range"
        p15.filter.list = [0.01, 5.0]

        # 16: Centroids
        p16 = arcpy.Parameter(
            displayName="Generate Centroid Points",
            name="do_centroids",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input"
        )
        p16.value = True

        # 17: Output folder
        p17 = arcpy.Parameter(
            displayName="Output Folder",
            name="out_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )

        # 18-19: Derived outputs
        p18 = arcpy.Parameter(
            displayName="Output Building Results",
            name="out_results",
            datatype="DEFeatureClass",
            parameterType="Derived",
            direction="Output"
        )
        p19 = arcpy.Parameter(
            displayName="Output Building Points",
            name="out_pts",
            datatype="DEFeatureClass",
            parameterType="Derived",
            direction="Output"
        )

        return [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                p10, p11, p12, p13, p14, p15, p16, p17, p18, p19]

    # ----------------------------------------------------------
    # DYNAMIC UPDATES
    # ----------------------------------------------------------
    def updateParameters(self, parameters):
        # Populate building class dropdown from raster unique values
        if parameters[0].altered and parameters[0].value:
            try:
                r = arcpy.Raster(str(parameters[0].value))
                if r.isInteger and r.bandCount == 1:
                    vals = []
                    ras_path = str(parameters[0].valueAsText)
                    try:
                        with arcpy.da.SearchCursor(
                            ras_path, ["Value"]
                        ) as scur:
                            for row in scur:
                                if row[0] is not None and row[0] > 0:
                                    vals.append(int(row[0]))
                    except Exception:
                        vals = []

                    if len(vals) >= 2:
                        parameters[1].enabled = True
                        parameters[1].filter.type = "ValueList"
                        parameters[1].filter.list = sorted(vals)
                        if (parameters[1].value is not None and
                                int(parameters[1].value) not in vals):
                            parameters[1].value = None
                    elif len(vals) == 1:
                        parameters[1].enabled = True
                        parameters[1].filter.type = "ValueList"
                        parameters[1].filter.list = vals
                    else:
                        parameters[1].enabled = True
                        parameters[1].filter.type = "Range"
                        parameters[1].filter.list = [1, 9999]
                else:
                    parameters[1].enabled = False
                    parameters[1].value = None
                del r
            except Exception:
                parameters[1].enabled = True
                parameters[1].filter.type = "Range"
                parameters[1].filter.list = [1, 9999]
        elif not parameters[0].value:
            parameters[1].enabled = False
            parameters[1].value = None

        # Custom cell size toggle
        if parameters[6].altered:
            if str(parameters[6].value) == "Custom":
                parameters[7].enabled = True
            else:
                parameters[7].enabled = False
                parameters[7].value = None

        # Majority skip toggle
        if parameters[12].altered:
            if str(parameters[12].value) == "Skip":
                parameters[13].enabled = False
                parameters[13].value = 1
            else:
                parameters[13].enabled = True
        return

    # ----------------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------------
    def updateMessages(self, parameters):
        # Single band check
        if parameters[0].altered and parameters[0].value:
            try:
                r = arcpy.Raster(str(parameters[0].value))
                if r.bandCount != 1:
                    parameters[0].setErrorMessage(
                        f"Expected single-band raster, "
                        f"got {r.bandCount} bands.")
                elif not r.isInteger:
                    parameters[0].setErrorMessage(
                        "Raster must have integer pixel type "
                        "(class values).")
                del r
            except Exception:
                pass

        # Building class required
        if (parameters[0].value and
                parameters[1].enabled and
                parameters[1].value is None):
            parameters[1].setErrorMessage(
                "Select the class value that represents buildings.")

        # Area check
        if (parameters[8].value is not None and
                parameters[9].value is not None):
            if float(parameters[8].value) >= float(parameters[9].value):
                parameters[9].setErrorMessage(
                    "Max area must be greater than min area.")

        # Custom cell size
        if (parameters[6].value and
                str(parameters[6].value) == "Custom"):
            if (not parameters[7].value or
                    float(parameters[7].value) <= 0):
                parameters[7].setErrorMessage(
                    "Enter a positive cell size in meters.")

        # EIGHT + many passes warning
        if (parameters[12].value and
                str(parameters[12].value) == "EIGHT" and
                parameters[13].value is not None and
                int(parameters[13].value) > 1):
            parameters[13].setWarningMessage(
                "EIGHT neighborhood with multiple passes "
                "may erode small buildings.")

        # Overwrite warning
        if parameters[17].altered and parameters[17].value:
            rp = os.path.join(
                str(parameters[17].valueAsText),
                "buildings", "building_results.shp")
            if os.path.exists(rp):
                parameters[17].setWarningMessage(
                    "Existing results will be OVERWRITTEN.")
        return

    # ----------------------------------------------------------
    # LICENSE
    # ----------------------------------------------------------
    def isLicensed(self):
        try:
            return (arcpy.CheckExtension("Spatial") == "Available"
                    and arcpy.CheckExtension("3D") == "Available")
        except Exception:
            return False

    # ----------------------------------------------------------
    # EXECUTE
    # ----------------------------------------------------------
    def execute(self, parameters, messages):
        import _building_core

        final_path, pts_path = _building_core.run_building_pipeline(
            in_cls=str(parameters[0].valueAsText),
            bld_val=int(parameters[1].value),
            in_ndsm=str(parameters[2].valueAsText),
            in_therm=(str(parameters[3].valueAsText)
                      if parameters[3].value else None),
            in_aoi=(str(parameters[4].valueAsText)
                    if parameters[4].value else None),
            out_crs_text=(parameters[5].valueAsText
                          if parameters[5].value else None),
            cs_opt=str(parameters[6].value),
            cs_custom=(float(parameters[7].value)
                       if parameters[7].value else None),
            min_area=float(parameters[8].value),
            max_area=float(parameters[9].value),
            min_ndsm=float(parameters[10].value),
            reject_null_ndsm=bool(parameters[11].value),
            maj_nbr=str(parameters[12].value),
            maj_passes=int(parameters[13].value),
            reg_method=str(parameters[14].value),
            reg_tolerance=float(parameters[15].value),
            do_centroids=bool(parameters[16].value),
            out_folder=str(parameters[17].valueAsText),
        )

        parameters[18].value = final_path
        if pts_path:
            parameters[19].value = pts_path

    def postExecute(self, parameters):
        return


# ═══════════════════════════════════════════════════════════════════════════════
#  3. TREE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class TreeAnalysis(object):

    def __init__(self):
        self.label = "3. Tree Analysis"
        self.description = (
            "Tree crown detection and analysis pipeline. "
            "Optional watershed and grid-based crown separation. "
            "See the help page (?) for full documentation."
        )
        self.canRunInBackground = False
        self.category = "Tree"

    # ----------------------------------------------------------
    # PARAMETERS
    # ----------------------------------------------------------
    def getParameterInfo(self):

        # 0: Classified raster (single-band, integer classes)
        p0 = arcpy.Parameter(
            displayName="Classified Raster",
            name="in_classified",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input"
        )

        # 1: Tree class value (populated from raster)
        p1 = arcpy.Parameter(
            displayName="Tree Class Value",
            name="tree_class",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        p1.enabled = False

        # 2: nDSM
        p2 = arcpy.Parameter(
            displayName="nDSM Raster (m)",
            name="in_ndsm",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input"
        )

        # 3: Thermal
        p3 = arcpy.Parameter(
            displayName="Thermal Raster (\u00b0C)",
            name="in_thermal",
            datatype="GPRasterLayer",
            parameterType="Optional",
            direction="Input"
        )

        # 4: AOI
        p4 = arcpy.Parameter(
            displayName="Area of Interest",
            name="in_aoi",
            datatype=["GPFeatureLayer", "GPRasterLayer",
                       "DEFeatureClass", "DERasterDataset",
                       "DEShapefile"],
            parameterType="Optional",
            direction="Input"
        )

        # 5: Output CRS
        p5 = arcpy.Parameter(
            displayName="Output Coordinate System",
            name="out_crs",
            datatype="GPCoordinateSystem",
            parameterType="Optional",
            direction="Input"
        )

        # 6: Cell size option
        p6 = arcpy.Parameter(
            displayName="Processing Cell Size",
            name="cell_size_option",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p6.filter.type = "ValueList"
        p6.filter.list = [
            "Classified raster cell size",
            "Maximum of inputs (coarsest)",
            "nDSM cell size",
            "Custom"
        ]
        p6.value = "Classified raster cell size"

        # 7: Custom cell size
        p7 = arcpy.Parameter(
            displayName="Custom Cell Size (m)",
            name="cell_size_custom",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )
        p7.enabled = False

        # 8: Min area
        p8 = arcpy.Parameter(
            displayName="Minimum Tree Crown Area (m\u00b2)",
            name="min_area",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p8.value = 2.0
        p8.filter.type = "Range"
        p8.filter.list = [0.1, 100.0]

        # 9: Max area
        p9 = arcpy.Parameter(
            displayName="Maximum Tree Crown Area (m\u00b2)",
            name="max_area",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p9.value = 260.0
        p9.filter.type = "Range"
        p9.filter.list = [10.0, 10000.0]

        # 10: Min nDSM
        p10 = arcpy.Parameter(
            displayName="Minimum nDSM Height (m)",
            name="min_ndsm",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p10.value = 1.0
        p10.filter.type = "Range"
        p10.filter.list = [0.0, 20.0]

        # 11: Reject NULL nDSM
        p11 = arcpy.Parameter(
            displayName="Reject Features with NULL nDSM",
            name="reject_null_ndsm",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input"
        )
        p11.value = False

        # 12: Majority neighborhood
        p12 = arcpy.Parameter(
            displayName="Majority Filter Neighborhood",
            name="majority_neighborhood",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p12.filter.type = "ValueList"
        p12.filter.list = ["FOUR", "EIGHT", "Skip"]
        p12.value = "FOUR"

        # 13: Majority passes
        p13 = arcpy.Parameter(
            displayName="Majority Filter Passes",
            name="majority_passes",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        p13.value = 1
        p13.filter.type = "Range"
        p13.filter.list = [1, 5]

        # 14: Smooth tolerance
        p14 = arcpy.Parameter(
            displayName="Smooth Tolerance (m) \u2014 PAEK",
            name="smooth_tol",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p14.value = 0.30
        p14.filter.type = "Range"
        p14.filter.list = [0.01, 5.0]

        # 15: Crown Separation Method (mutually exclusive)
        p15 = arcpy.Parameter(
            displayName="Crown Separation Method",
            name="crown_method",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p15.filter.type = "ValueList"
        p15.filter.list = ["None", "Watershed", "Grid"]
        p15.value = "None"

        # 16: Min Area for Watershed (m2)
        p16 = arcpy.Parameter(
            displayName="Min Area for Watershed (m\u00b2)",
            name="ws_min_area",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )
        p16.value = 120.0
        p16.enabled = False
        p16.filter.type = "Range"
        p16.filter.list = [10.0, 5000.0]

        # 17: Watershed Search Radius (m)
        p17 = arcpy.Parameter(
            displayName="Watershed Search Radius (m)",
            name="ws_search_radius",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )
        p17.value = 6.0
        p17.enabled = False
        p17.filter.type = "Range"
        p17.filter.list = [1.0, 20.0]

        # 18: Grid cell size
        p18 = arcpy.Parameter(
            displayName="Grid Cell Size (m)",
            name="grid_size",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )
        p18.value = 5.0
        p18.enabled = False
        p18.filter.type = "Range"
        p18.filter.list = [1.0, 50.0]

        # 19: Centroids
        p19 = arcpy.Parameter(
            displayName="Generate Centroid Points",
            name="do_centroids",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input"
        )
        p19.value = True

        # 20: Output folder
        p20 = arcpy.Parameter(
            displayName="Output Folder",
            name="out_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )

        # 21-22: Derived outputs
        p21 = arcpy.Parameter(
            displayName="Output Tree Results",
            name="out_results",
            datatype="DEFeatureClass",
            parameterType="Derived",
            direction="Output"
        )
        p22 = arcpy.Parameter(
            displayName="Output Tree Points",
            name="out_pts",
            datatype="DEFeatureClass",
            parameterType="Derived",
            direction="Output"
        )

        return [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                p10, p11, p12, p13, p14, p15, p16, p17,
                p18, p19, p20, p21, p22]

    # ----------------------------------------------------------
    # DYNAMIC UPDATES
    # ----------------------------------------------------------
    def updateParameters(self, parameters):
        # Populate tree class dropdown from raster unique values
        if parameters[0].altered and parameters[0].value:
            try:
                r = arcpy.Raster(str(parameters[0].value))
                if r.isInteger and r.bandCount == 1:
                    vals = []
                    ras_path = str(parameters[0].valueAsText)
                    try:
                        with arcpy.da.SearchCursor(
                            ras_path, ["Value"]
                        ) as scur:
                            for row in scur:
                                if row[0] is not None and row[0] > 0:
                                    vals.append(int(row[0]))
                    except Exception:
                        vals = []

                    if len(vals) >= 2:
                        parameters[1].enabled = True
                        parameters[1].filter.type = "ValueList"
                        parameters[1].filter.list = sorted(vals)
                        if (parameters[1].value is not None and
                                int(parameters[1].value) not in vals):
                            parameters[1].value = None
                    elif len(vals) == 1:
                        parameters[1].enabled = True
                        parameters[1].filter.type = "ValueList"
                        parameters[1].filter.list = vals
                    else:
                        parameters[1].enabled = True
                        parameters[1].filter.type = "Range"
                        parameters[1].filter.list = [1, 9999]
                else:
                    parameters[1].enabled = False
                    parameters[1].value = None
                del r
            except Exception:
                parameters[1].enabled = True
                parameters[1].filter.type = "Range"
                parameters[1].filter.list = [1, 9999]
        elif not parameters[0].value:
            parameters[1].enabled = False
            parameters[1].value = None

        # Custom cell size toggle
        if parameters[6].altered:
            if str(parameters[6].value) == "Custom":
                parameters[7].enabled = True
            else:
                parameters[7].enabled = False
                parameters[7].value = None

        # Majority skip toggle
        if parameters[12].altered:
            if str(parameters[12].value) == "Skip":
                parameters[13].enabled = False
                parameters[13].value = 1
            else:
                parameters[13].enabled = True

        # Crown separation method toggle
        if parameters[15].altered:
            method = str(parameters[15].value)
            if method == "Watershed":
                parameters[16].enabled = True   # ws_min_area
                parameters[17].enabled = True   # ws_search_radius
                parameters[18].enabled = False  # grid_size
            elif method == "Grid":
                parameters[16].enabled = False
                parameters[17].enabled = False
                parameters[18].enabled = True
            else:  # None
                parameters[16].enabled = False
                parameters[17].enabled = False
                parameters[18].enabled = False
        return

    # ----------------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------------
    def updateMessages(self, parameters):
        # Single band check
        if parameters[0].altered and parameters[0].value:
            try:
                r = arcpy.Raster(str(parameters[0].value))
                if r.bandCount != 1:
                    parameters[0].setErrorMessage(
                        f"Expected single-band raster, "
                        f"got {r.bandCount} bands.")
                elif not r.isInteger:
                    parameters[0].setErrorMessage(
                        "Raster must have integer pixel type "
                        "(class values).")
                del r
            except Exception:
                pass

        # Tree class required
        if (parameters[0].value and
                parameters[1].enabled and
                parameters[1].value is None):
            parameters[1].setErrorMessage(
                "Select the class value that represents trees.")

        # Area check
        if (parameters[8].value is not None and
                parameters[9].value is not None):
            if float(parameters[8].value) >= float(parameters[9].value):
                parameters[9].setErrorMessage(
                    "Max area must be greater than min area.")

        # Custom cell size
        if (parameters[6].value and
                str(parameters[6].value) == "Custom"):
            if (not parameters[7].value or
                    float(parameters[7].value) <= 0):
                parameters[7].setErrorMessage(
                    "Enter a positive cell size in meters.")

        # EIGHT + many passes warning
        if (parameters[12].value and
                str(parameters[12].value) == "EIGHT" and
                parameters[13].value is not None and
                int(parameters[13].value) > 1):
            parameters[13].setWarningMessage(
                "EIGHT neighborhood with multiple passes "
                "may erode small tree crowns.")

        # Grid size required if grid method selected
        if (parameters[15].value and
                str(parameters[15].value) == "Grid"):
            if (not parameters[18].value or
                    float(parameters[18].value) <= 0):
                parameters[18].setErrorMessage(
                    "Enter a positive grid cell size in meters.")

        # Watershed params required if watershed selected
        if (parameters[15].value and
                str(parameters[15].value) == "Watershed"):
            if (not parameters[16].value or
                    float(parameters[16].value) <= 0):
                parameters[16].setErrorMessage(
                    "Enter a positive min area for watershed.")
            if (not parameters[17].value or
                    float(parameters[17].value) <= 0):
                parameters[17].setErrorMessage(
                    "Enter a positive search radius in meters.")

        # Overwrite warning
        if parameters[20].altered and parameters[20].value:
            rp = os.path.join(
                str(parameters[20].valueAsText),
                "trees", "tree_results.shp")
            if os.path.exists(rp):
                parameters[20].setWarningMessage(
                    "Existing results will be OVERWRITTEN.")
        return

    # ----------------------------------------------------------
    # LICENSE
    # ----------------------------------------------------------
    def isLicensed(self):
        try:
            return (arcpy.CheckExtension("Spatial") == "Available"
                    and arcpy.CheckExtension("3D") == "Available")
        except Exception:
            return False

    # ----------------------------------------------------------
    # EXECUTE
    # ----------------------------------------------------------
    def execute(self, parameters, messages):
        import _tree_core

        final_path, pts_path = _tree_core.run_tree_pipeline(
            in_cls=str(parameters[0].valueAsText),
            tree_val=int(parameters[1].value),
            in_ndsm=str(parameters[2].valueAsText),
            in_therm=(str(parameters[3].valueAsText)
                      if parameters[3].value else None),
            in_aoi=(str(parameters[4].valueAsText)
                    if parameters[4].value else None),
            out_crs_text=(parameters[5].valueAsText
                          if parameters[5].value else None),
            cs_opt=str(parameters[6].value),
            cs_custom=(float(parameters[7].value)
                       if parameters[7].value else None),
            min_area=float(parameters[8].value),
            max_area=float(parameters[9].value),
            min_ndsm=float(parameters[10].value),
            reject_null_ndsm=bool(parameters[11].value),
            maj_nbr=str(parameters[12].value),
            maj_passes=int(parameters[13].value),
            smooth_tol=float(parameters[14].value),
            crown_method=str(parameters[15].value),
            ws_min_area=(float(parameters[16].value)
                         if parameters[16].value else 120.0),
            ws_search_radius=(float(parameters[17].value)
                              if parameters[17].value else 6.0),
            grid_size=(float(parameters[18].value)
                       if parameters[18].value else 5.0),
            do_centroids=bool(parameters[19].value),
            out_folder=str(parameters[20].valueAsText),
        )

        parameters[21].value = final_path
        if pts_path:
            parameters[22].value = pts_path

    def postExecute(self, parameters):
        return

# ═══════════════════════════════════════════════════════════════════════════════
#  4. ROAD ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class RoadAnalysis(object):

    def __init__(self):
        self.label = "4. Road Analysis"
        self.description = (
            "Road network post-processing pipeline. "
            "Simplifies, dissolves adjacent segments, classifies by "
            "nDSM/length/width, and optionally extracts centerlines. "
            "See the help page (?) for full documentation."
        )
        self.canRunInBackground = False
        self.category = "Road"

    # ----------------------------------------------------------
    # PARAMETERS
    # ----------------------------------------------------------
    def getParameterInfo(self):

        # 0: Classified raster (single-band, integer classes)
        p0 = arcpy.Parameter(
            displayName="Classified Raster",
            name="in_classified",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input"
        )

        # 1: Road class value (populated from raster)
        p1 = arcpy.Parameter(
            displayName="Road Class Value",
            name="road_class",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        p1.enabled = False

        # 2: nDSM
        p2 = arcpy.Parameter(
            displayName="nDSM Raster (m)",
            name="in_ndsm",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input"
        )

        # 3: Thermal
        p3 = arcpy.Parameter(
            displayName="Thermal Raster (\u00b0C)",
            name="in_thermal",
            datatype="GPRasterLayer",
            parameterType="Optional",
            direction="Input"
        )

        # 4: AOI
        p4 = arcpy.Parameter(
            displayName="Area of Interest",
            name="in_aoi",
            datatype=["GPFeatureLayer", "GPRasterLayer",
                       "DEFeatureClass", "DERasterDataset",
                       "DEShapefile"],
            parameterType="Optional",
            direction="Input"
        )

        # 5: Output CRS
        p5 = arcpy.Parameter(
            displayName="Output Coordinate System",
            name="out_crs",
            datatype="GPCoordinateSystem",
            parameterType="Optional",
            direction="Input"
        )

        # 6: Cell size option
        p6 = arcpy.Parameter(
            displayName="Processing Cell Size",
            name="cell_size_option",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p6.filter.type = "ValueList"
        p6.filter.list = [
            "Classified raster cell size",
            "Maximum of inputs (coarsest)",
            "nDSM cell size",
            "Custom"
        ]
        p6.value = "Classified raster cell size"

        # 7: Custom cell size
        p7 = arcpy.Parameter(
            displayName="Custom Cell Size (m)",
            name="cell_size_custom",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )
        p7.enabled = False

        # 8: Min area
        p8 = arcpy.Parameter(
            displayName="Minimum Road Segment Area (m\u00b2)",
            name="min_area",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p8.value = 2.0
        p8.filter.type = "Range"
        p8.filter.list = [0.1, 10000.0]

        # 9: Max nDSM (roads are ground-level)
        p9 = arcpy.Parameter(
            displayName="Maximum nDSM Height (m)",
            name="max_ndsm",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p9.value = 0.5
        p9.filter.type = "Range"
        p9.filter.list = [0.0, 10.0]

        # 10: Reject NULL nDSM
        p10 = arcpy.Parameter(
            displayName="Reject Features with NULL nDSM",
            name="reject_null_ndsm",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input"
        )
        p10.value = True

        # 11: Majority neighborhood
        p11 = arcpy.Parameter(
            displayName="Majority Filter Neighborhood",
            name="majority_neighborhood",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p11.filter.type = "ValueList"
        p11.filter.list = ["FOUR", "EIGHT", "Skip"]
        p11.value = "FOUR"

        # 12: Majority passes
        p12 = arcpy.Parameter(
            displayName="Majority Filter Passes",
            name="majority_passes",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        p12.value = 1
        p12.filter.type = "Range"
        p12.filter.list = [1, 5]

        # 13: Simplify tolerance
        p13 = arcpy.Parameter(
            displayName="Simplify Tolerance (m)",
            name="simplify_tol",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p13.value = 0.15
        p13.filter.type = "Range"
        p13.filter.list = [0.01, 5.0]

        # 14: Min MBR length
        p14 = arcpy.Parameter(
            displayName="Minimum Road Length (m)",
            name="min_length",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p14.value = 3.0
        p14.filter.type = "Range"
        p14.filter.list = [0.1, 500.0]

        # 15: Min MBR width
        p15 = arcpy.Parameter(
            displayName="Minimum Road Width (m)",
            name="min_width",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p15.value = 1.5
        p15.filter.type = "Range"
        p15.filter.list = [0.1, 50.0]

        # 16: Generate centerlines
        p16 = arcpy.Parameter(
            displayName="Generate Centerline Polylines",
            name="do_centerlines",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input"
        )
        p16.value = True

        # 17: Min centerline length
        p17 = arcpy.Parameter(
            displayName="Minimum Centerline Length (m)",
            name="min_cl_length",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="Centerline Parameters"
        )
        p17.value = 5.0
        p17.filter.type = "Range"
        p17.filter.list = [0.1, 500.0]

        # 18: Max hole area (Voronoi: fill small holes)
        p18 = arcpy.Parameter(
            displayName="Max Hole Area to Fill (m²)",
            name="max_hole_area",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="Centerline Parameters"
        )
        p18.value = 20.0
        p18.filter.type = "Range"
        p18.filter.list = [0.0, 500.0]

        # 19: Prune factor (Voronoi: branch pruning)
        p19 = arcpy.Parameter(
            displayName="Prune Factor (× median road width)",
            name="prune_factor",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="Centerline Parameters"
        )
        p19.value = 2.0
        p19.filter.type = "Range"
        p19.filter.list = [0.5, 10.0]

        # 20: Densify factor (Voronoi: boundary sampling)
        p20 = arcpy.Parameter(
            displayName="Densify Factor (× cell size)",
            name="densify_factor",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="Centerline Parameters"
        )
        p20.value = 5.0
        p20.filter.type = "Range"
        p20.filter.list = [1.0, 20.0]

        # 21: Generate centroids
        p21 = arcpy.Parameter(
            displayName="Generate Centroid Points",
            name="do_centroids",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input"
        )
        p21.value = True

        # 22: Output folder
        p22 = arcpy.Parameter(
            displayName="Output Folder",
            name="out_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )

        # 23-25: Derived outputs
        p23 = arcpy.Parameter(
            displayName="Output Road Results",
            name="out_results",
            datatype="DEFeatureClass",
            parameterType="Derived",
            direction="Output"
        )
        p24 = arcpy.Parameter(
            displayName="Output Road Centerlines",
            name="out_centerlines",
            datatype="DEFeatureClass",
            parameterType="Derived",
            direction="Output"
        )
        p25 = arcpy.Parameter(
            displayName="Output Road Points",
            name="out_pts",
            datatype="DEFeatureClass",
            parameterType="Derived",
            direction="Output"
        )

        return [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                p10, p11, p12, p13, p14, p15, p16, p17,
                p18, p19, p20, p21, p22, p23, p24, p25]

    # ----------------------------------------------------------
    # DYNAMIC UPDATES
    # ----------------------------------------------------------
    def updateParameters(self, parameters):
        # Populate road class dropdown from raster unique values
        if parameters[0].altered and parameters[0].value:
            try:
                r = arcpy.Raster(str(parameters[0].value))
                if r.isInteger and r.bandCount == 1:
                    vals = []
                    ras_path = str(parameters[0].valueAsText)
                    try:
                        with arcpy.da.SearchCursor(
                            ras_path, ["Value"]
                        ) as scur:
                            for row in scur:
                                if row[0] is not None and row[0] > 0:
                                    vals.append(int(row[0]))
                    except Exception:
                        vals = []

                    if len(vals) >= 2:
                        parameters[1].enabled = True
                        parameters[1].filter.type = "ValueList"
                        parameters[1].filter.list = sorted(vals)
                        if (parameters[1].value is not None and
                                int(parameters[1].value) not in vals):
                            parameters[1].value = None
                    elif len(vals) == 1:
                        parameters[1].enabled = True
                        parameters[1].filter.type = "ValueList"
                        parameters[1].filter.list = vals
                    else:
                        parameters[1].enabled = True
                        parameters[1].filter.type = "Range"
                        parameters[1].filter.list = [1, 9999]
                else:
                    parameters[1].enabled = False
                    parameters[1].value = None
                del r
            except Exception:
                parameters[1].enabled = True
                parameters[1].filter.type = "Range"
                parameters[1].filter.list = [1, 9999]
        elif not parameters[0].value:
            parameters[1].enabled = False
            parameters[1].value = None

        # Custom cell size toggle
        if parameters[6].altered:
            if str(parameters[6].value) == "Custom":
                parameters[7].enabled = True
            else:
                parameters[7].enabled = False
                parameters[7].value = None

        # Majority skip toggle
        if parameters[11].altered:
            if str(parameters[11].value) == "Skip":
                parameters[12].enabled = False
                parameters[12].value = 1
            else:
                parameters[12].enabled = True

        # Centerline parameters toggle
        if parameters[16].altered:
            cl_on = bool(parameters[16].value)
            parameters[17].enabled = cl_on      # min_cl_length
            parameters[18].enabled = cl_on      # max_hole_area
            parameters[19].enabled = cl_on      # prune_factor
            parameters[20].enabled = cl_on      # densify_factor
        return

    # ----------------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------------
    def updateMessages(self, parameters):
        # Single band check
        if parameters[0].altered and parameters[0].value:
            try:
                r = arcpy.Raster(str(parameters[0].value))
                if r.bandCount != 1:
                    parameters[0].setErrorMessage(
                        f"Expected single-band raster, "
                        f"got {r.bandCount} bands.")
                elif not r.isInteger:
                    parameters[0].setErrorMessage(
                        "Raster must have integer pixel type "
                        "(class values).")
                del r
            except Exception:
                pass

        # Road class required
        if (parameters[0].value and
                parameters[1].enabled and
                parameters[1].value is None):
            parameters[1].setErrorMessage(
                "Select the class value that represents roads.")

        # Custom cell size
        if (parameters[6].value and
                str(parameters[6].value) == "Custom"):
            if (not parameters[7].value or
                    float(parameters[7].value) <= 0):
                parameters[7].setErrorMessage(
                    "Enter a positive cell size in meters.")

        # Length > Width check (logical)
        if (parameters[14].value is not None and
                parameters[15].value is not None):
            if float(parameters[14].value) < float(parameters[15].value):
                parameters[14].setWarningMessage(
                    "Min length is less than min width — verify "
                    "this is intentional.")

        # EIGHT + many passes warning
        if (parameters[11].value and
                str(parameters[11].value) == "EIGHT" and
                parameters[12].value is not None and
                int(parameters[12].value) > 1):
            parameters[12].setWarningMessage(
                "EIGHT neighborhood with multiple passes "
                "may erode narrow road segments.")

        # Overwrite warning
        if parameters[22].altered and parameters[22].value:
            rp = os.path.join(
                str(parameters[22].valueAsText),
                "roads", "road_results.shp")
            if os.path.exists(rp):
                parameters[22].setWarningMessage(
                    "Existing results will be OVERWRITTEN.")
        return

    # ----------------------------------------------------------
    # LICENSE
    # ----------------------------------------------------------
    def isLicensed(self):
        try:
            return (arcpy.CheckExtension("Spatial") == "Available"
                    and arcpy.CheckExtension("3D") == "Available")
        except Exception:
            return False

    # ----------------------------------------------------------
    # EXECUTE
    # ----------------------------------------------------------
    def execute(self, parameters, messages):
        import _road_core

        final_path, cl_path, pts_path = _road_core.run_road_pipeline(
            in_cls=str(parameters[0].valueAsText),
            road_val=int(parameters[1].value),
            in_ndsm=str(parameters[2].valueAsText),
            in_therm=(str(parameters[3].valueAsText)
                      if parameters[3].value else None),
            in_aoi=(str(parameters[4].valueAsText)
                    if parameters[4].value else None),
            out_crs_text=(parameters[5].valueAsText
                          if parameters[5].value else None),
            cs_opt=str(parameters[6].value),
            cs_custom=(float(parameters[7].value)
                       if parameters[7].value else None),
            min_area=float(parameters[8].value),
            max_ndsm=float(parameters[9].value),
            reject_null_ndsm=bool(parameters[10].value),
            maj_nbr=str(parameters[11].value),
            maj_passes=int(parameters[12].value),
            simp_tol=float(parameters[13].value),
            min_length=float(parameters[14].value),
            min_width=float(parameters[15].value),
            do_centerlines=bool(parameters[16].value),
            # BUG 1 FIX: safe defaults when centerline params are
            # disabled (None) after user toggled do_centerlines off
            min_cl_length=(float(parameters[17].value)
                           if parameters[17].value else 5.0),
            max_hole_area=(float(parameters[18].value)
                           if parameters[18].value else 20.0),
            prune_factor=(float(parameters[19].value)
                          if parameters[19].value else 2.0),
            densify_factor=(float(parameters[20].value)
                            if parameters[20].value else 5.0),
            do_centroids=bool(parameters[21].value),
            out_folder=str(parameters[22].valueAsText),
        )

        parameters[23].value = final_path
        if cl_path:
            parameters[24].value = cl_path
        if pts_path:
            parameters[25].value = pts_path

    def postExecute(self, parameters):
        return
