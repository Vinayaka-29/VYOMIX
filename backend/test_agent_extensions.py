"""
Unit and Integration Tests for SatQuery AI Agent Extensions
SIH 2026 Problem Statement 26167 | Team Vyomix

Tests:
1. Area of Interest (AOI) parsing, normalization, and PIL cropping
2. Georeferenced coordinate transformation (pixel -> geo)
3. Physical measurements (area m², hectares, percentage)
4. Mission / Investigation multi-turn session state
5. NL geospatial operations & constraint parsing
6. End-to-end integration into query interpretation and routing
"""
import unittest
from PIL import Image
import numpy as np

from agent.aoi import parse_aoi, crop_to_aoi, AOISpec
from agent.geo_evidence import bbox_pixel_to_geo, enrich_grounding_with_geo
from agent.geo_measurements import compute_region_area, compute_change_measurements, compute_aoi_percentage
from agent.mission import create_mission, get_mission, list_missions
from agent.nl_geo_ops import parse_geo_constraints, filter_regions_by_constraints
from agent.query_interpreter import interpret_query


class TestAgentExtensions(unittest.TestCase):

    # -------------------------------------------------------------------------
    # 1. Area of Interest (AOI)
    # -------------------------------------------------------------------------
    def test_aoi_parsing_rectangle_normalized(self):
        spec = parse_aoi({"type": "rectangle", "bbox": [0.1, 0.2, 0.8, 0.9], "coord_system": "normalized"})
        self.assertIsNotNone(spec)
        self.assertEqual(spec.aoi_type, "rectangle")
        self.assertEqual(spec.coord_system, "normalized")
        self.assertEqual(len(spec.coordinates), 4)

    def test_aoi_parsing_invalid_rectangle(self):
        spec = parse_aoi({"type": "rectangle", "bbox": [0.1, 0.2]})  # incomplete
        self.assertIsNone(spec)

    def test_aoi_crop(self):
        img = Image.new("RGB", (200, 100), color=(100, 150, 200))
        spec = AOISpec("rectangle", [0.0, 0.0, 0.5, 0.5], coord_system="normalized")
        cropped, crop_info = crop_to_aoi(img, spec)
        self.assertEqual(cropped.size, (100, 50))
        self.assertEqual(crop_info["aoi_normalized_bbox"], [0.0, 0.0, 0.5, 0.5])

    # -------------------------------------------------------------------------
    # 2. Georeferenced Evidence
    # -------------------------------------------------------------------------
    def test_geo_evidence_non_georeferenced(self):
        # Non-georeferenced image returns None safely without error
        pixel_bbox = [10, 20, 50, 80]
        geo_box = bbox_pixel_to_geo(pixel_bbox, "non_existent_image.tif")
        self.assertIsNone(geo_box)

    def test_enrich_grounding_with_geo(self):
        grounding_result = {"found": True, "bbox": [10, 10, 60, 60], "confidence": 0.85}
        enriched = enrich_grounding_with_geo(grounding_result, "non_existent_image.tif")
        self.assertIn("geographic_note", enriched)
        self.assertIsNone(enriched["geographic_bbox"])

    # -------------------------------------------------------------------------
    # 3. Geospatial Measurements
    # -------------------------------------------------------------------------
    def test_compute_region_area(self):
        # 100 pixels, 10m x 10m resolution -> 100 * 100 = 10,000 m² = 1.0 hectare
        measure = compute_region_area(100, 10.0, 10.0, "meters")
        self.assertEqual(measure["area_m2"], 10000.0)
        self.assertEqual(measure["area_hectares"], 1.0)
        self.assertFalse(measure["is_estimate"])

    def test_compute_aoi_percentage(self):
        pct = compute_aoi_percentage(250, 1000)
        self.assertEqual(pct, 25.0)

    def test_compute_change_measurements(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 255  # 400 pixels changed
        meta = {
            "resolution": {"x": 10.0, "y": 10.0, "unit": "meters"},
            "crs": "EPSG:32643",
        }
        res = compute_change_measurements(mask, meta)
        self.assertEqual(res["pixel_count"], 400)
        self.assertEqual(res["total_pixels"], 10000)
        self.assertEqual(res["percentage_of_scene"], 4.0)
        self.assertEqual(res["area_hectares"], 4.0)

    # -------------------------------------------------------------------------
    # 4. Mission / Investigation Mode
    # -------------------------------------------------------------------------
    def test_mission_lifecycle(self):
        m = create_mission("upload_session_123", "Flood Damage Assessment")
        self.assertIsNotNone(m.mission_id)
        
        # Add query result
        m.add_query_result("What changed?", {
            "status": "completed",
            "task": "change_vqa",
            "query_id": "q-1",
            "answer": "12% water increase",
            "visual_artifacts": {"change_overlay_path": "data/change_overlay.png"}
        })
        self.assertEqual(len(m.queries), 1)
        self.assertEqual(len(m.analysis_history), 1)
        self.assertIn("last_visual_artifacts", m.cached_results)

        retrieved = get_mission(m.mission_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.objective, "Flood Damage Assessment")
        self.assertTrue(len(list_missions()) >= 1)

    # -------------------------------------------------------------------------
    # 5. NL Geo Operations Parser
    # -------------------------------------------------------------------------
    def test_nl_geo_ops_parsing(self):
        q = "Count the number of agricultural patches larger than 10 hectares in the northern half"
        c = parse_geo_constraints(q)
        self.assertEqual(c.min_area_ha, 10.0)
        self.assertEqual(c.spatial_selector, "northern_half")
        self.assertTrue(c.count_regions)
        self.assertEqual(c.filter_by_type, "agriculture")

    def test_nl_geo_filter_regions(self):
        regions = [
            {"id": 1, "area_ha": 3.5, "bbox": [0.1, 0.1, 0.3, 0.3]},
            {"id": 2, "area_ha": 15.0, "bbox": [0.1, 0.1, 0.4, 0.4]},
            {"id": 3, "area_ha": 25.0, "bbox": [0.1, 0.6, 0.4, 0.9]},  # southern
        ]
        q = "Find regions larger than 10 hectares in the northern half"
        c = parse_geo_constraints(q)
        filtered = filter_regions_by_constraints(regions, c)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["id"], 2)

    # -------------------------------------------------------------------------
    # 6. Query Interpreter Integration
    # -------------------------------------------------------------------------
    def test_query_interpreter_geo_constraints(self):
        res = interpret_query("Highlight urban areas exceeding 5 ha in the southern half")
        self.assertIsNotNone(res.get("geo_constraints"))
        self.assertEqual(res["geo_constraints"]["min_area_ha"], 5.0)
        self.assertEqual(res["geo_constraints"]["spatial_selector"], "southern_half")
        self.assertIn("spatial:southern", res["signals"])


if __name__ == "__main__":
    unittest.main()
