import unittest

from backend.exposure import (
    ExposureInventory,
    GridBounds,
    count_exposure,
    parse_osm_map_xml,
)


class ExposureTests(unittest.TestCase):
    def test_parses_osm_building_centers_and_road_geometry(self) -> None:
        inventory = parse_osm_map_xml(
            b"""<osm>
              <node id="1" lon="1.0" lat="2.0"/>
              <node id="2" lon="2.0" lat="2.0"/>
              <node id="3" lon="2.0" lat="3.0"/>
              <node id="4" lon="-1.0" lat="2.5"/>
              <node id="5" lon="5.0" lat="2.5"/>
              <way id="10">
                <nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="1"/>
                <tag k="building" v="yes"/>
              </way>
              <way id="11">
                <nd ref="4"/><nd ref="5"/>
                <tag k="highway" v="residential"/>
              </way>
            </osm>"""
        )

        self.assertEqual(len(inventory.buildings), 1)
        self.assertAlmostEqual(inventory.buildings[0][0], 5.0 / 3.0)
        self.assertAlmostEqual(inventory.buildings[0][1], 7.0 / 3.0)
        self.assertEqual(len(inventory.roads), 1)
        self.assertEqual(inventory.roads[0], ((-1.0, 2.5), (5.0, 2.5)))

    def test_counts_only_features_intersecting_wet_cells(self) -> None:
        bounds = GridBounds(west=0.0, south=0.0, east=4.0, north=4.0)
        inventory = ExposureInventory(
            buildings=((0.5, 3.5), (3.5, 0.5)),
            roads=(
                ((-1.0, 2.5), (5.0, 2.5)),
                ((-1.0, -1.0), (-0.5, -0.5)),
            ),
        )
        depths = [0.0] * 16
        depths[0] = 0.02
        depths[1 * 4 + 2] = 0.03

        counts = count_exposure(depths, 4, bounds, inventory)

        self.assertEqual(counts.buildings, 1)
        self.assertEqual(counts.road_ways, 1)
        self.assertEqual(counts.buildings_by_cell[0], 1)
        self.assertGreater(counts.road_ways_by_cell[1 * 4 + 2], 0)

    def test_reports_zero_exposure_for_dry_cells(self) -> None:
        inventory = ExposureInventory(
            buildings=((0.5, 0.5),),
            roads=(((0.0, 0.5), (1.0, 0.5)),),
        )
        counts = count_exposure(
            [0.0] * 4,
            2,
            GridBounds(west=0.0, south=0.0, east=1.0, north=1.0),
            inventory,
        )
        self.assertEqual(counts.buildings, 0)
        self.assertEqual(counts.road_ways, 0)


if __name__ == "__main__":
    unittest.main()
