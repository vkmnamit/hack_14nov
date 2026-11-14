"""OpenStreetMap exposure inventory and grid-based impact counts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os
import xml.etree.ElementTree as ET
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


EXPOSURE_THRESHOLD_M = 0.01
DEFAULT_OSM_MAP_URL = "https://api.openstreetmap.org/api/0.6/map"


@dataclass(frozen=True)
class GridBounds:
    west: float
    south: float
    east: float
    north: float


@dataclass(frozen=True)
class ExposureInventory:
    buildings: tuple[tuple[float, float], ...]
    roads: tuple[tuple[tuple[float, float], ...], ...]


@dataclass(frozen=True)
class ExposureCounts:
    buildings: int
    road_ways: int
    buildings_by_cell: tuple[int, ...]
    road_ways_by_cell: tuple[int, ...]


class ExposureDataError(RuntimeError):
    """Raised when an OSM exposure inventory cannot be loaded or parsed."""


def grid_bounds(
    longitude: float,
    latitude: float,
    size: int,
    cell_size_m: float,
) -> GridBounds:
    half_height_m = size * cell_size_m / 2.0
    lat_delta = half_height_m / 111_320.0
    lon_delta = half_height_m / (
        111_320.0 * max(math.cos(math.radians(latitude)), 1e-6)
    )
    return GridBounds(
        west=longitude - lon_delta,
        south=latitude - lat_delta,
        east=longitude + lon_delta,
        north=latitude + lat_delta,
    )


def parse_osm_map_xml(payload: bytes) -> ExposureInventory:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as error:
        raise ExposureDataError("OpenStreetMap returned invalid XML.") from error

    nodes: dict[str, tuple[float, float]] = {}
    buildings: list[tuple[float, float]] = []
    roads: list[tuple[tuple[float, float], ...]] = []
    for node in root.findall("node"):
        node_id = node.get("id")
        if node_id is None:
            continue
        point = (float(node.attrib["lon"]), float(node.attrib["lat"]))
        nodes[node_id] = point
        if any(tag.get("k") == "building" for tag in node.findall("tag")):
            buildings.append(point)

    for way in root.findall("way"):
        tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
        points = tuple(
            nodes[reference]
            for reference in (
                node.get("ref") for node in way.findall("nd")
            )
            if reference is not None and reference in nodes
        )
        if "building" in tags and points:
            buildings.append(_polygon_centroid(points))
        if "highway" in tags and len(points) >= 2:
            roads.append(points)

    return ExposureInventory(tuple(buildings), tuple(roads))


def _polygon_centroid(
    points: tuple[tuple[float, float], ...],
) -> tuple[float, float]:
    area_twice = 0.0
    weighted_longitude = 0.0
    weighted_latitude = 0.0
    for (longitude, latitude), (next_longitude, next_latitude) in zip(
        points,
        (*points[1:], points[0]),
    ):
        cross = longitude * next_latitude - next_longitude * latitude
        area_twice += cross
        weighted_longitude += (longitude + next_longitude) * cross
        weighted_latitude += (latitude + next_latitude) * cross
    if abs(area_twice) < 1e-12:
        return (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
    return (
        weighted_longitude / (3.0 * area_twice),
        weighted_latitude / (3.0 * area_twice),
    )


@lru_cache(maxsize=2)
def load_osm_inventory(
    south: float,
    west: float,
    north: float,
    east: float,
) -> ExposureInventory:
    endpoint = os.environ.get("VARUNA_OSM_MAP_URL", DEFAULT_OSM_MAP_URL)
    query = urlencode({"bbox": f"{west},{south},{east},{north}"})
    request = Request(
        f"{endpoint}?{query}",
        headers={
            "User-Agent": "VARUNA flood exposure prototype (OpenStreetMap data consumer)",
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise ExposureDataError(f"OpenStreetMap inventory unavailable: {error}") from error
    return parse_osm_map_xml(payload)


def _cell_index(
    longitude: float,
    latitude: float,
    bounds: GridBounds,
    size: int,
) -> tuple[int, int] | None:
    if not (bounds.west <= longitude <= bounds.east and bounds.south <= latitude <= bounds.north):
        return None
    column = min(size - 1, int((longitude - bounds.west) / (bounds.east - bounds.west) * size))
    row = min(size - 1, int((bounds.north - latitude) / (bounds.north - bounds.south) * size))
    return row, column


def _clip_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    bounds: GridBounds,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    lower = 0.0
    upper = 1.0
    for direction, distance in (
        (-delta_x, start[0] - bounds.west),
        (delta_x, bounds.east - start[0]),
        (-delta_y, start[1] - bounds.south),
        (delta_y, bounds.north - start[1]),
    ):
        if direction == 0:
            if distance < 0:
                return None
            continue
        fraction = distance / direction
        if direction < 0:
            lower = max(lower, fraction)
        else:
            upper = min(upper, fraction)
        if lower > upper:
            return None
    return (
        (start[0] + lower * delta_x, start[1] + lower * delta_y),
        (start[0] + upper * delta_x, start[1] + upper * delta_y),
    )


def count_exposure(
    depths: list[float],
    size: int,
    bounds: GridBounds,
    inventory: ExposureInventory,
    threshold_m: float = EXPOSURE_THRESHOLD_M,
) -> ExposureCounts:
    if len(depths) != size * size:
        raise ValueError("depth array length does not match the grid size")
    wet = [depth >= threshold_m for depth in depths]

    buildings_by_cell = [0] * (size * size)
    for longitude, latitude in inventory.buildings:
        cell = _cell_index(longitude, latitude, bounds, size)
        if cell is not None:
            buildings_by_cell[cell[0] * size + cell[1]] += 1

    affected_roads = 0
    road_ways_by_cell = [0] * (size * size)
    for way in inventory.roads:
        intersects_wet_cell = False
        touched_cells: set[int] = set()
        for start, end in zip(way, way[1:]):
            clipped = _clip_segment(start, end, bounds)
            if clipped is None:
                continue
            start, end = clipped
            samples = max(
                math.ceil(
                    abs(end[0] - start[0])
                    / (bounds.east - bounds.west)
                    * size
                ),
                math.ceil(
                    abs(end[1] - start[1])
                    / (bounds.north - bounds.south)
                    * size
                ),
                1,
            )
            for step in range(samples + 1):
                fraction = step / samples
                longitude = start[0] + (end[0] - start[0]) * fraction
                latitude = start[1] + (end[1] - start[1]) * fraction
                cell = _cell_index(longitude, latitude, bounds, size)
                if cell is not None:
                    cell_index = cell[0] * size + cell[1]
                    touched_cells.add(cell_index)
                    if wet[cell_index]:
                        intersects_wet_cell = True
        for cell_index in touched_cells:
            road_ways_by_cell[cell_index] += 1
        if intersects_wet_cell:
            affected_roads += 1

    affected_buildings = sum(
        count for index, count in enumerate(buildings_by_cell) if wet[index]
    )
    return ExposureCounts(
        affected_buildings,
        affected_roads,
        tuple(buildings_by_cell),
        tuple(road_ways_by_cell),
    )
