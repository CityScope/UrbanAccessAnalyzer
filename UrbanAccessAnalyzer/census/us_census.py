import copy
import sys
import inspect
from typing import Union, List, Optional, Dict, Any, Iterable

import pandas as pd
import geopandas as gpd
import shapely

import pygris
import pygris.utils
from pygris.data import get_census, get_lodes
import us  # For state normalization
import gzip
import re
from pyproj import Geod
from .. import geometry_utils 
from .. import configs
from .. import api_keys

import os 
import platformdirs

CENSUS_LATEST_YEARS: Dict[str, int] = {
    "dec/dhc": 2020,  # Available for blocks
    "acs/acs5": 2023,  # Only block groups and higher
    "acs/acs1": 2024,  # Only places and higher
    "lodes/LODES7/rac": 2021,  # LODES7 data available through 2021
    "lodes/LODES7/wac": 2021,  # LODES7 data available through 2021
    "lodes/LODES8/rac": 2023,  # LODES8 data available through 2023 (may be incomplete for recent years)
    "lodes/LODES8/wac": 2023,  # LODES8 data available through 2023 (may be incomplete for recent years)
}

CENSUS_VALID_LEVELS: Dict[str, list] = {
    "dec/dhc": [
        "state",
        "county",
        "tract",
        "blockgroup",
        "block",
    ],  # Available for blocks
    "acs/acs5": [
        "state",
        "county",
        "tract",
        "blockgroup",
        "block",
    ],  # Only block groups and higher
    "acs/acs1": [
        "state",
        "county",
        "tract",
        "blockgroup",
        "block",
    ],  # Only places and higher
    "lodes/LODES7/rac": [
        "state",
        "county",
        "tract",
        "blockgroup",
        "block",
    ],  # LODES7 data available through 2021
    "lodes/LODES7/wac": [
        "state",
        "county",
        "tract",
        "blockgroup",
        "block",
    ],  # LODES7 data available through 2021
    "lodes/LODES8/rac": [
        "state",
        "county",
        "tract",
        "blockgroup",
        "block",
    ],  # LODES8 data available through 2023 (may be incomplete for recent years)
    "lodes/LODES8/wac": [
        "state",
        "county",
        "tract",
        "blockgroup",
        "block",
    ],  # LODES8 data available through 2023 (may be incomplete for recent years)
}

CENSUS_RESAMPLE = {
    # Clean hierarchy
    "blockgroup": "blockgroup",
    "tract": "blockgroup",
    "county": "tract",
    "state": "county",
    "nation": "state",

    # County-built statistical areas
    "core_based_statistical_area": "county",
    "combined_statistical_area": "county",
    "metro_division": "county",
    "division": "state",
    "region": "state",
    "new_england": "state",

    # Irregular boundaries (best possible without blocks)
    "place": "blockgroup",
    "urban_area": "blockgroup",
    "puma": "blockgroup",
    "congressional_district": "blockgroup",
    "state_legislative_district": "blockgroup",
    "voting_district": "blockgroup",
    "school_district": "blockgroup",
    "native_area": "blockgroup",
    "tribal_block_group": "blockgroup",
    "tribal_tract": "tract",
    "tribal_subdivisions_national": "blockgroup",
    "county_subdivision": "blockgroup",
    "alaska_native_regional_corporation": "county",
}

# Fetch census data
GEO_HIERARCHIES = {
    "block": ["state", "county", "tract", "block"],
    "blockgroup": ["state", "county", "tract", "block group"],
    "tract": ["state", "county", "tract"],
    "county": ["state", "county"],
    "state": ["state"],
}

# Prepare geography functions
GEOMETRY_FUNCS = {
    # Census geographies
    "block": pygris.blocks,
    "blockgroup": pygris.block_groups,
    "block group": pygris.block_groups,
    "block_group": pygris.block_groups,
    "tract": pygris.tracts,
    "place": pygris.places,
    "county": pygris.counties,
    "counties": pygris.counties,
    "state": pygris.states,
    "nation": pygris.nation,
    # Statistical areas
    "division": pygris.divisions,
    "region": pygris.regions,
    "core_based_statistical_area": pygris.core_based_statistical_areas,
    "combined_statistical_area": pygris.combined_statistical_areas,
    "metro_division": pygris.metro_divisions,
    "new_england": pygris.new_england,
    "puma": pygris.pumas,
    "urban_area": pygris.urban_areas,
    # Political / voting districts
    "congressional_district": pygris.congressional_districts,
    "state_legislative_district": pygris.state_legislative_districts,
    "voting_district": pygris.voting_districts,
    # School / educational
    "school_district": pygris.school_districts,
    # Tribal / native
    "native_area": pygris.native_areas,
    "alaska_native_regional_corporation": pygris.alaska_native_regional_corporations,
    "tribal_block_group": pygris.tribal_block_groups,
    "tribal_tract": pygris.tribal_tracts,
    "tribal_subdivisions_national": pygris.tribal_subdivisions_national,
    # Additional administrative
    "county_subdivision": pygris.county_subdivisions,
}


CENSUS_FIELDS_CATEGORIES = {
    # ============================================================
    # TOTAL POPULATION (RESIDENCE)
    # ============================================================
    "population": {
        "source": "decennial_dhc",
        "years": 2020,
        "fields": {
            "total": ["P1_001N"],
            "housingUnits": ["H1_001N"],
        },
        "fields_universe": {
            "default": "total",
            "total": "DENSITY_ONLY",
        },
        "agg_weights": {},
    },
    "households": {
        "source": "acs5",
        "years": 2023,
        "fields": {
            "total": ["B25044_001"],
            "owners": ["B25003_002"],
            "renters": ["B25003_003"],
            "meanSize": ["B25010_001"],
            "vacant": ["B25002_003"],
        },
        "fields_universe": {
            "default": "total",
            "total": "DENSITY_ONLY",
            "meanSize": "NO_DENSITY_OR_RATIO",
        },
        "agg_weights": {
            "meanSize": "total",
        },
    },
    # ============================================================
    # GENDER (RESIDENCE)
    # ============================================================
    "gender": {
        "source": "decennial_dhc",
        "years": 2020,
        "fields": {
            "total": ["P1_001N"],
            "male": [
                "P12_002N",  # Male total (or sum of all male age bins)
            ],
            "female": [
                "P12_026N"  # Female total (or sum of all female age bins)
            ],
            # "other": [],  # Not available in 2020 DHC
        },
        "fields_universe": {"default": "total"},
        "agg_weights": {},
    },
    # ============================================================
    # AGE (RESIDENCE)
    # ============================================================
    "age": {
        "source": "decennial_dhc",
        "years": 2020,
        "fields": {
            "total": ["P12_001N"],
            "Under18": [
                "P12_003N",
                "P12_004N",
                "P12_005N",
                "P12_006N",  # Male: <5, 5-9, 10-14, 15-17
                "P12_027N",
                "P12_028N",
                "P12_029N",
                "P12_030N",  # Female: <5, 5-9, 10-14, 15-17
            ],
            "18to64": [
                "P12_007N",
                "P12_008N",
                "P12_009N",
                "P12_010N",
                "P12_011N",
                "P12_012N",
                "P12_013N",
                "P12_014N",
                "P12_015N",
                "P12_016N",
                "P12_017N",
                "P12_018N",
                "P12_019N",  # Male: 18 through 64
                "P12_031N",
                "P12_032N",
                "P12_033N",
                "P12_034N",
                "P12_035N",
                "P12_036N",
                "P12_037N",
                "P12_038N",
                "P12_039N",
                "P12_040N",
                "P12_041N",
                "P12_042N",
                "P12_043N",  # Female: 18 through 64
            ],
            "Over65": [
                "P12_020N",
                "P12_021N",
                "P12_022N",
                "P12_023N",
                "P12_024N",
                "P12_025N",  # Male: 65+
                "P12_044N",
                "P12_045N",
                "P12_046N",
                "P12_047N",
                "P12_048N",
                "P12_049N",  # Female: 65+
            ],
        },
        "fields_universe": {"default": "total"},
        "agg_weights": {},
    },
    # ============================================================
    # RACE / ETHNICITY (RESIDENCE)
    # ============================================================
    "race": {
        "source": "decennial_dhc",
        "years": 2020,
        "fields": {
            "total": ["P3_001N"],
            "white": ["P3_002N"],
            "nonWhite": [
                "P3_003N",
                "P3_004N",
                "P3_005N",
                "P3_006N",
                "P3_007N",
                "P3_008N",
            ],
            "black": ["P3_003N"],
            "native": ["P3_004N"],
            "asian": ["P3_005N"],
            "others": ["P3_006N", "P3_007N", "P3_008N"],
            "hispanic": ["P5_010N"],
            "hispanicOrNonWhite": [
                "P5_004N",
                "P5_005N",
                "P5_006N",
                "P5_007N",
                "P5_008N",
                "P5_009N",
                "P5_010N",
            ],
        },
        "fields_universe": {
            "default": "total",
        },
        "agg_weights": {},
    },
    # ============================================================
    # INCOME / POVERTY (RESIDENCE)
    # ============================================================
    "income": {
        "source": "acs5",
        "years": 2023,
        "fields": {
            "population": ["B01003_001"],
            "populationPoverty": ["C17002_001"],
            "population16+":["B23025_001"],
            "households": ["B25044_001"],
            "medianHousehold": ["B19013_001"],  # median household income
            "meanCapita": ["B19301_001"],  # mean per capita income
            "poverty050": ["C17002_002"],  # <50% of poverty line
            "poverty100": ["C17002_002", "C17002_003"],  # <100%
            "poverty150": ["C17002_002", "C17002_003", "C17002_004", "C17002_005"],
            "poverty200": [
                "C17002_002",
                "C17002_003",
                "C17002_004",
                "C17002_005",
                "C17002_006",
                "C17002_007",
            ],
            # Unemployment
            "unemployedCount": [
                "B23025_005"
            ],  # unemployed persons 16+ (residence-based)
            "laborForce": ["B23025_003"],  # total in labor force
        },
        "fields_universe": {
            "default": "population16+",  # population 16+
            "medianHousehold": "NO_DENSITY_OR_RATIO",
            "meanCapita": "NO_DENSITY_OR_RATIO",
            "poverty050": "populationPoverty",  # poverty universe
            "poverty100": "populationPoverty",  # poverty universe
            "poverty150": "populationPoverty",  # poverty universe
            "poverty200": "populationPoverty",  # poverty universe
            "unemployedCount": "laborForce",  # use labor force as universe for % unemployed
            "labor_force": "population16+",  # use population 16+ as universe for % working
        },
        "agg_weights": {
            "medianHousehold": "households",  # households
            "meanCapita": "population",  # population total
        },
    },
    # ============================================================
    # WORKERS BY RESIDENCE (ACS — SURVEY)
    # ============================================================
    "workers_residence_acs": {
        "source": "acs5",
        "years": 2023,
        "fields": {
            "total": ["B08301_001"],
            "transit": ["B08301_010"],
            "bus": ["B08301_011"],
            "rapidTransit": ["B08301_012", "B08301_014"],
            "commuterRail": ["B08301_013"],
            "car": ["B08301_002"],
            "walk": ["B08301_019"],
            "bike": ["B08301_018"],
            "walkBike": ["B08301_018", "B08301_019"],
            "otherModes": ["B08301_015", "B08301_016", "B08301_017", "B08301_020"],
            "meanCommuteTime": ["B08303_001"],
            "meanTransitCommuteTime": ["B08136_001"],
        },
        "fields_universe": {
            "default": "total",
            "meanCommuteTime": "NO_DENSITY_OR_RATIO",
            "meanTransitCommuteTime": "NO_DENSITY_OR_RATIO",
        },
        "agg_weights": {
            "meanCommuteTime": "total",
            "meanTransitCommuteTime": "total",
        },
    },
    # ============================================================
    # WORKERS BY RESIDENCE (LODES RAC — ADMIN)
    # ============================================================
    "workers_residence_lodes": {
        "source": "lodes8_rac",
        "years": 2021,
        "fields": {
            "total": ["C000"],
            "lowIncome": ["CE01"],
            "midIncome": ["CE02"],
            "highIncome": ["CE03"],
            "young": ["CA01"],
            "primeAge": ["CA02"],
            "older": ["CA03"],
        },
        "fields_universe": {
            "default": "total",
        },
        "agg_weights": {},
    },
    # ============================================================
    # JOBS BY WORKPLACE (LODES WAC)
    # ============================================================
    "jobs_workplace_lodes": {
        "source": "lodes8_wac",
        "years": 2021,
        "fields": {
            "total": ["C000"],
            "lowIncome": ["CE01"],
            "midIncome": ["CE02"],
            "highIncome": ["CE03"],
            "young": ["CA01"],
            "primeAge": ["CA02"],
            "older": ["CA03"],
        },
        "fields_universe": {
            "default": "total",
        },
        "agg_weights": {},
    },
    # ============================================================
    # VEHICLE AVAILABILITY (RESIDENCE)
    # ============================================================
    "vehicles": {
        "source": "acs5",
        "years": 2023,
        "fields": {
            "households": ["B25044_001"],
            "total": ["B25046_001"],
            "0inHousehold": ["B25044_003", "B25044_010"],
            "0or1inHousehold": [
                "B25044_003",
                "B25044_004",
                "B25044_010",
                "B25044_011",
            ],
        },
        "fields_universe": {
            "default": "households",
        },
        "agg_weights": {},
    },
}

# ----------------------------
# Utilities
# ----------------------------

def set_pygris_folder(path):
    os.makedirs(path,exist_ok=True)
    # Set up local pygrid cache dir
    platformdirs.user_cache_dir = lambda appname="pygris", **kwargs: os.path.abspath(path)

def pick_geoid_column(cols: Iterable[str]) -> Optional[str]:
    def priority(col: str) -> int:
        if col == "GEOID":
            return 0
        if col == "GEOID20":
            return 1
        if col == "GEOID10":
            return 2
        if re.fullmatch(r"GEOID\d{2}", col):
            return 3
        if re.fullmatch(r"GEOID\d+", col):
            return 4
        if "GEOID" in col:
            return 5
        return 999

    best = min(cols, key=priority, default=None)

    if best is None or priority(best) == 999:
        return None

    return best

def get_census_fields(selection, categories_dict=CENSUS_FIELDS_CATEGORIES):
    """
    Filter CENSUS_FIELDS_CATEGORIES based on user selection.

    Supports arbitrary combinations and ordering of transform suffixes:
    _ratio, _log, _density (e.g. field_density_log_ratio, field_log_density, etc.)
    """
    import copy

    # All supported transform suffixes
    TRANSFORM_SUFFIXES = ("_ratio", "_log", "_density")

    def strip_suffixes(field):
        """
        Remove any number of known transform suffixes from the end of a field name.
        Order and repetition do not matter.
        """
        base = field
        changed = True
        while changed:
            changed = False
            for suffix in TRANSFORM_SUFFIXES:
                if base.endswith(suffix):
                    base = base[:-len(suffix)]
                    changed = True
        return base

    def resolve_fields(cat, fields_spec):
        cat_data = copy.deepcopy(categories_dict[cat])
        all_fields = cat_data.get("fields", {})
        all_universe = cat_data.get("fields_universe", {})

        # normalize fields_spec into a list
        if fields_spec in (None, "all", []):
            selected_fields = all_fields
        else:
            if isinstance(fields_spec, str):
                fields_spec = [fields_spec]

            selected_fields = {}
            for f in fields_spec:
                key = strip_suffixes(f)
                if key in all_fields:
                    selected_fields[key] = all_fields[key]

        cat_data["fields"] = selected_fields

        # always include universe/ratio fields
        cat_data["fields_universe"] = {
            k: all_universe.get(k, "default") for k in selected_fields
        }

        return cat_data

    def parse_key_string(key):
        if key in ("all", None, []):
            return key, "all"

        base = strip_suffixes(key)
        parts = base.split("_", 1)

        if len(parts) == 1:
            return parts[0], "all"

        return parts[0], parts[1]

    # Normalize selection into a dict: {category: field_spec}
    selection_dict = {}

    if selection in ("all", None, []):
        selection_dict = {k: "all" for k in categories_dict}

    elif isinstance(selection, str):
        cat, field = parse_key_string(selection)
        selection_dict[cat] = field

    elif isinstance(selection, list):
        for item in selection:
            cat, field = parse_key_string(item)
            if cat in selection_dict:
                existing = selection_dict[cat]
                if existing == "all":
                    continue
                if isinstance(existing, list):
                    selection_dict[cat] = list(set(existing + [field]))
                else:
                    selection_dict[cat] = list(set([existing, field]))
            else:
                selection_dict[cat] = field

    elif isinstance(selection, dict):
        for cat, field_spec in selection.items():
            if field_spec in ("all", None, []):
                selection_dict[cat] = "all"
            elif isinstance(field_spec, (str, list)):
                selection_dict[cat] = field_spec
            else:
                raise ValueError(f"Invalid field_spec for category {cat}: {field_spec}")

    else:
        raise ValueError(f"Invalid selection: {selection}")

    # Resolve fields for each category
    filtered = {}
    for cat, fields_spec in selection_dict.items():
        if cat not in categories_dict:
            continue
        if isinstance(fields_spec, str) and fields_spec != "all":
            fields_spec = [fields_spec]
        filtered[cat] = resolve_fields(cat, fields_spec)

    return filtered


def _to_list(x: Any) -> List:
    """
    Convert a parameter to a list.

    Parameters
    ----------
    x : any or list
        Parameter that is either a single element or a list.

    Returns
    -------
    list
        Parameter as a list.
    """
    return x if isinstance(x, list) else [x]


def format_filter(filter: Dict[str, Union[str, List[str]]]) -> Dict[str, List[str]]:
    """
    Normalize filter values for consistent matching.

    - 'state' -> USPS abbreviation (case-insensitive, accepts full name or abbrev)
    - other keys (county, place) -> lowercase for case-insensitive matching

    Parameters
    ----------
    filter : dict
        Filter dictionary with keys like 'state', 'county', 'place'.

    Returns
    -------
    dict
        Normalized filter dictionary with lists of strings.
    """
    target = {}

    for key, val in filter.items():
        vals = [val] if isinstance(val, str) else val
        if key.lower() == "state":
            target["state"] = [us.states.lookup(v.strip()).abbr for v in vals]
        else:
            target[key.lower()] = [v.strip().lower() for v in vals]

    return target

def format_categories_dict(
    categories: Dict[str, Dict] = CENSUS_FIELDS_CATEGORIES,
) -> Dict[str, Dict]:
    """
    Format a categories dictionary for census data requests, fully resolving 
    field names, universes, and aggregation weights with standardized sources, 
    years, and column prefixes.
    """
    import copy
    years_as_prefix = True
    categories = copy.deepcopy(categories)

    def source_to_api_dir(source: str) -> str:
        if source.startswith("acs"):
            return "acs/acs1" if source == "acs1" else "acs/acs5"
        elif source.startswith("dec"):
            dec_suffix = source.split("_")[-1] if "_" in source else "dhc"
            return f"dec/{dec_suffix}"
        elif source == "dhc":
            return "dec/dhc"
        elif source == "lodes7_rac":
            return "lodes/LODES7/rac"
        elif source == "lodes7_wac":
            return "lodes/LODES7/wac"
        elif source == "lodes8_rac":
            return "lodes/LODES8/rac"
        elif source == "lodes8_wac":
            return "lodes/LODES8/wac"
        else:
            print(f"Warning: Unrecognized source '{source}'. Returning as is.", file=sys.stderr)
            return source

    def format_years(source: str, years: Optional[Union[int, List[int]]], census_latest_years=CENSUS_LATEST_YEARS) -> List[int]:
        return _to_list(census_latest_years[source]) if years is None else _to_list(years)

    # Standardize source and years
    for cat_name, cat_dict in categories.items():
        cat_dict["source"] = source_to_api_dir(cat_dict["source"])
        cat_dict["years"] = format_years(cat_dict["source"], cat_dict.get("years", None))

    # Determine if year prefixes are needed
    years_by_source = {source: set() for source in [cat["source"] for cat in categories.values()]}
    for cat_name, cat_dict in categories.items():
        years_by_source[cat_dict["source"]].update(cat_dict["years"])
    # years_as_prefix = any(len(y) > 1 for y in years_by_source.values())

    def get_field_name(cat_name: str, field_name: str, year: Optional[int] = None) -> str:
        return f"{year}_{cat_name}_{field_name}" if years_as_prefix and year else f"{cat_name}_{field_name}"

    # Reformat fields, universes, and agg_weights
    for cat_name, cat_dict in categories.items():
        fields_formatted = {}
        fields_universe_formatted = {}
        agg_weights_formatted = {}

        # Create full field names
        field_name_mapping = {}  # map logical field -> full name
        for field_name, field_codes in cat_dict["fields"].items():
            for year in cat_dict["years"]:
                full_name = get_field_name(cat_name, field_name, year=year)
                fields_formatted[full_name] = field_codes
                field_name_mapping[field_name] = full_name  # for universe & agg_weights resolution

        # Resolve universes to actual full column names
        for field_name, universe in cat_dict.get("fields_universe", {}).items():
            if universe in ("NO_DENSITY_OR_RATIO", "DENSITY_ONLY"):
                fields_universe_formatted[field_name_mapping.get(field_name, field_name)] = universe
            else:
                # map logical universe to full field name
                resolved = field_name_mapping.get(universe, universe)  # fallback if universe not in fields
                fields_universe_formatted[field_name_mapping.get(field_name, field_name)] = resolved

        # Resolve agg_weights to actual full column names
        for field_name, weight in cat_dict.get("agg_weights", {}).items():
            resolved_weight = field_name_mapping.get(weight, weight)
            agg_weights_formatted[field_name_mapping.get(field_name, field_name)] = resolved_weight

        cat_dict["fields"] = fields_formatted
        cat_dict["fields_universe"] = fields_universe_formatted
        cat_dict["agg_weights"] = agg_weights_formatted

    return categories


# ----------------------------
# Geospatial Functions
# ----------------------------


def load_shapes(
    level: str = "block",
    state=None,
    county=None,
    year=2024,
    erase_water: bool = False,
    crs=4326,
    cache: bool = True,
    cb: bool = False,
    aoi=None,
    pygris_path=None,
):
    if pygris_path is not None:
        set_pygris_folder(pygris_path)

    # Allow plural names
    for k, v in list(GEOMETRY_FUNCS.items()):
        GEOMETRY_FUNCS[k + "s"] = v

    func = GEOMETRY_FUNCS[level]
    if isinstance(state, str):
        state = [state]
    elif state is None:
        state = [None]

    if isinstance(county, str):
        county = [county]
    elif county is None:
        county = [None]

    df = []
    for s in state:
        for c in county:
            # Build kwargs only with parameters that the function accepts
            sig = inspect.signature(func)
            func_args = {}
            if "state" in sig.parameters and s is not None:
                func_args["state"] = s
            if "county" in sig.parameters and c is not None:
                func_args["county"] = c
            if "year" in sig.parameters:
                func_args["year"] = year
            if "cache" in sig.parameters:
                func_args["cache"] = cache
            if "cb" in sig.parameters:
                func_args["cb"] = cb

            # Call the Pygris function
            shapes = func(**func_args)

            # Reproject if needed
            if crs:
                shapes = shapes.to_crs(crs)

            if erase_water:
                shapes = pygris.utils.erase_water(shapes, year=year, cache=cache)

            df.append(shapes.to_crs(4326))

    df = pd.concat(df).to_crs(4326)
    if aoi is not None:
        aoi = aoi.to_crs(4326).union_all()
        df = df[df.intersects(aoi)]

    return df


# ----------------------------
# Census Tabular Functions
# ----------------------------

# filter: Dict[str, Union[str, List[str]]] = None,
def load_fields(
    state,
    categories: Dict[str, Dict] = None,
    api_key: str = api_keys.US_CENSUS,
    level: str = "blockgroup",
    geo_hierarchies: Dict[str, List[str]] = None,
    add_place_names: bool = False,
    cache:bool=False,
    fields_path=configs.US_CENSUS_FIELDS_PATH
) -> pd.DataFrame:
    """
    Pull tabular census data (Decennial, ACS, LODES) for standard geographic levels.
    
    Notes
    -----
    - No universe columns are computed.
    - No ratios are computed; that is left for a separate function.
    """
    filter = {"state":state}
    # if filter is None:
    #     filter = {}

    if fields_path is not None:
        os.makedirs(fields_path,exist_ok=True)

    if categories is None:
        categories = copy.deepcopy(CENSUS_FIELDS_CATEGORIES)
    if geo_hierarchies is None:
        geo_hierarchies = GEO_HIERARCHIES

    filter = format_filter(filter)
    states = _to_list(filter.get("state", None))
    excluded_territories = ["AS", "GU"]
    state_objs = [
        s_obj
        for s in states
        if (s_obj := us.states.lookup(s)) is not None and s not in excluded_territories
    ]
    if len(state_objs) == 0:
        return None

    states_abbr = [s.abbr for s in state_objs]
    
    # Format categories and fields
    categories = format_categories_dict(categories)
    fields_list = []

    # Prepare fields per category/source/year
    for cat_name, cat_dict in categories.items():
        if cat_dict["source"].startswith("acs") and level == "block":
            continue  # skip ACS block-level

        for field_name, field_codes in cat_dict["fields"].items():
            for year in cat_dict["years"]:
                fields_list.append(
                    {
                        "name": field_name,
                        "source": cat_dict["source"],
                        "year": year,
                        "sum_codes": field_codes,
                    }
                )

    # Add 'E' suffix for ACS fields
    def add_e(c: str) -> str:
        if c is None:
            return c
        return c if c.endswith("E") else c + "E"

    for field in fields_list:
        if field["source"].startswith("acs"):
            field["sum_codes"] = [add_e(c) for c in _to_list(field["sum_codes"])]

    # Organize codes by source/year
    codes_by_source_year: Dict[tuple, set] = {}
    for field in fields_list:
        key = (field["source"], field["year"])
        codes_by_source_year.setdefault(key, set()).update(field["sum_codes"])
        if add_place_names and level.startswith("place"):
            codes_by_source_year[key].add("NAME")

    df_by_source_year: Dict[tuple, pd.DataFrame] = {}
    # Fetch data per source/year
    for (source, year), all_fields in codes_by_source_year.items():
        if len(all_fields) == 0:
            continue
        
        if source not in CENSUS_VALID_LEVELS.keys() or level not in CENSUS_VALID_LEVELS[source]:
            raise Exception(f"Level {level} is not valid for source {source}.")
        
        if cache:
            census_file = os.path.normpath(
                fields_path + "/" + 
                str(states_abbr[0]) + "_" + 
                str(year) + "_" + 
                str(source).replace("/","_") + "_" + 
                str(level) + ".csv"
            )
            if os.path.isfile(census_file):
                available_cols = pd.read_csv(census_file, nrows=0).columns
                cols_to_load = [
                    col for col in (
                        all_fields | {"GEOID", "NAME"}
                    ) if col in available_cols
                ]
                fields_not_in_df = set(
                    col for col in all_fields if not(col in available_cols)
                )
                all_fields = fields_not_in_df
                orig_df = pd.read_csv(census_file, usecols=cols_to_load)
                if len(all_fields) == 0:
                    df_by_source_year[(source, year)] = orig_df
                    continue
        else:
            orig_df = None
        
        if source.startswith("lodes/"):
            parts = source[len("lodes/") :].split("/")  # ["LODES8", "rac"]
            lodes_version = parts[0]
            lodes_type = parts[-1]
            if lodes_type not in ["od", "rac", "wac"]:
                raise Exception(f"Lodes type {lodes_type} not valid. Must be 'od','rac','wac'.")
            if lodes_type == "od":
                raise Exception("LODES 'od' type not implemented for aggregation.")

            if level == "state":
                agg_level = "county"
            elif level == "place":
                agg_level = "block"
            elif level == "blockgroup":
                agg_level = "block group"
            else:
                agg_level = level

            dfs = []
            for s in states_abbr:
                try:
                    df_state = get_lodes(
                        version=lodes_version,
                        state=s,
                        year=year,
                        lodes_type=lodes_type,
                        agg_level=agg_level,
                        cache=True,
                    )
                except gzip.BadGzipFile:
                    print(f"LODES not available for {s}")
                    continue

                geocode_col = "h_geocode" if lodes_type == "rac" else "w_geocode"
                if level == "state":
                    df_state[geocode_col] = df_state[geocode_col].str.slice(stop=2)
                df_state = df_state.groupby(geocode_col).agg("sum").reset_index()
                dfs.append(df_state)

            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                df = df.rename(columns={geocode_col: "GEOID"})
                df_by_source_year[(source, year)] = df

        else:
            # Standard census data
            geo_hierarchy = geo_hierarchies[level]
            geo_for = geo_hierarchy[-1] + ":*"
            geo_in = [
                f"state:{s.fips}" if g == "state" else f"{g}:*"
                for g in geo_hierarchy[:-1]
                for s in state_objs
            ]

            df = pd.DataFrame(
                get_census(
                    dataset=source,
                    year=year,
                    variables=list(all_fields),
                    params={"for": geo_for, "in": geo_in, "key": api_key},
                    return_geoid=True,
                    guess_dtypes=True,
                )
            )
            if cache and orig_df is not None:
                if "GEOID" in orig_df.columns:
                    on = "GEOID"
                elif "NAME" in orig_df.columns:
                    on = "NAME"
                else:
                    raise Exception("No valid merge column")
                
                df = df.merge(orig_df,on=on,how='outer')

            df_by_source_year[(source, year)] = df
            if cache:
                df.to_csv(census_file)

    # Process and sum fields (no ratios)
    df_processed_by_source_year: Dict[tuple, pd.DataFrame] = {}
    final_df_fields: List[str] = []
    final_df_fields_set: set = set()

    def add_fieldname(field_name: str, prepend: bool = False):
        if field_name not in final_df_fields_set:
            if prepend:
                final_df_fields.insert(0, field_name)
            else:
                final_df_fields.append(field_name)
            final_df_fields_set.add(field_name)

    for field in fields_list:
        add_fieldname(field["name"])
        source, year = field["source"], field["year"]
        df_raw = df_by_source_year.get((source, year))
        if df_raw is None or df_raw.empty:
            print(f"Skipping field '{field['name']}' from source '{source}' year {year} because data is missing")
            continue

        df_proc = df_processed_by_source_year.get((source, year), pd.DataFrame())

        # Preserve GEOID/NAME
        for col in ["GEOID", "NAME"]:
            if col in df_raw.columns and col not in df_proc.columns:
                df_proc[col] = df_raw[col]
                add_fieldname(col, prepend=True)

        # Sum columns
        existing_cols = [c for c in field["sum_codes"] if c in df_raw.columns]
        if existing_cols:
            df_proc[field["name"]] = df_raw[existing_cols].sum(axis=1)
        else:
            print(f"WARNING: No columns found for field '{field['name']}' in {source} {year}, skipping")
            continue

        df_processed_by_source_year[(source, year)] = df_proc

    # Merge all sources
    join_on = ["GEOID"] + (["NAME"] if "NAME" in final_df_fields_set else [])
    df_final: Optional[pd.DataFrame] = None
    for df_proc in df_processed_by_source_year.values():
        df_final = df_proc if df_final is None else df_final.merge(df_proc, on=join_on, how="outer")

    existing_fields = [f for f in final_df_fields if f in df_final.columns]
    return df_final[existing_fields]


# ----------------------------
# Join Geospatial and Census
# ----------------------------


def join_census(
    df_geo: gpd.GeoDataFrame,
    df_census: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Join census tabular data with geospatial shapes using GEOID (and optionally NAME).

    Parameters
    ----------
    df_geo : GeoDataFrame
        Geospatial data with geometries.
    df_census : DataFrame
        Census tabular data.

    Returns
    -------
    GeoDataFrame
        Geospatial data joined with census tabular data.
    """
    # Detect GEOID columns
    geoid_col_census = pick_geoid_column(df_census.columns)
    geoid_col_geo = pick_geoid_column(df_geo.columns)

    if geoid_col_census is None:
        raise KeyError("No GEOID column found in df_census.")
    if geoid_col_geo is None:
        raise KeyError("No GEOID column found in df_geo.")

    # Rename to match if necessary
    if geoid_col_census != geoid_col_geo:
        df_geo = df_geo.rename(columns={geoid_col_geo: geoid_col_census})

    join_on = [geoid_col_census]

    # Optionally include NAME if both datasets have it
    if "NAME" in df_geo.columns and "NAME" in df_census.columns:
        join_on.append("NAME")

    # Drop rows with missing join keys
    df_geo = df_geo.dropna(subset=join_on)
    df_census = df_census.dropna(subset=join_on)

    # Ensure types match for merge
    for col in join_on:
        df_geo[col] = df_geo[col].astype(int)
        df_census[col] = df_census[col].astype(int)

    # Merge
    df_joined = df_geo.merge(df_census, on=join_on, how="inner")

    return df_joined


def compute_densities_and_ratios(
    gdf: Union[gpd.GeoDataFrame, pd.DataFrame],
    categories: Optional[Dict[str, Dict]] = None,
    densities: bool = True,
    density_fields: Optional[List[str]] = None,
    ratios: bool = True,
    ratio_fields: Optional[List[str]] = None,
    ratio_universe_fields: Optional[List[str]] = None,
    geoid_col: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Compute densities and ratios for fields in a GeoDataFrame or DataFrame.

    Automatically resolves universe placeholders in categories dict.

    Parameters
    ----------
    gdf : GeoDataFrame or DataFrame
    categories : dict, optional
        Categories dictionary containing fields and universes for ratios/densities.
    densities : bool
    density_fields : list of str, optional
    ratios : bool
    ratio_fields : list of str, optional
    ratio_universe_fields : list of str, optional
    geoid_col : str, optional

    Returns
    -------
    GeoDataFrame
    """
    df = gdf.copy()

    # Detect GEOID column if not provided
    if geoid_col is None:
        geoid_col = pick_geoid_column(df.columns)
        if geoid_col is None:
            raise KeyError("No GEOID column found in gdf.")

    # Format categories if provided
    if categories is not None:
        categories = format_categories_dict(categories)

    # -------------------------
    # Compute ratios
    # -------------------------
    if ratios:
        if ratio_fields is not None and ratio_universe_fields is not None:
            if len(ratio_fields) != len(ratio_universe_fields):
                raise ValueError("ratio_fields and ratio_universe_fields must have the same length.")
            for field, universe in zip(ratio_fields, ratio_universe_fields):
                if field in df.columns and universe in df.columns:
                    df[f"{field}_ratio"] = df[field] / df[universe]
        elif categories is not None:
            for cat_dict in categories.values():
                fields = cat_dict.get("fields", {})
                fields_universe = cat_dict.get("fields_universe", {})

                for field_name in fields.keys():
                    # 1. Determine universe placeholder
                    if field_name in fields_universe:
                        universe_col = fields_universe[field_name]
                    else:
                        universe_col = fields_universe.get("default")

                    # 2. Skip if no universe defined
                    if universe_col is None:
                        continue

                    # 3. Skip control flags
                    if universe_col in ("NO_DENSITY_OR_RATIO", "DENSITY_ONLY"):
                        continue

                    # 4. Compute ratio only if both columns exist
                    if field_name in df.columns and universe_col in df.columns:
                        df[f"{field_name}_ratio"] = df[field_name] / df[universe_col]
    # -------------------------
    # Compute densities
    # -------------------------
    if densities:
        if isinstance(df, gpd.GeoDataFrame):
            # Compute areas in m²
            try:
                df_proj = df.to_crs(df.estimate_utm_crs())
                df['area'] = df_proj.geometry.area
            except Exception:
                # fallback: geodesic area
                df = df.to_crs(4326)
                df['area'] = df.geometry.map(lambda geom: geometry_utils.geodesic_area(geom))

            # Determine fields for density
            fields_for_density = density_fields or []
            if categories is not None and not density_fields:
                for cat_dict in categories.values():
                    fields_for_density.extend(cat_dict["fields"].keys())
            fields_for_density = [f for f in fields_for_density if f in df.columns]

            # Compute densities (per km²)
            for f in fields_for_density:
                # Try converting to numeric
                numeric = pd.to_numeric(df[f], errors="coerce")

                # If at least one non-NaN value exists, treat as numeric
                if numeric.notna().any():
                    df[f"{f}_density"] = numeric / (df["area"] / 1e6)
        else:
            print("Input is not a GeoDataFrame. Skipping density computation.")

    return df


def resample(
    census_gdf,
    geometries,
    categories=None, 
    columns=None,
    weights=None
):
    if columns is None:
        columns = []
    if weights is None:
        weights = []

    if len(weights) == 0 and len(columns) != 0:
        weights = [None for i in range(len(columns))]

    census_gdf = census_gdf.copy()
    geometries = geometries.copy()
    if categories is not None:
        categories = format_categories_dict(categories)

    geometries['_idx'] = geometries.index
    census_gdf = geometry_utils.source_ids_to_dst_geometry(
        geometries,
        census_gdf,
        contain='centroid',
        id_column="_idx"
    )
    census_gdf["_idx"] = census_gdf["_idx"].str[0]
    # -------------------------
    # Apply weighted aggregation
    # -------------------------
    if categories is not None:
        for cat_name, cat_dict in categories.items():
            fields = cat_dict.get("fields", {})
            agg_weights = cat_dict.get("agg_weights", {})

            columns.extend(list(fields.keys()))

            for field_name in fields.keys():
                agg_weight_key = agg_weights.get(field_name)

                if agg_weight_key:
                    weights.append(agg_weight_key)
                else:
                    weights.append(None)    

    _columns = []
    _weights = []
    for i in range(len(weights)):
        col = columns[i]
        w_col = weights[i]
        if col in census_gdf.columns:
            numeric = pd.to_numeric(census_gdf[col], errors="coerce")
            # If at least one non-NaN value exists, treat as numeric
            if numeric.notna().any():
                census_gdf[col] = numeric
                _columns.append(col)

            if w_col in census_gdf.columns:
                numeric = pd.to_numeric(census_gdf[w_col], errors="coerce")
                # If at least one non-NaN value exists, treat as numeric
                if numeric.notna().any():
                    census_gdf[w_col] = numeric
                    _weights.append(w_col)
                else:
                    _weights.append(None)
            else:
                _weights.append(None)

    columns = [col if col in census_gdf.columns else None for col in columns]
    weights = [col if col in census_gdf.columns else None for col in weights]
    if len(columns) != len(weights):
        raise Exception(f"Length mismatch. Length of columns is {len(columns)}. Length of weights is {len(weights)}")
    
    for i in range(len(weights)):
        if weights[i] is None:
            continue 

        census_gdf[columns[i]] *= census_gdf[weights[i]]

    census_df = census_gdf[["_idx",*columns]].groupby("_idx").agg('sum').reset_index()
    for i in range(len(weights)):
        if weights[i] is not None:
            census_gdf[columns[i]] /= census_gdf[weights[i]]
        
    census_df = census_df[[col for col in census_df.columns if col not in geometries.columns]]
    geometries_with_census = geometries.merge(census_df,on="_idx",how="right")
    geometries_with_census = geometries_with_census.drop(columns=["_idx"])
    return geometries_with_census
        