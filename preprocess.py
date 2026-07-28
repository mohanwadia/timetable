"""
Condenses stop_times.txt + trips.txt + routes.txt into a pattern-based JSON file.
Instead of storing every departure instance, stores only the unique combinations of
(stop, route, direction, service) with their departure times.

This reduces file size by 90%+ since departures don't repeat for every single day.

Usage:
    pip install pandas
    python3 preprocess.py

Expects stop_times.txt, trips.txt, routes.txt in the same folder as this
script. Writes condensed_stop_times.json in the same folder.

Re-run this whenever you download an updated GTFS feed.
"""

import json
import pandas as pd

STOP_TIMES_PATH = "../gtfs/4/google_transit/stop_times.txt"
TRIPS_PATH = "../gtfs/4/google_transit/trips.txt"
ROUTES_PATH = "../gtfs/4/google_transit/routes.txt"
OUTPUT_PATH = "condensed_stop_times.json"

CHUNK_SIZE = 1_000_000  # rows of stop_times.txt processed per chunk


def load_reference_tables():
    print("Loading trips.txt...")
    trips = pd.read_csv(
        TRIPS_PATH,
        dtype=str,
        usecols=["trip_id", "route_id", "service_id", "direction_id", "trip_headsign"],
    )

    print("Loading routes.txt...")
    routes = pd.read_csv(
        ROUTES_PATH,
        dtype=str,
        usecols=["route_id", "route_short_name", "route_color"],
    )

    trips = trips.merge(routes, on="route_id", how="left")

    # Precompute the "direction" label once here
    headsign = trips["trip_headsign"].fillna("").str.strip()
    fallback = "Direction " + trips["direction_id"].fillna("")
    trips["direction"] = headsign.where(headsign != "", fallback)

    trips["route_short_name"] = trips["route_short_name"].fillna(trips["route_id"])
    trips["route_color"] = trips["route_color"].fillna("075AAA")

    trips = trips.set_index("trip_id")[
        ["route_id", "route_short_name", "route_color", "direction", "service_id"]
    ]
    return trips


def get_day_pattern(service_row):
    """
    Classify a service by which days it operates.
    Returns 'weekday', 'saturday', 'sunday', or a custom pattern.
    """
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    active_days = [days[i] for i in range(7) if service_row[days[i]] == '1']
    
    # Check for common patterns
    weekday_days = {'monday', 'tuesday', 'wednesday', 'thursday', 'friday'}
    active_set = set(active_days)
    
    if active_set == weekday_days:
        return 'weekday'
    elif active_set == {'saturday'}:
        return 'saturday'
    elif active_set == {'sunday'}:
        return 'sunday'
    elif active_set == weekday_days | {'saturday', 'sunday'}:
        return 'all'
    else:
        # For other patterns, use a compact representation
        pattern = ''.join('1' if days[i] in active_set else '0' for i in range(7))
        return f'pattern_{pattern}'


def main():
    trips = load_reference_tables()

    # Load calendar to classify services by day pattern
    print("Loading calendar.txt to classify services...")
    calendar = pd.read_csv("calendar.txt", dtype=str)
    
    # Build service -> day_pattern mapping
    service_to_pattern = {}
    for row in calendar.itertuples(index=False):
        service_to_pattern[row.service_id] = get_day_pattern(row._asdict())

    # First pass: collect all unique directions and build lookup
    print("Collecting unique directions...")
    directions_set = set(trips["direction"].unique())
    directions_list = sorted(directions_set)
    directions_lookup = {d: i for i, d in enumerate(directions_list)}
    directions_reverse = {str(i): d for i, d in enumerate(directions_list)}

    # stop_id -> "route_id|direction_id|day_pattern" -> [times]
    by_stop = {}
    routes_lookup = {}

    print("Streaming stop_times.txt in chunks and joining...")
    reader = pd.read_csv(
        STOP_TIMES_PATH,
        dtype=str,
        usecols=["trip_id", "stop_id", "departure_time"],
        chunksize=CHUNK_SIZE,
    )

    rows_processed = 0
    for chunk in reader:
        merged = chunk.join(trips, on="trip_id", how="left")
        merged = merged.dropna(subset=["route_short_name"])

        for stop_id, group in merged.groupby("stop_id"):
            stop_data = by_stop.setdefault(stop_id, {})
            
            for row in group.itertuples(index=False):
                route_id = row.route_id
                direction = row.direction
                direction_id = directions_lookup[direction]
                service_id = row.service_id
                time = row.departure_time

                # Look up the day pattern for this service
                day_pattern = service_to_pattern.get(service_id, 'unknown')

                # Build route lookup
                if route_id not in routes_lookup:
                    routes_lookup[route_id] = {
                        "name": row.route_short_name,
                        "color": row.route_color,
                    }

                # Create nested structure: route_id|direction_id|day_pattern -> [times]
                # Using pipe as separator to safely handle route_ids with underscores
                key = f"{route_id}|{direction_id}|{day_pattern}"
                if key not in stop_data:
                    stop_data[key] = []
                
                stop_data[key].append(time)

        rows_processed += len(chunk)
        print(f"  {rows_processed:,} stop_times rows processed...")

    # Sort times within each pattern and deduplicate
    for stop_data in by_stop.values():
        for key in stop_data:
            stop_data[key] = sorted(list(set(stop_data[key])))

    output = {
        "routes": routes_lookup,
        "directions": directions_reverse,
        "stops": by_stop,
    }

    print(f"Writing {OUTPUT_PATH} ({len(by_stop):,} stops)...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    print("Done.")


if __name__ == "__main__":
    main()