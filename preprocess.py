"""
Condenses stop_times.txt + trips.txt + routes.txt into a pattern-based JSON file
for the GTFS Weekly Timetable Builder.

Unlike the previous version, day-of-week classification is no longer done by
hand-parsing calendar.txt (which ignored calendar_dates.txt exceptions
entirely). Instead this uses `partridge` (https://github.com/remix/partridge)
to resolve, date-by-date, exactly which service_ids are actually running --
correctly folding in calendar_dates.txt additions/removals. This matters
because the timetable is for one *specific* week (see WEEK_START below), and
a service exception on any of those 7 dates (a public holiday, a one-off
timetable change, etc.) would previously have been silently ignored even
though calendar_dates.txt was being shipped to the browser for exactly that
purpose.

Each route/direction/day-pattern key is now a 7-character bitstring, one
character per Mon..Sun of the target week (1 = service runs that day). Common
patterns are given friendly labels (weekday/saturday/sunday/daily); anything
else is treated as a genuine exception and kept distinct, with the concrete
dates it covers recorded in the output so the client can label it.

Usage:
    pip install pandas partridge
    python3 preprocess.py

Expects a GTFS feed directory (stop_times.txt, trips.txt, routes.txt,
calendar.txt, calendar_dates.txt) at GTFS_DIR. Writes
condensed_stop_times.json in the same folder as this script.

Re-run this whenever you download an updated GTFS feed, or when you change
WEEK_START to build a timetable for a different week.
"""

import datetime
import json

import pandas as pd
import partridge as ptg

GTFS_DIR = "../gtfs/4/google_transit/"
STOP_TIMES_PATH = GTFS_DIR + "stop_times.txt"
TRIPS_PATH = GTFS_DIR + "trips.txt"
ROUTES_PATH = GTFS_DIR + "routes.txt"
OUTPUT_PATH = "condensed_stop_times.json"

# Monday of the target week. Must match getCurrentWeekDates() in index.html.
WEEK_START = datetime.date(2026, 3, 2)
WEEK_DATES = [WEEK_START + datetime.timedelta(days=i) for i in range(7)]
DAY_KEYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

CHUNK_SIZE = 1_000_000  # rows of stop_times.txt processed per chunk

# Friendly labels for the bitstrings partridge is likely to produce.
KNOWN_PATTERNS = {
    "1111100": "weekday",
    "0000010": "saturday",
    "0000001": "sunday",
    "1111111": "daily",
}


def build_service_bitstrings():
    """
    Uses partridge to resolve exactly which service_ids run on each date of
    the target week (correctly applying calendar_dates.txt exceptions), then
    turns that into a service_id -> 7-char bitstring (Mon..Sun) map.
    """
    print("Resolving service_ids per date with partridge (accounts for calendar_dates.txt exceptions)...")
    service_ids_by_date = ptg.read_service_ids_by_date(GTFS_DIR)

    week_service_sets = [service_ids_by_date.get(d, frozenset()) for d in WEEK_DATES]
    all_service_ids = set().union(*week_service_sets) if week_service_sets else set()

    service_bitstring = {}
    for service_id in all_service_ids:
        bits = "".join("1" if service_id in week_service_sets[i] else "0" for i in range(7))
        service_bitstring[service_id] = bits

    return service_bitstring


def build_pattern_metadata(service_bitstring):
    """
    Builds the day_patterns lookup shipped to the client: bitstring -> label +
    concrete dates, so the browser can render sensible column headers without
    needing calendar.txt/calendar_dates.txt itself.
    """
    used_bitstrings = set(service_bitstring.values())
    patterns = {}
    for bits in used_bitstrings:
        if bits == "0000000":
            continue
        dates = [WEEK_DATES[i].isoformat() for i in range(7) if bits[i] == "1"]
        label = KNOWN_PATTERNS.get(bits)
        if label is None:
            # Genuine exception pattern (e.g. a public holiday mid-week) --
            # describe it by the days it actually covers.
            active_names = [DAY_KEYS[i][:3].capitalize() for i in range(7) if bits[i] == "1"]
            label = "custom: " + ", ".join(active_names) if active_names else "no service"
        patterns[bits] = {"label": label, "dates": dates}
    return patterns


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

    headsign = trips["trip_headsign"].fillna("").str.strip()
    fallback = "Direction " + trips["direction_id"].fillna("")
    trips["direction"] = headsign.where(headsign != "", fallback)

    trips["route_short_name"] = trips["route_short_name"].fillna(trips["route_id"])
    trips["route_color"] = trips["route_color"].fillna("075AAA")

    trips = trips.set_index("trip_id")[
        ["route_id", "route_short_name", "route_color", "direction", "service_id"]
    ]
    return trips


def main():
    service_bitstring = build_service_bitstrings()
    day_patterns = build_pattern_metadata(service_bitstring)
    print(f"Found {len(day_patterns)} distinct service pattern(s) across the target week:")
    for bits, meta in sorted(day_patterns.items()):
        print(f"  {bits}  {meta['label']}  ({', '.join(meta['dates'])})")

    trips = load_reference_tables()

    print("Collecting unique directions...")
    directions_set = set(trips["direction"].unique())
    directions_list = sorted(directions_set)
    directions_lookup = {d: i for i, d in enumerate(directions_list)}
    directions_reverse = {str(i): d for i, d in enumerate(directions_list)}

    # stop_id -> "route_id|direction_id|day_pattern_bits" -> [times]
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
    rows_outside_week = 0
    for chunk in reader:
        merged = chunk.join(trips, on="trip_id", how="left")
        merged = merged.dropna(subset=["route_short_name"])

        for stop_id, group in merged.groupby("stop_id"):
            stop_data = by_stop.setdefault(stop_id, {})

            for row in group.itertuples(index=False):
                bits = service_bitstring.get(row.service_id)
                if not bits or bits == "0000000":
                    # This service doesn't run at all during the target week
                    # (e.g. it's only active outside WEEK_START..+6days).
                    rows_outside_week += 1
                    continue

                route_id = row.route_id
                direction_id = directions_lookup[row.direction]
                time = row.departure_time

                if route_id not in routes_lookup:
                    routes_lookup[route_id] = {
                        "name": row.route_short_name,
                        "color": row.route_color,
                    }

                key = f"{route_id}|{direction_id}|{bits}"
                if key not in stop_data:
                    stop_data[key] = []

                stop_data[key].append(time)

        rows_processed += len(chunk)
        print(f"  {rows_processed:,} stop_times rows processed...")

    for stop_data in by_stop.values():
        for key in stop_data:
            stop_data[key] = sorted(list(set(stop_data[key])))

    output = {
        "week_start": WEEK_START.isoformat(),
        "day_patterns": day_patterns,
        "routes": routes_lookup,
        "directions": directions_reverse,
        "stops": by_stop,
    }

    print(f"Writing {OUTPUT_PATH} ({len(by_stop):,} stops, {rows_outside_week:,} rows skipped as outside the target week)...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    print("Done.")


if __name__ == "__main__":
    main()