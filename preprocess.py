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

The GTFS stop_id in stop_times.txt/stops.txt is not the number commuters
actually see or search for (e.g. on PTV signage or the transport.vic.gov.au
stop page). stops.txt's stop_url column embeds that public-facing number,
e.g. https://transport.vic.gov.au/stop/10056/?... -> "10056". This script
extracts that number from stop_url and uses it as the key everywhere in the
output JSON (stops/stop_names), so the client displays and searches by the
number people actually recognise, not the internal GTFS stop_id. Stops
without a parseable stop_url fall back to their GTFS stop_id.

Many routes run "short workings" -- trips that share a route_id and
direction_id with the route's main service but terminate early at a
different headsign (e.g. route 902 mostly runs to Airport West Shopping
Centre, but some trips only go as far as Chelsea Railway Station). Grouping
strictly by headsign (as earlier versions did) gave each of these its own
checkbox/column in the client, which is noisy and makes it hard to see the
route as a whole. Instead, for each (route_id, direction_id) pair, the
headsign with the most scheduled trips (i.e. the one that actually runs most
often) is treated as that direction's canonical headsign and shown as the
checkbox label. Every other, less-frequent headsign sharing that route +
direction is folded into the canonical one, with each of its departure times
tagged with a "Terminates at <headsign>" note that the client renders as a
superscript footnote rather than a separate row/column.

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
import re

import pandas as pd
import partridge as ptg

GTFS_DIR = "../gtfs/4/google_transit/"
STOP_TIMES_PATH = GTFS_DIR + "stop_times.txt"
TRIPS_PATH = GTFS_DIR + "trips.txt"
ROUTES_PATH = GTFS_DIR + "routes.txt"
STOPS_PATH = GTFS_DIR + "stops.txt"
OUTPUT_PATH = "condensed_stop_times.json"

# Monday of the target week. Must match getCurrentWeekDates() in index.html.
WEEK_START = datetime.date(2026, 7, 27)
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
    trips["direction_id"] = trips["direction_id"].fillna("")

    print("Loading routes.txt...")
    routes = pd.read_csv(
        ROUTES_PATH,
        dtype=str,
        usecols=["route_id", "route_short_name", "route_color"],
    )

    trips = trips.merge(routes, on="route_id", how="left")

    headsign = trips["trip_headsign"].fillna("").str.strip()
    fallback = "Direction " + trips["direction_id"]
    trips["direction"] = headsign.where(headsign != "", fallback)

    trips["route_short_name"] = trips["route_short_name"].fillna(trips["route_id"])
    trips["route_color"] = trips["route_color"].fillna("075AAA")

    trips = trips.set_index("trip_id")[
        ["route_id", "direction_id", "route_short_name", "route_color", "direction", "service_id"]
    ]
    return trips


def compute_canonical_headsigns(trips):
    """
    For each (route_id, direction_id) pair, picks the headsign with the most
    scheduled trips (i.e. the one that actually runs most often) as that
    direction's canonical headsign. Every other, less-frequent headsign
    sharing the same route_id + direction_id is a short working of it.

    Returns (canonical, short_working_count) where:
      canonical            -> { (route_id, direction_id): canonical_headsign_text }
      short_working_count  -> number of distinct headsigns folded into some
                               other canonical headsign (for logging)
    """
    trip_counts = (
        trips.reset_index()
        .groupby(["route_id", "direction_id", "direction"])
        .size()
        .reset_index(name="trip_count")
    )

    canonical = {}
    short_working_count = 0
    for (route_id, direction_id), sub in trip_counts.groupby(["route_id", "direction_id"]):
        best_row = sub.loc[sub["trip_count"].idxmax()]
        canonical[(route_id, direction_id)] = best_row["direction"]
        short_working_count += len(sub) - 1

    return canonical, short_working_count


STOP_URL_NUMBER_RE = re.compile(r"/stop/(\d+)/")


def load_stop_names_and_display_ids():
    """
    Returns (stop_names, display_id) where both are keyed by the internal
    GTFS stop_id:
      - stop_names[gtfs_id]  -> stop_name
      - display_id[gtfs_id]  -> public-facing stop number parsed out of
                                 stop_url (e.g. ".../stop/10056/..." -> "10056"),
                                 falling back to gtfs_id itself if stop_url is
                                 missing or doesn't match the expected pattern.
    """
    print("Loading stops.txt...")
    stops = pd.read_csv(
        STOPS_PATH,
        dtype=str,
        usecols=["stop_id", "stop_name", "stop_url"],
        encoding="utf-8-sig",   # strips leading BOM if present
    )

    stop_names = stops.set_index("stop_id")["stop_name"].to_dict()

    display_id = {}
    unmatched = 0
    for row in stops.itertuples(index=False):
        match = STOP_URL_NUMBER_RE.search(row.stop_url) if isinstance(row.stop_url, str) else None
        if match:
            display_id[row.stop_id] = match.group(1)
        else:
            display_id[row.stop_id] = row.stop_id
            unmatched += 1

    if unmatched:
        print(f"  Note: {unmatched:,} stop(s) had no parseable stop_url; "
              f"falling back to their GTFS stop_id as the display id.")

    return stop_names, display_id


def finalize_entries(raw_entries):
    """
    raw_entries: list of (time, note_or_None) tuples collected while
    streaming stop_times.txt.

    Dedupes by exact departure time, merging any notes attached to that time
    (e.g. "Terminates at X" for a short working), and returns the format the
    client expects: a bare "HH:MM:SS" string when there are no notes, or
    [time, [note, ...]] when there are.
    """
    by_time = {}
    for time, note in raw_entries:
        notes = by_time.setdefault(time, set())
        if note:
            notes.add(note)
    result = []
    for time in sorted(by_time):
        notes = sorted(by_time[time])
        result.append([time, notes] if notes else time)
    return result


def main():
    service_bitstring = build_service_bitstrings()

    if not service_bitstring:
        print("ERROR: No service IDs found for the target week "
              f"({WEEK_START} to {WEEK_DATES[-1]}).")
        print("       The GTFS feed may not cover this date range.")
        print("       Checking what dates partridge can actually see...")
        all_dates = ptg.read_service_ids_by_date(GTFS_DIR)
        if all_dates:
            dates = sorted(all_dates.keys())
            print(f"       Feed covers: {dates[0]} to {dates[-1]}")
            print(f"       Fix: set WEEK_START to a Monday within that range and re-run.")
        else:
            print("       partridge found NO dates at all.")
            print("       Check that GTFS_DIR points to a valid feed with calendar.txt "
                  "or calendar_dates.txt.")
        raise SystemExit(1)

    day_patterns = build_pattern_metadata(service_bitstring)
    print(f"Found {len(day_patterns)} distinct service pattern(s) across the target week:")
    for bits, meta in sorted(day_patterns.items()):
        print(f"  {bits}  {meta['label']}  ({', '.join(meta['dates'])})")

    stop_names, display_id = load_stop_names_and_display_ids()
    trips = load_reference_tables()

    canonical_headsigns, short_working_count = compute_canonical_headsigns(trips)
    print(f"Found {short_working_count} short-working headsign(s) across "
          f"{len(canonical_headsigns)} route/direction group(s); folding them into "
          f"their parent route with a 'Terminates at' footnote.")

    print("Collecting canonical directions...")
    directions_set = set(canonical_headsigns.values())
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
                canonical = canonical_headsigns.get((route_id, row.direction_id), row.direction)
                direction_index = directions_lookup[canonical]
                note = None if row.direction == canonical else f"Terminates at {row.direction}"
                time = row.departure_time

                if route_id not in routes_lookup:
                    routes_lookup[route_id] = {
                        "name": row.route_short_name,
                        "color": row.route_color,
                    }

                key = f"{route_id}|{direction_index}|{bits}"
                stop_data.setdefault(key, []).append((time, note))

        rows_processed += len(chunk)
        print(f"  {rows_processed:,} stop_times rows processed...")

    print("Remapping GTFS stop_id -> public-facing (transport.vic.gov.au) stop number...")
    display_stops = {}
    display_stop_names = {}
    collisions = 0
    for gtfs_id, stop_data in by_stop.items():
        disp_id = display_id.get(gtfs_id, gtfs_id)
        if disp_id in display_stops:
            # Two GTFS stop_ids resolved to the same public-facing number --
            # merge their departures rather than silently dropping one.
            collisions += 1
            existing = display_stops[disp_id]
            for key, entries in stop_data.items():
                existing.setdefault(key, []).extend(entries)
        else:
            display_stops[disp_id] = stop_data
        display_stop_names[disp_id] = stop_names.get(gtfs_id, disp_id)

    if collisions:
        print(f"  Note: {collisions:,} GTFS stop(s) shared a display id with another "
              f"stop and were merged together.")

    for stop_data in display_stops.values():
        for key in stop_data:
            stop_data[key] = finalize_entries(stop_data[key])

    output = {
        "week_start": WEEK_START.isoformat(),
        "day_patterns": day_patterns,
        "routes": routes_lookup,
        "directions": directions_reverse,
        "stop_names": display_stop_names,
        "stops": display_stops,
    }

    print(f"Writing {OUTPUT_PATH} ({len(display_stops):,} stops, {rows_outside_week:,} rows skipped as outside the target week)...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    print("Done.")


if __name__ == "__main__":
    main()