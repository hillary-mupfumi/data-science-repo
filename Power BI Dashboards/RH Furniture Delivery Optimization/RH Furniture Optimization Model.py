import pandas as pd
import datetime
import json
from itertools import permutations

# ── Prepare data ──────────────────────────────────────────────────────────────
df = dataset.copy()

TRUCK_VOL        = pd.to_numeric(df['Truck_Vol'],            errors='coerce').iloc[0]
TRUCK_WGT        = pd.to_numeric(df['Truck_Wgt'],            errors='coerce').iloc[0]
START_DATE       = pd.to_datetime(df['START_DATE'].iloc[0])
WAGE_CREW        = pd.to_numeric(df['Wage_Crew'],            errors='coerce').iloc[0]
WAGE_DRIVER_PREM = pd.to_numeric(df['Wage_Driver_Prem'],     errors='coerce').iloc[0]
BENEFITS_LOAD    = pd.to_numeric(df['Benefits_Load'],        errors='coerce').iloc[0]
AVG_OP_COST      = pd.to_numeric(df['Avg_operating_cost'],   errors='coerce').iloc[0]
CO2_DRIVING      = pd.to_numeric(df['CO2_Driving'],          errors='coerce').iloc[0]
CO2_IDLING       = pd.to_numeric(df['CO2_Idling'],           errors='coerce').iloc[0]
HOLD_LARGE       = pd.to_numeric(df['Hold_Large'],           errors='coerce').iloc[0]
HOLD_SMALL       = pd.to_numeric(df['Hold_Small'],           errors='coerce').iloc[0]
PAY_ROUND_MIN    = pd.to_numeric(df['Pay_Round_Min'],        errors='coerce').iloc[0]
FIXED_DAY_START  = pd.to_numeric(df['Fixed_TimeDay_Start'],  errors='coerce').iloc[0]
FIXED_STOP_START = pd.to_numeric(df['Fixed_Time_Stop_Start'],errors='coerce').iloc[0]
FIXED_STOP_END   = pd.to_numeric(df['Fixed_Time_Stop_End'],  errors='coerce').iloc[0]
WINDOW_END   = pd.to_numeric(df['WINDOW_END'],  errors='coerce').iloc[0]
WINDOW_START   = pd.to_numeric(df['WINDOW_START'],  errors='coerce').iloc[0]


# ── Read cost breakdown (midpoints — display only) ────────────────
def mid(col_low, col_high):
    lo = pd.to_numeric(df[col_low],  errors='coerce').iloc[0]
    hi = pd.to_numeric(df[col_high], errors='coerce').iloc[0]
    return round((lo + hi) / 2, 4)

COST_FUEL        = mid('Fuel-low',                      'Fuel-high')
COST_MAINTENANCE = mid('Maintenance-low',                'Maintenance-high')
COST_INSURANCE   = mid('Insurance-low',                  'Insurance-high')
COST_OVERHEAD    = mid('Fixed Overhead-low',             'Fixed Overhead-high')
COST_EQUIP_FIN   = mid('Equipment Finance-low',          'Equipment Finance-high')
COST_VAR_DRIVE   = mid('Variable Driving Expenses-low',  'Variable Driving Expenses-high')

# ── Read distance matrix ──────────────────────────────────────────────────────
dist_records = json.loads(df['DistMatrix'].iloc[0])
dist_df      = pd.DataFrame(dist_records)
dist_lookup  = {}
dur_lookup   = {}
for _, row in dist_df.iterrows():
    key = (row['Origin'], row['Destination'])
    dist_lookup[key] = float(row['Distance (miles)'])
    dur_lookup[key]  = float(row['Duration (min)'])

# ── Drop helper columns ───────────────────────────────────────────────────────
drop_cols = ['Truck_Vol','Truck_Wgt','START_DATE','DistMatrix',
             'Wage_Crew','Wage_Driver_Prem','Benefits_Load','Avg_operating_cost',
             'CO2_Driving','CO2_Idling','Hold_Large','Hold_Small','Pay_Round_Min',
             'Fixed_TimeDay_Start','Fixed_Time_Stop_Start','Fixed_Time_Stop_End',
             'Fuel-low','Fuel-high',
             'Maintenance-low','Maintenance-high',
             'Insurance-low','Insurance-high',
             'Fixed Overhead-low','Fixed Overhead-high',
             'Equipment Finance-low','Equipment Finance-high',
             'Variable Driving Expenses-low','Variable Driving Expenses-high']
df = df.drop(columns=drop_cols, errors='ignore')

# ── Helper functions ──────────────────────────────────────────────────────────
def get_dist(o, d):
    return dist_lookup.get((o, d), dist_lookup.get((d, o), 0))

def get_dur(o, d):
    return dur_lookup.get((o, d), dur_lookup.get((d, o), 0))

def total_route_dist(stop_list):
    locs = ['Warehouse'] + stop_list + ['Warehouse']
    return sum(get_dist(locs[i], locs[i+1]) for i in range(len(locs)-1))

def total_route_dur(stop_list):
    locs = ['Warehouse'] + stop_list + ['Warehouse']
    return sum(get_dur(locs[i], locs[i+1]) for i in range(len(locs)-1))

def round_pay(hours, interval_min):
    minutes   = hours * 60
    threshold = interval_min / 2
    slots     = minutes / interval_min
    remainder = (slots % 1) * interval_min
    rounded   = (int(slots) + (1 if remainder >= threshold else 0)) * interval_min
    return rounded / 60

def mins_to_time(mins):
    h = int(mins // 60)
    m = int(mins % 60)
    return str(h).zfill(2) + ":" + str(m).zfill(2)

def get_load_time(trip_ids, items):
    day_items = items[items['Trip ID'].isin(trip_ids)]
    total = day_items['Total loading time (min)'].sum()
    return round(FIXED_DAY_START + total, 2)

def get_unload_time(trip_id, items):
    trip_items = items[items['Trip ID'] == trip_id]
    total = trip_items['Total unloading time (min)'].sum()
    return round(FIXED_STOP_START + total + FIXED_STOP_END, 2)

def get_delivery_date(day_num, start_date):
    date = pd.to_datetime(start_date)
    working_days = 0
    while working_days < day_num:
        date += datetime.timedelta(days=1)
        if date.weekday() not in [6, 0]:
            working_days += 1
    return date.strftime('%Y-%m-%d')

def estimate_end_time(trip_ids, cust_ids, items):
    if len(cust_ids) == 1:
        best_route = cust_ids
    else:
        best_route = cust_ids
        best_dist  = float('inf')
        for perm in permutations(cust_ids):
            d = total_route_dist(list(perm))
            if d < best_dist:
                best_dist  = d
                best_route = list(perm)
    full_route   = ['Warehouse'] + best_route + ['Warehouse']
    load_min     = get_load_time(trip_ids, items)
    current_time = WINDOW_START + load_min
    for i, cust in enumerate(best_route):
        leg_dur  = get_dur(full_route[i], full_route[i+1])
        arrive   = current_time + leg_dur
        trip_id  = next((t for t in trip_ids if t.startswith(cust)), None)
        unload   = get_unload_time(trip_id, items) if trip_id else 30
        current_time = arrive + unload
    return current_time + get_dur(best_route[-1], 'Warehouse')

# ── Build item-level trips ──────────────────────────────────────────
trips = []
for cust_id, group in df.groupby('Customer ID', sort=False):
    group    = group.sort_values('Total Volume (cu ft)', ascending=False)
    trip_idx = 0
    trip_vol = 0
    trip_wgt = 0
    for _, row in group.iterrows():
        item_vol = row['Total Volume (cu ft)']
        item_wgt = row['Total Weight (lbs)']
        if trip_vol + item_vol > TRUCK_VOL or trip_wgt + item_wgt > TRUCK_WGT:
            trip_idx += 1
            trip_vol  = 0
            trip_wgt  = 0
        trip_label = chr(65 + trip_idx)
        trip_id    = cust_id + "_" + trip_label
        trips.append({
            'Trip ID':                    trip_id,
            'Customer ID':                cust_id,
            'Trip':                       trip_label,
            'Item Name':                  row['Brand'],
            'Variant / Size':             row['Variant / Size'],
            'Category':                   row['Category'],
            'Size Class':                 row['Size Class'],
            'Qty':                        row['Qty'],
            'Item Volume (cu ft)':        item_vol,
            'Item Weight (lbs)':          item_wgt,
            'Total loading time (min)':   row['Total loading time (min)'],
            'Total unloading time (min)': row['Total unloading time (min)'],
            'Needs Split':                trip_idx > 0,
        })
        trip_vol += item_vol
        trip_wgt += item_wgt

items_df = pd.DataFrame(trips)

# ── Trip summary + bin packing with window constraint ────────────────
trip_summary = (
    items_df
    .groupby(['Trip ID', 'Customer ID', 'Trip'], as_index=False)
    .agg(Trip_Volume=('Item Volume (cu ft)', 'sum'),
         Trip_Weight=('Item Weight (lbs)',   'sum'))
    .round(2)
)

trips_sorted = trip_summary.sort_values('Trip_Volume', ascending=False)
days = []

for _, trip in trips_sorted.iterrows():
    trip_id = trip['Trip ID']
    cust_id = trip['Customer ID']
    vol     = trip['Trip_Volume']
    wgt     = trip['Trip_Weight']
    placed  = False

    for d in days:
        if cust_id in d['customers']:
            continue
        if d['vol'] + vol > TRUCK_VOL or d['wgt'] + wgt > TRUCK_WGT:
            continue
        proposed_trips = d['trips'] + [trip_id]
        proposed_custs = list(d['customers']) + [cust_id]
        est_end = estimate_end_time(proposed_trips, proposed_custs, items_df)
        if est_end > WINDOW_END:
            continue
        d['trips'].append(trip_id)
        d['customers'].add(cust_id)
        d['vol'] += vol
        d['wgt'] += wgt
        placed = True
        break

    if not placed:
        days.append({
            'trips':     [trip_id],
            'customers': {cust_id},
            'vol':       vol,
            'wgt':       wgt,
        })

# ── Naive baseline ────────────────────────────────────────────────────────────
naive_dist_by_trip = {}
for _, trip in trip_summary.iterrows():
    cust_id = trip['Customer ID']
    trip_id = trip['Trip ID']
    naive_dist_by_trip[trip_id] = round(
        get_dist('Warehouse', cust_id) + get_dist(cust_id, 'Warehouse'), 2)

# ── Route sequencing + costs + manifest ─────────────────────────
trip_lookup   = trip_summary.set_index('Trip ID').to_dict('index')
schedule_rows = []
cost_rows     = []
manifest_rows = []
cum_cost      = 0
cum_co2       = 0
cum_naive_co2 = 0
CREW_ASSIGNMENT = "1 Driver + 2 Crew"

for day_num, d in enumerate(days, 1):
    date      = get_delivery_date(day_num, START_DATE)
    trip_list = d['trips']
    customers = [trip_lookup[t]['Customer ID'] for t in trip_list]

    # ── Route sequencing ─────────────────────────────────────────────────────
    if len(trip_list) == 1:
        best_order = [0]
    else:
        best_order    = list(range(len(customers)))
        best_distance = float('inf')
        for perm in permutations(range(len(customers))):
            ordered = [customers[i] for i in perm]
            dist    = total_route_dist(ordered)
            if dist < best_distance:
                best_distance = dist
                best_order    = list(perm)

    best_customers = [customers[i] for i in best_order]
    route_dist     = round(total_route_dist(best_customers), 2)
    route_dur      = round(total_route_dur(best_customers),  2)
    full_route     = ['Warehouse'] + best_customers + ['Warehouse']
    ordered_trips  = [trip_list[i] for i in best_order]

    # ── Timeline ─────────────────────────────────────────────────────────────
    load_time_min    = get_load_time(ordered_trips, items_df)
    total_unload_min = sum(get_unload_time(t, items_df) for t in ordered_trips)
    warehouse_depart = WINDOW_START + load_time_min
    current_time     = warehouse_depart

    stop_arrive = []
    stop_depart = []
    stop_unload = []

    for stop_num, idx in enumerate(best_order, 1):
        trip_id  = trip_list[idx]
        leg_dur  = get_dur(full_route[stop_num-1], full_route[stop_num])
        arrive   = current_time + leg_dur
        unload   = get_unload_time(trip_id, items_df)
        depart   = arrive + unload
        stop_arrive.append(arrive)
        stop_depart.append(depart)
        stop_unload.append(unload)
        current_time = depart

    return_dur    = get_dur(best_customers[-1], 'Warehouse')
    end_time_min  = current_time + return_dur
    end_time_str  = mins_to_time(end_time_min)
    within_window = end_time_min <= WINDOW_END

    # ── Delivery stop rows ────────────────────────────────────────────────────
    for stop_num, idx in enumerate(best_order, 1):
        trip_id   = trip_list[idx]
        trip_info = trip_lookup[trip_id]
        leg_from  = full_route[stop_num - 1]
        leg_to    = full_route[stop_num]
        arrive    = stop_arrive[stop_num - 1]
        depart    = stop_depart[stop_num - 1]
        unload    = stop_unload[stop_num - 1]

        if stop_num == 1:
            source_depart = mins_to_time(warehouse_depart)
        else:
            source_depart = mins_to_time(stop_depart[stop_num - 2])

        schedule_rows.append({
            'Day':                    day_num,
            'Date':                   date,
            'Crew Assignment':        CREW_ASSIGNMENT,
            'Trip ID':                trip_id,
            'Customer ID':            trip_info['Customer ID'],
            'Stop Order':             stop_num,
            'Leg From':               leg_from,
            'Leg To':                 leg_to,
            'Source Depart Time':     source_depart,
            'Drive Time (min)':       round(get_dur(leg_from, leg_to), 2),
            'Leg Distance (miles)':   round(get_dist(leg_from, leg_to), 2),
            'Arrive Time':            mins_to_time(arrive),
            'Depart Time':            mins_to_time(depart),
            'Delivery Window Start':  mins_to_time(arrive),
            'Delivery Window End':    mins_to_time(depart),
            'Unload + Stage (min)':   round(unload, 2),
            'Trip Volume':            round(trip_info['Trip_Volume'], 2),
            'Trip Weight':            round(trip_info['Trip_Weight'], 2),
            'Day Volume':             round(d['vol'], 2),
            'Day Weight':             round(d['wgt'], 2),
            'Vol Used %':             round(d['vol'] / TRUCK_VOL * 100, 1),
            'Route Distance (miles)': route_dist,
            'Route Duration (min)':   route_dur,
        })

    # ── Return leg row ────────────────────────────────────────────────────────
    schedule_rows.append({
        'Day':                    day_num,
        'Date':                   date,
        'Crew Assignment':        CREW_ASSIGNMENT,
        'Trip ID':                'RETURN',
        'Customer ID':            'Warehouse',
        'Stop Order':             len(best_customers) + 1,
        'Leg From':               best_customers[-1],
        'Leg To':                 'Warehouse',
        'Source Depart Time':     mins_to_time(stop_depart[-1]),
        'Drive Time (min)':       round(return_dur, 2),
        'Leg Distance (miles)':   round(get_dist(best_customers[-1], 'Warehouse'), 2),
        'Arrive Time':            end_time_str,
        'Depart Time':            end_time_str,
        'Delivery Window Start':  '',
        'Delivery Window End':    '',
        'Unload + Stage (min)':   0,
        'Trip Volume':            0,
        'Trip Weight':            0,
        'Day Volume':             round(d['vol'], 2),
        'Day Weight':             round(d['wgt'], 2),
        'Vol Used %':             round(d['vol'] / TRUCK_VOL * 100, 1),
        'Route Distance (miles)': route_dist,
        'Route Duration (min)':   route_dur,
    })

    # ── fact_manifest ─────────────────────────────────────────────────────────
    num_stops = len(ordered_trips)
    for load_pos, trip_id in enumerate(reversed(ordered_trips), 1):
        delivery_stop = num_stops - load_pos + 1
        trip_items = items_df[items_df['Trip ID'] == trip_id].copy()
        trip_items = trip_items.sort_values('Item Volume (cu ft)', ascending=False)
        for _, item in trip_items.iterrows():
            manifest_rows.append({
                'Day':                        day_num,
                'Date':                       date,
                'Load Position':              load_pos,
                'Delivery Stop':              delivery_stop,
                'Trip ID':                    trip_id,
                'Customer ID':                item['Customer ID'],
                'Item Name':                  item['Item Name'],
                'Variant / Size':             item['Variant / Size'],
                'Category':                   item['Category'],
                'Size Class':                 item['Size Class'],
                'Qty':                        item['Qty'],
                'Item Volume (cu ft)':        item['Item Volume (cu ft)'],
                'Item Weight (lbs)':          item['Item Weight (lbs)'],
                'Total Loading Time (min)':   item['Total loading time (min)'],
                'Total Unloading Time (min)': item['Total unloading time (min)'],
            })

    # ── fact_costs_co2 ────────────────────────────────────────────────────────
    total_work_min = load_time_min + route_dur + total_unload_min
    total_work_hrs = round_pay(total_work_min / 60, PAY_ROUND_MIN)

    wage_driver  = WAGE_CREW * (1 + WAGE_DRIVER_PREM)
    labor_cost   = round(
        (2 * WAGE_CREW + wage_driver) * total_work_hrs * (1 + BENEFITS_LOAD), 2)

    # Operating cost — full Exhibit III midpoint (display breakdown only)
    operating_cost  = round(route_dist * AVG_OP_COST, 2)
    fuel_cost       = round(route_dist * COST_FUEL,        2)
    maintenance_cost= round(route_dist * COST_MAINTENANCE, 2)
    insurance_cost  = round(route_dist * COST_INSURANCE,   2)
    overhead_cost   = round(route_dist * COST_OVERHEAD,     2)
    equip_fin_cost  = round(route_dist * COST_EQUIP_FIN,    2)
    var_drive_cost  = round(route_dist * COST_VAR_DRIVE,    2)

    days_waiting = day_num - 1
    day_items    = items_df[items_df['Trip ID'].isin(ordered_trips)]
    large_qty    = day_items[day_items['Size Class'] == 'Large']['Qty'].sum()
    small_qty    = day_items[day_items['Size Class'] == 'Small']['Qty'].sum()
    holding_cost = round(
        (large_qty * HOLD_LARGE + small_qty * HOLD_SMALL) * days_waiting, 2)

    # Total cost uses full operating cost — breakdown columns are display only
    total_cost   = round(labor_cost + operating_cost + holding_cost, 2)

    co2_driving  = round(route_dist * CO2_DRIVING, 2)
    idling_hrs   = round((load_time_min + total_unload_min) / 60, 4)
    co2_idling   = round(idling_hrs * CO2_IDLING, 2)
    co2_total    = round(co2_driving + co2_idling, 2)

    day_naive_dist = sum(naive_dist_by_trip.get(t, 0) for t in ordered_trips)
    day_naive_co2  = round(day_naive_dist * CO2_DRIVING, 2)
    co2_saved      = round(day_naive_co2 - co2_total, 2)
    dist_saved     = round(day_naive_dist - route_dist, 2)

    cum_cost      = round(cum_cost + total_cost, 2)
    cum_co2       = round(cum_co2 + co2_total, 2)
    cum_naive_co2 = round(cum_naive_co2 + day_naive_co2, 2)

    cost_rows.append({
        'Day':                       day_num,
        'Date':                      date,
        'Crew Assignment':           CREW_ASSIGNMENT,
        'Customers':                 ', '.join(best_customers),
        'Num Stops':                 len(best_customers),
        'Start Time':                '08:00',
        'Loading End Time':          mins_to_time(warehouse_depart),
        'End Time':                  end_time_str,
        'Within Window':             within_window,
        'Loading Time (min)':        round(load_time_min, 2),
        'Unloading Time (min)':      round(total_unload_min, 2),
        'Total Work Min':            round(total_work_min, 2),
        'Total Work Hrs':            total_work_hrs,
        'Route Distance (miles)':    route_dist,
        'Route Duration (min)':      route_dur,
        'Labor Cost ($)':            labor_cost,
        'Operating Cost ($)':        operating_cost,
        'Fuel Cost ($)':             fuel_cost,
        'Maintenance Cost ($)':      maintenance_cost,
        'Insurance Cost ($)':        insurance_cost,
        'Overhead Cost ($)':         overhead_cost,
        'Equip Finance Cost ($)':    equip_fin_cost,
        'Variable Drive Cost ($)':   var_drive_cost,
        'Holding Cost ($)':          holding_cost,
        'Total Cost ($)':            total_cost,
        'Cumulative Cost ($)':       cum_cost,
        'CO2 Driving (kg)':          co2_driving,
        'CO2 Idling (kg)':           co2_idling,
        'CO2 Total (kg)':            co2_total,
        'Idling Hours':              idling_hrs,
        'Naive Distance (miles)':    round(day_naive_dist, 2),
        'Naive CO2 (kg)':            day_naive_co2,
        'CO2 Saved (kg)':            co2_saved,
        'Distance Saved (miles)':    dist_saved,
        'Cumulative CO2 (kg)':       cum_co2,
        'Cumulative Naive CO2 (kg)': cum_naive_co2,
        'Cumulative CO2 Saved (kg)': round(cum_naive_co2 - cum_co2, 2),
    })

schedule_df = pd.DataFrame(schedule_rows)
costs_df    = pd.DataFrame(cost_rows)
manifest_df = pd.DataFrame(manifest_rows)