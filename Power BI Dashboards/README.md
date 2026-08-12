# Power BI Dashboards

## Overview

This folder contains two independent Power BI projects. Marriott SAT Workforce Analytics is a pure Power BI / DAX workforce-compliance dashboard built on real Kronos and When-to-Work timekeeping exports. RH Furniture Delivery Optimization pairs a Power BI report with a Python optimization model that runs inside Power Query (via Power BI's "Run Python script" data transform) to solve daily delivery routing, load-building, labor cost, and CO2 for a furniture White Glove delivery operation.

## Contents

```
Power BI Dashboards/
├── README.md
├── Marriott SAT Workforce Analytics/
│   ├── Marriott_SAT_Workforce_BI.pbix
│   └── Executive Summary Report.docx
└── RH Furniture Delivery Optimization/
    ├── RH Furniture Dashboard.pbix
    ├── RH Furniture Optimization Model.py      # runs inside the .pbix as a Power Query Python script
    ├── RH_Input_Template.xlsx                  # source workbook: orders, addresses, distance matrix, catalog, assumptions
    └── RH Management Recomendation Memo.docx
```

## Security Note

`RH_Input_Template.xlsx` originally had a real OpenRouteService API key hardcoded in its Assumptions sheet (used to geocode the distance matrix). It has been redacted in the version here to a placeholder (`SET_VIA_ENV:OPENROUTESERVICE_API_KEY`); every other cell in the workbook is byte-identical to the original. Since that key was about to be committed to a public repository, it is worth rotating it in the OpenRouteService dashboard regardless of this cleanup.

---

## Project 1: Marriott SAT Workforce Analytics

### Executive Summary

Three systemic risks are driving labor cost inflation and compliance exposure at the property analyzed: $314,550.00 in preventable overtime, 997 unresolved timekeeping anomalies, and 97 student workers nearing thresholds that trigger benefits eligibility and up to 4x cost increases.

Overtime totals $314,550.00, or 13% of payroll, with 92% concentrated in Dining and Custodial; some employees exceed 80 hours per week under existing unionized scheduling constraints, driving reliance on $37.50/hr overtime instead of $19.17/hr student labor, a 96% cost premium. Kronos's weak validation controls have let 997 timekeeping anomalies (missed punches, overnight shifts, suspected rounding manipulation) go unresolved, spiking in February-March 2026. Separately, 97 student workers are approaching the 19-hour weekly cap and one has already exceeded the 950-hour lifetime threshold, a breach that triggers benefits eligibility or reclassification at up to 4x cost impact, compounded by 30% absenteeism in Events.

The recommended fix for the immediate risk (an unstaffed September 2026 Travel Market Symposium) is a pre-assigned staffing plan adding two student workers per shift across Custodial and Dining at $19.17/hr for the six-day event, costing approximately $1,840.00 in additional labor while reducing overtime cost by 98%. HR is separately recommended to clear all 997 open anomalies and hold open anomalies below 20 per pay period from May 2026 onward.

A stated limitation: W2W and Kronos have no shared employee key, so fuzzy matching was used to link records, matching 156 of 210 employees and leaving 54 unmatched, which likely understates absenteeism. Timekeeping data starts July 1, 2025, so pre-period hours are missing and the 950/1,000-hour lifetime threshold reflects a partial view.

### Data

Real Kronos punch/timekeeping exports and When-to-Work (W2W) scheduling exports for one property, linked via fuzzy name/ID matching due to the absence of a shared employee key. Neither raw source file is included in this repository; the cleaned, linked model lives inside the `.pbix`.

### Report Pages

Executive Summary, Department Manager Dashboard, HR Compliance & Punch Audit, Overtime Tracking, Absenteeism & Schedule Adherence (five pages, confirmed from the report's own layout metadata).

### How to Open

Open `Marriott_SAT_Workforce_BI.pbix` in Power BI Desktop, or view the published report:
https://app.powerbi.com/groups/me/reports/ab9d8e05-0be9-463c-bf71-351f65be4a59/638b83900b32bad849e6?experience=power-bi

---

## Project 2: RH Furniture Delivery Optimization

### Executive Summary

RH Unlimited Furniture's current flat-rate delivery fee of $299.00 per order is significantly below the true cost of White Glove delivery. The optimization model, run against real order, customer address, furniture catalog, and operational-assumption data, computes an average delivery cost of $1,071.00 per order, a shortfall of $772.00 per delivery, meaning the current fee covers less than one-third of actual cost. Efficient local multi-stop deliveries cost about $267.00; longer-distance, high-volume deliveries can exceed $1,440.00, so a single flat rate is economically misaligned across the order mix. Warehouse holding costs for pre-packaged inventory awaiting delivery windows account for roughly 70% of total cost, with longer customer-caused delays increasing this further.

Against a naive one-truck-per-order baseline, the optimizer's multi-stop route sequencing and bin-packing (by volume and weight, subject to the delivery time window) reduced travel by 114 miles and CO2 emissions by 47 kg over the modeled schedule.

The recommendation is a dynamic pricing model: raise the base fee to $599.00, add variable tiers by distance and load volume, and introduce deferred-delivery storage fees to directly offset holding costs driven by customer-side delays, positioned to customers as a "Sustainable Luxury Delivery" offer.

### Method

The Python model (embedded in the `.pbix` as a Power Query "Run Python script" transform, reproduced standalone in `RH Furniture Optimization Model.py`) does the following against a 26-ft box truck with 1,700 cu ft / 7,000 lb capacity:

1. Splits each customer's order into truck-sized trips (bin-packed by volume and weight) when a single order exceeds truck capacity.
2. Groups trips into delivery days, greedily assigning each trip to the first day with remaining capacity whose estimated route (built by brute-force permutation over stops, since stop counts per day are small) finishes inside the 8:00 AM-5:00 PM delivery window.
3. For the placed trips each day, brute-force searches stop permutations for the minimum-distance route, then builds a full timeline (load time, drive time, per-stop unload/stage time, return leg).
4. Prices each day's route on labor cost (driver + 2 crew, wage premium and benefits load applied, pay rounded to the nearest 15 minutes), a full operating-cost breakdown (fuel, maintenance, insurance, overhead, equipment financing, variable driving expenses, all $/mile), and inventory holding cost (per-item-class $/day while awaiting its delivery day).
5. Computes CO2 from driving distance and idling time, and compares both distance and CO2 against a naive per-order round-trip baseline to quantify the savings from consolidation.

### Data

`RH_Input_Template.xlsx` contains the real (company-internal) source data: 165 order lines across active customers, a 375-row customer address book with geocoordinates, a full origin-destination distance/duration matrix, a 45-item furniture catalog (volume, weight, size class, loading/unloading time), and a 48-parameter operational assumptions sheet (wages, benefits load, truck capacity, per-mile cost bounds sourced from ATRI/EIA/EPA benchmarks, delivery window, holding costs, and the current $299 flat fee) that the model reads directly.

### How to Run

```bash
pip install pandas
```

Open `RH Furniture Dashboard.pbix` in Power BI Desktop with `RH_Input_Template.xlsx` as the data source to re-run the embedded Python step, or view the published report:
https://app.powerbi.com/groups/2440a4db-484b-4bac-bd15-1d6cefe06f27/reports/1288c49f-da9d-4f10-b8d0-814af8eb0c4d/bc6d1ffcea273b9749ee?experience=power-bi

### Report Pages

Driver Route Schedule, Truck Loading Manifest, Cost Analysis, CO2 Analysis, Assumptions (five pages, confirmed from the report's own layout metadata).
