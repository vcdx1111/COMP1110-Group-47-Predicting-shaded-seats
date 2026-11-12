# Predicting shaded seats at day games in Gwangju-Kia Champions Field

COMP1110 group project (Group 47). The code in this repository produces every number and
figure in our final report: which seating sections are in direct sun, and for how long,
during a 14:00 Sunday game.

## Install

Python 3.10 or newer (developed with 3.12).

```bash
python -m venv .venv
```

Activate the environment (Windows: `.venv\Scripts\activate`, macOS/Linux:
`source .venv/bin/activate`), then install the pinned packages:

```bash
pip install -r requirements.txt
```

## Run

```bash
python scripts/run_all.py
```

```bash
python scripts/make_figures.py
```

```bash
python -m pytest
```

`run_all.py` writes all tables and the worked example (about one second). `make_figures.py`
reads those tables and draws the figures. `pytest` runs the checks listed under
[Tests](#tests).

All settings are in [data/config.yaml](data/config.yaml): dates, time window, dimensions,
sensitivity variants. Each entry has a `source` note. Change a scenario or a dimension
there, not in the code.

## Outputs

Everything under `outputs/` is generated and is not committed.

| File | Content |
|---|---|
| `outputs/tables/exposure.csv` | Direct-sun minutes E and rank for every variant, date and section |
| `outputs/tables/q_t.csv` | Shaded share q(t) at every time step, with the sun angles |
| `outputs/tables/report_exposure_table.csv` | Base exposure with the range across the sensitivity variants, as printed in the report |
| `outputs/tables/sensitivity_summary.csv` | Exposure under each variant, side by side |
| `outputs/tables/rank_stability.csv` | Does each section keep its base-case rank in every variant? |
| `outputs/tables/rank_agreement.csv` | Per variant: same shadiest section, same order, Spearman correlation |
| `outputs/tables/resolution_check.csv` | Base case rerun with 10-minute steps and with a 5 × 5 grid |
| `outputs/tables/baseline_vs_model.csv` | The under-roof baseline rule against the ray model |
| `outputs/tables/geometry_checks.csv` | Orientation, scale checks against published distances, foul-line angle |
| `outputs/tables/sample_points.csv` | Position, head height and zone of every sample point |
| `outputs/tables/points_shaded_base.csv` | Sun or shade for every sample point and time step (base case) |
| `outputs/tables/validation_*.csv` | Validation against `data/labels.csv`: counts, scores, confusion matrices, per-label predictions |
| `outputs/figures/fig1_shade_fraction.png` | q(t) on the three scenario dates, with the sensitivity range |
| `outputs/figures/fig2_shade_map.png` | Plan view: which sample points are shaded at four times |
| `outputs/figures/fig3_cross_section_3B.png` | Third-base stand cross-section with sun rays |
| `outputs/figures/fig4_sensitivity.png` | Exposure under each sensitivity variant |
| `outputs/figures/fig5_baseline_vs_model.png` | Baseline rule against the ray model |
| `outputs/figures/fig6_plan_view.png` | The digitised stadium with the sample points |
| `docs/worked_example.md` | One sample point, every calculation step with numbers (committed) |

## Method summary

1. **Geometry.** Home plate, the foul poles, centre field, three roof outlines and the
   seating sections were digitised on a satellite map. `shade.geometry` converts them to
   metres (x = east, y = north, origin = home plate) and checks the result against
   published distances. Any polygon named `SEC_*` in `data/stadium.geojson` is picked up
   automatically and gets a 3 × 3 grid of sample points.
2. **Scene.** The stadium is a set of 293 opaque triangles: the roofs (ear-clipped, flat
   at 22.5 m), the outer wall under the roofs' outer edge, and the stand surfaces built
   from a cross-section profile for each side. Each sample point sits 1.2 m (seated head
   height) above the stand surface.
3. **Sun.** pvlib (NREL Solar Position Algorithm) gives the sun's elevation and azimuth at
   the start of every 5-minute step from 14:00 to 17:00 (36 steps).
4. **Shade test.** From each sample point a ray goes towards the sun. If it hits any
   triangle (Möller–Trumbore intersection test), the point is in shade.
5. **Summary.** q(t) is the shaded share of a section's points. The direct-sun exposure is
   E = Σ (1 − q(t)) × 5 min, out of 180 minutes. Rank 1 is the shadiest section.

Scenarios: Sunday 19 April, 24 May and 20 September 2026. The summer case is in May
because the KBO schedules no 14:00 games in July.

**Sensitivity.** The model is rerun with the vertical scale ±10 %, the roof overhang ±10 %
of the roof width, and the stadium orientation ±2°. The rank-stability table shows whether
the ranking of the sections survives these changes.

**Resolution checks.** The base case is rerun with 10-minute steps and with a 5 × 5 grid.

**Baseline.** The simple rule "a seat is shaded only if it is directly under the roof" is
compared with the ray model.

**Validation.** `shade.validation` compares the model with labelled frames from real games.
`data/labels.csv` holds 204 labels from three Sunday games (26 April, 24 May and
13 September 2026), all listed as test dates in the config because nothing in the model
was tuned to them. To add labels, follow the layout of `data/labels_template.csv` and list
the new dates under `validation` in the config.

- Columns: `frame_id, date, source, timecode, time_kst, section, zone, label, labeller, excluded_reason`.
- `zone` is `front`, `middle` or `back`: one row of the section's 3 × 3 grid (front = nearest the field).
- `label` is `sun`, `shade`, `mixed` or `unknown`. Rows labelled `unknown`, and rows with an
  `excluded_reason`, are excluded.
- `labeller` should be a code (for example `L1`, `L2`), not a name.
- When a frame, section and zone is labelled by two labellers, only labels on which both
  agree enter the confusion matrices, and agreement (%) and Cohen's kappa are reported.
  The current labels come from one labeller, so `validation.use_single_labelled` is `true`
  in the config and every non-excluded label is used.
- The model predicts `sun` if 0 of the zone's 3 points are shaded, `shade` if 3 of 3, and
  `mixed` otherwise, at the exact time of the frame.
- Per date: a 3-class confusion matrix, a binary sun/shade matrix (rows where the label or
  the prediction is `mixed` are dropped, and counted), accuracy, balanced accuracy, and the
  share of predicted-shade items that were actually sun.
- Calibration dates and test dates are listed separately in the config and are never pooled.
  A date that is in neither list stops the run.

## Tests

`python -m pytest` runs:

- the five ray–triangle unit tests (hit, miss, behind, parallel, edge);
- the ear-clipping area check (triangle areas add up to the polygon area);
- agreement between the simple one-ray loop and the vectorised version;
- the regression test: with the original settings (15 April, 15 July, 15 October,
  14:00–17:00) the exposure must match `tests/reference/exposure.csv` within ±0.5 min;
- the worked-example values, the config rules, the metrics and the validation rules.

## Limitations

- **Coverage.** Four sections are modelled. Sections under the roof are hidden in satellite
  imagery and need another source.
- **Vertical scale.** The building register gives the roof at 20 m above ground; row counts
  and photo proportions give about 22.5 m above the field. The ±10 % vertical variant covers
  both, but the true value is not confirmed.
- **Riser heights** come from seat-view listings and have not been checked on site.
- **Scale checks.** The foul poles and centre field are within 1.1 % of the published
  distances and the foul-line angle is 89.8°. Second base (−8.5 %) and the stand front
  behind home plate (+18.7 % against an unofficial figure) are outside the 2 % tolerance.
  Neither point is used by the shade model.
- **Simplified shapes.** The roof is a flat opaque plate at one height. Stands follow one
  cross-section per side. Light towers, the scoreboard, railings and neighbouring buildings
  are not modelled.
- **Direct sun only.** Clouds, haze, reflected and diffuse light are ignored. A seat counts
  as shaded or sunny, with no partial shade.
- **Sampling.** Nine points per section and 5-minute steps; the resolution checks show how
  much the results move with a finer grid or a coarser step.
- **Baseline rule.** No sample point lies under a roof in plan view, so the baseline calls
  every section fully sunny on every date.
- **Validation.** The 204 labels come from one labeller, so labeller agreement and Cohen's
  kappa cannot be computed. The 13 September frames were taken between 17:05 and 18:10,
  after the 14:00–17:00 window of the scenarios. Shade is rare in the April and May labels
  (10 of 132), so the sun/shade accuracy on those dates rests mostly on sunny seats.

## Data provenance

| Data | Source |
|---|---|
| `data/stadium.geojson` | Points and polygons digitised by the group from satellite imagery (geojson.io) |
| Field distances (99 m, 121 m) | KIA Tigers stadium information |
| Roof height | Building register (20 m above ground); row arithmetic and photo proportions (22–23 m above the field) |
| Stand profiles | Row counts and riser heights from seat-view listings; depths from satellite imagery |
| Sun position | Computed with pvlib (NREL SPA); no measured data |
| `tests/reference/` | Results of the group's first model run, kept as the regression reference |
| `data/labels.csv` | Sun, shade or mixed labels read by the group from frames of three 2026 games; one labeller (code `L1`) |

The source of every individual number is noted next to it in `data/config.yaml`.

## Repository layout

```
data/        stadium outline, config, empty labels template
src/shade/   the model (geometry, sun, raytrace, model, metrics, baseline, validation)
scripts/     run_all.py (tables), make_figures.py (figures)
tests/       pytest suite and reference results
docs/        worked example
outputs/     generated tables and figures (not committed)
```
