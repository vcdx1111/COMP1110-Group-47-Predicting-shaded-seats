# Worked example — one sample point, two times

Section **529 (3B upper)**, sample point 5 of 9 (the middle point).

- plan position: x = -46.87 m east, y = 56.78 m north of home plate
- seat surface at this point: 17.33 m; head height p_z = seat + 1.2 = **18.53 m**
- p = **(-46.87, 56.78, 18.53)**

Scene: base case. The roof front edge is at 22.5 m, and the scene has 293 triangles (37 roof, 52 wall, 204 stand).

Dates: **15 Oct** (cases A and B); **20 Sep** (cases C and D).

## Case A (sunny): 15 Oct 15:00 KST

**Step 1 — sun position (pvlib).**
Elevation α = 32.02°, azimuth A = 229.13° (clockwise from north).

**Step 2 — direction from the seat towards the sun.**
s = (cos α sin A, cos α cos A, sin α)
= (0.8478×-0.7562, 0.8478×-0.6543, 0.5303) = **(-0.6412, -0.5547, 0.5303)**

**Step 3 — no triangle is hit.** Below is the roof triangle #260 that the ray passes closest to, to show why it misses

V0 = (-79.53, 10.51, 22.50), V1 = (-61.88, 21.34, 22.50), V2 = (-58.06, 43.01, 22.50)

Solve p + λs = V0 + u·E1 + v·E2 (Möller & Trumbore 1997):

| quantity | formula | value |
|---|---|---|
| E1, E2 | V1 − V0, V2 − V0 | (17.65, 10.84, 0.00), (21.48, 32.51, 0.00) |
| h | s × E2 | (-17.238, 11.388, -8.929) |
| a (determinant) | E1 · h | -180.9192  (≠ 0, so the ray is not parallel) |
| w | p − V0 | (32.669, 46.275, -3.974) |
| u | (w · h) / a | **0.0038** |
| q | w × E1 | (43.063, -70.158, -462.938) |
| v | (s · q) / a | **1.2944** |
| λ | (E2 · q) / a | **7.494 m** |

Check: u ≥ 0 → True, v ≥ 0 → True, u + v ≤ 1 → False, λ > 0 → True.

The ray reaches the roof's height (22.5 m) 7.5 m away, at (-51.67, 52.62, 22.50), but (u, v) falls outside the triangle. That spot is still in front of the roof edge (not under any roof), so the ray reaches roof height before it gets to the roof and passes over it. None of the 293 triangles in the scene satisfies all four conditions.

**Result: DIRECT SUN.**

## Case B (shaded): 15 Oct 16:30 KST

**Step 1 — sun position (pvlib).**
Elevation α = 16.44°, azimuth A = 246.61° (clockwise from north).

**Step 2 — direction from the seat towards the sun.**
s = (cos α sin A, cos α cos A, sin α)
= (0.9591×-0.9178, 0.9591×-0.3969, 0.2830) = **(-0.8803, -0.3807, 0.2830)**

**Step 3 — the first triangle the ray hits** (triangle #266, type: roof)

V0 = (-79.53, 10.51, 22.50), V1 = (-41.41, 83.81, 22.50), V2 = (-38.49, 102.06, 22.50)

Solve p + λs = V0 + u·E1 + v·E2 (Möller & Trumbore 1997):

| quantity | formula | value |
|---|---|---|
| E1, E2 | V1 − V0, V2 − V0 | (38.13, 73.31, 0.00), (41.04, 91.56, 0.00) |
| h | s × E2 | (-25.913, 11.616, -64.973) |
| a (determinant) | E1 · h | -136.4747  (≠ 0, so the ray is not parallel) |
| w | p − V0 | (32.669, 46.275, -3.974) |
| u | (w · h) / a | **0.3724** |
| q | w × E1 | (291.337, -151.526, 630.545) |
| v | (s · q) / a | **0.1489** |
| λ | (E2 · q) / a | **14.041 m** |

Check: u ≥ 0 → True, v ≥ 0 → True, u + v ≤ 1 → True, λ > 0 → True.

All four hold, so the ray hits this roof triangle 14.0 m away, at (-59.23, 51.43, 22.50). Its height is 22.50 m, which equals the roof height (22.5 m).

**Result: SHADE.**

## What this point does over the whole game (15 Oct)

On 15 Oct this point is in direct sun from 14:00 until **15:55**, then in shade until 17:00. Doing this for all 9 points of 529 gives q(t). Summing (1 − q(t)) × 5 min gives E = 107 minutes of direct sun.

## Case C (sunny): 20 Sep 15:00 KST

**Step 1 — sun position (pvlib).**
Elevation α = 40.58°, azimuth A = 235.03° (clockwise from north).

**Step 2 — direction from the seat towards the sun.**
s = (cos α sin A, cos α cos A, sin α)
= (0.7594×-0.8195, 0.7594×-0.5731, 0.6506) = **(-0.6223, -0.4353, 0.6506)**

**Step 3 — no triangle is hit.** Below is the roof triangle #260 that the ray passes closest to, to show why it misses

V0 = (-79.53, 10.51, 22.50), V1 = (-61.88, 21.34, 22.50), V2 = (-58.06, 43.01, 22.50)

Solve p + λs = V0 + u·E1 + v·E2 (Möller & Trumbore 1997):

| quantity | formula | value |
|---|---|---|
| E1, E2 | V1 − V0, V2 − V0 | (17.65, 10.84, 0.00), (21.48, 32.51, 0.00) |
| h | s × E2 | (-21.149, 13.972, -10.884) |
| a (determinant) | E1 · h | -221.9628  (≠ 0, so the ray is not parallel) |
| w | p − V0 | (32.669, 46.275, -3.974) |
| u | (w · h) / a | **0.0050** |
| q | w × E1 | (43.063, -70.158, -462.938) |
| v | (s · q) / a | **1.3400** |
| λ | (E2 · q) / a | **6.109 m** |

Check: u ≥ 0 → True, v ≥ 0 → True, u + v ≤ 1 → False, λ > 0 → True.

The ray reaches the roof's height (22.5 m) 6.1 m away, at (-50.67, 54.12, 22.50), but (u, v) falls outside the triangle. That spot is still in front of the roof edge (not under any roof), so the ray reaches roof height before it gets to the roof and passes over it. None of the 293 triangles in the scene satisfies all four conditions.

**Result: DIRECT SUN.**

## Case D (shaded): 20 Sep 16:30 KST

**Step 1 — sun position (pvlib).**
Elevation α = 24.03°, azimuth A = 253.19° (clockwise from north).

**Step 2 — direction from the seat towards the sun.**
s = (cos α sin A, cos α cos A, sin α)
= (0.9134×-0.9573, 0.9134×-0.2891, 0.4072) = **(-0.8743, -0.2641, 0.4072)**

**Step 3 — the first triangle the ray hits** (triangle #261, type: roof)

V0 = (-79.53, 10.51, 22.50), V1 = (-58.06, 43.01, 22.50), V2 = (-51.14, 63.47, 22.50)

Solve p + λs = V0 + u·E1 + v·E2 (Möller & Trumbore 1997):

| quantity | formula | value |
|---|---|---|
| E1, E2 | V1 − V0, V2 − V0 | (21.48, 32.51, 0.00), (28.39, 52.96, 0.00) |
| h | s × E2 | (-21.565, 11.560, -38.811) |
| a (determinant) | E1 · h | -87.3303  (≠ 0, so the ray is not parallel) |
| w | p − V0 | (32.669, 46.275, -3.974) |
| u | (w · h) / a | **0.1754** |
| q | w × E1 | (129.188, -85.346, 68.202) |
| v | (s · q) / a | **0.7174** |
| λ | (E2 · q) / a | **9.760 m** |

Check: u ≥ 0 → True, v ≥ 0 → True, u + v ≤ 1 → True, λ > 0 → True.

All four hold, so the ray hits this roof triangle 9.8 m away, at (-55.40, 54.20, 22.50). Its height is 22.50 m, which equals the roof height (22.5 m).

**Result: SHADE.**

## What this point does over the whole game (20 Sep)

On 20 Sep this point is in direct sun from 14:00 until **16:20**, then in shade until 17:00. Doing this for all 9 points of 529 gives q(t). Summing (1 − q(t)) × 5 min gives E = 127 minutes of direct sun.

## Cost

T × P × F = 36 time steps × 36 points × 293 triangles = 379,728 ray–triangle tests per date (all 4 sections). With 12 sections (108 points) it would be 1,139,184. That is linear in each factor, so it runs in well under a second with numpy.
