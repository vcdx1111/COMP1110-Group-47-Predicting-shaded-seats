"""Shade model for day games at Gwangju-Kia Champions Field (COMP1110 Group 47).

Modules
    config      reads data/config.yaml
    geometry    satellite coordinates -> local metres, sample grids, ear clipping
    sun         sun position (pvlib) and direction vectors
    raytrace    Moller-Trumbore ray-triangle tests (scalar and vectorised)
    model       builds the triangle scene and runs the shade test
    metrics     exposure, ranks, sensitivity, rank stability, resolution checks
    baseline    the simple "under the roof" rule
    validation  comparison with labelled broadcast frames
"""
