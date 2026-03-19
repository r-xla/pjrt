# Situation Report

Get a situation report on the pjrt package installation and
configuration. This function checks system information, package
versions, plugin availability, and device status to help diagnose
configuration issues.

## Usage

``` r
pjrt_sitrep()
```

## Value

Invisibly returns `NULL`. Called for its side effect of printing a
diagnostic report.

## Examples

``` r
pjrt_sitrep()
#> 
#> ── pjrt situation report ───────────────────────────────────────────────────────
#> 
#> ── System information ──
#> 
#> • OS: linux
#> • arch: amd64
#> • R version: 4.5.3
#> 
#> ── Plugins ──
#> 
#> ✔ "cpu": plugin found
#> 
#> ── Environment variables ──
#> 
#> ℹ No pjrt-related environment variables are set.
```
