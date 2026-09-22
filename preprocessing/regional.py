# preprocessing/regional.py

import pandas as pd


def aggregate_region(
    cir_by_reach: dict,
    volume_by_reach: dict,
    reach_ids: list,
    region_id: str = "SWP_ALL",
):
    """Collapse several reaches' per-reach CIR/volume series into one
    regional-total series, for forecasting the SWP region's cumulated
    (summed-across-reaches) sediment volume directly, rather than each
    reach individually.

    Regional CIR is the SUM of the reaches' own `CIR_studentt_daily` on
    dates common to all of them (summing rates is equivalent to
    differencing the summed volume). Regional volume is the SUM of the
    reaches' `volume_studentt_daily` on those same dates -- used only as
    the evaluation/back-calculation reference, never as a model input.

    Only dates where EVERY listed reach has a value are kept, so the sum is
    always a true region-wide total, never a partial one growing/shrinking
    as reaches drop in and out.

    Returns
    -------
    cir_by_region : dict[str, pd.Series]   -- {region_id: regional CIR series}
    volume_by_region : dict[str, pd.Series] -- {region_id: regional volume series}
    region_ids : list[str]                  -- [region_id]
    """
    cir_df = pd.concat(
        [cir_by_reach[r].rename(r) for r in reach_ids], axis=1, join="inner"
    ).dropna()
    volume_df = pd.concat(
        [volume_by_reach[r].rename(r) for r in reach_ids], axis=1, join="inner"
    ).dropna()

    regional_cir = cir_df.sum(axis=1).rename(region_id)
    regional_volume = volume_df.sum(axis=1).rename(region_id)

    print(
        f"Aggregated {len(reach_ids)} reaches into region '{region_id}': "
        f"{len(regional_cir)} common CIR days, {len(regional_volume)} common volume days "
        f"({regional_cir.index.min().date()} to {regional_cir.index.max().date()})"
    )

    return (
        {region_id: regional_cir},
        {region_id: regional_volume},
        [region_id],
    )
