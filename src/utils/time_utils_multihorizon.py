from datetime import timedelta


def compute_forecast_horizons(radar_lead_minutes: int) -> list:
    assert radar_lead_minutes > 0, (
        f"radar_lead_minutes must be positive, got {radar_lead_minutes}"
    )
    assert radar_lead_minutes % 15 == 0, (
        f"radar_lead_minutes must be a multiple of 15, got {radar_lead_minutes}. "
        f"Valid values: 15, 30, 45, 60, 75, 90, 105, 120, ..."
    )

    horizons = list(range(15, radar_lead_minutes + 1, 15))
    return horizons


def get_satellite_history_and_radar_targets(
    reference_time,
    cadence_minutes,
    history_minutes,
    radar_lead_minutes,
):

    n_steps = history_minutes // cadence_minutes

    satellite_times = [
        reference_time - timedelta(minutes=(n_steps - 1 - i) * cadence_minutes)
        for i in range(n_steps)
    ]

    horizons = compute_forecast_horizons(radar_lead_minutes)

    radar_target_times = [
        reference_time + timedelta(minutes=h)
        for h in horizons
    ]

    return satellite_times, radar_target_times