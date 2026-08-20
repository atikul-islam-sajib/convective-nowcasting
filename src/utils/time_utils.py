from datetime import timedelta

def get_satellite_history_and_radar_target(
    reference_time,
    cadence_minutes,
    history_minutes,
    lead_minutes,
):
    n_steps = history_minutes // cadence_minutes  # 6

    satellite_times = [
        reference_time - timedelta(minutes=(n_steps - 1 - i) * cadence_minutes)
        for i in range(n_steps)
    ]

    radar_target_time = reference_time + timedelta(minutes=lead_minutes)

    return satellite_times, radar_target_time

