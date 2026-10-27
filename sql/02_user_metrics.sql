-- Начинаем с назначений: отсутствие событий не должно удалять участника.
-- Окно стоит в ON у LEFT JOIN, а не в WHERE после соединения.
CREATE TEMP TABLE user_metrics AS
WITH participant_events AS (
    SELECT
        a.experiment_id,
        a.scenario,
        a.user_id,
        a.variant,
        a.assigned_at,
        CAST(a.assigned_at AT TIME ZONE 'UTC' AS DATE) AS assigned_date,
        u.platform,
        u.region,
        u.is_new_user,
        e.event_name
    FROM assignments AS a
    CROSS JOIN run_settings AS r
    JOIN users AS u ON u.user_id = a.user_id
    LEFT JOIN clean_events AS e
        ON e.experiment_id = a.experiment_id
        AND e.scenario = a.scenario
        AND e.user_id = a.user_id
        AND e.event_ts >= a.assigned_at
        AND e.event_ts < a.assigned_at + r.window_hours * INTERVAL '1 hour'
)
SELECT
    experiment_id, scenario, user_id, variant, assigned_at, assigned_date,
    platform, region, is_new_user,
    MAX(CASE WHEN event_name = 'listing_view' THEN 1 ELSE 0 END) AS has_listing_view,
    MAX(CASE WHEN event_name = 'contact_created' THEN 1 ELSE 0 END) AS has_contact,
    MAX(CASE WHEN event_name = 'technical_error' THEN 1 ELSE 0 END) AS has_error
FROM participant_events
GROUP BY
    experiment_id, scenario, user_id, variant, assigned_at, assigned_date,
    platform, region, is_new_user
ORDER BY experiment_id, scenario, user_id;
