-- День относится к назначению участника в UTC, даже если контакт был завтра.
-- Сохраняем только наблюдаемые группы; пустая витрина даст пустую таблицу.
CREATE TEMP TABLE daily_summary AS
WITH counts AS (
    SELECT
        experiment_id, scenario, variant, assigned_date,
        COUNT(*) AS n_users,
        CAST(SUM(has_listing_view) AS BIGINT) AS n_listing_view,
        CAST(SUM(has_contact) AS BIGINT) AS n_contact,
        CAST(SUM(has_error) AS BIGINT) AS n_error
    FROM user_metrics
    GROUP BY experiment_id, scenario, variant, assigned_date
)
SELECT
    experiment_id, scenario, variant, assigned_date,
    n_users, n_listing_view, n_contact, n_error,
    CAST(n_listing_view AS DOUBLE) / NULLIF(n_users, 0) AS view_cr,
    CAST(n_contact AS DOUBLE) / NULLIF(n_users, 0) AS contact_cr,
    CAST(n_error AS DOUBLE) / NULLIF(n_users, 0) AS error_rate,
    CAST(n_contact AS DOUBLE) / NULLIF(n_listing_view, 0) AS contact_given_view_rate
FROM counts
ORDER BY experiment_id, scenario, variant, assigned_date;
