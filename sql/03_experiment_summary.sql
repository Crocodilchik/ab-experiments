-- Одна строка user_metrics уже представляет одного назначенного участника.
-- Сумма флага 0/1 считает людей с результатом, а не количество событий.
CREATE TEMP TABLE experiment_summary AS
WITH counts AS (
    SELECT
        experiment_id, scenario, variant,
        COUNT(*) AS n_users,
        CAST(SUM(has_listing_view) AS BIGINT) AS n_listing_view,
        CAST(SUM(has_contact) AS BIGINT) AS n_contact,
        CAST(SUM(has_error) AS BIGINT) AS n_error
    FROM user_metrics
    GROUP BY experiment_id, scenario, variant
)
SELECT
    experiment_id, scenario, variant,
    n_users, n_listing_view, n_contact, n_error,
    -- DOUBLE сохраняет доли; NULLIF задаёт NULL при нулевом знаменателе.
    CAST(n_listing_view AS DOUBLE) / NULLIF(n_users, 0) AS view_cr,
    CAST(n_contact AS DOUBLE) / NULLIF(n_users, 0) AS contact_cr,
    CAST(n_error AS DOUBLE) / NULLIF(n_users, 0) AS error_rate,
    CAST(n_contact AS DOUBLE) / NULLIF(n_listing_view, 0) AS contact_given_view_rate
FROM counts
ORDER BY experiment_id, scenario, variant;
