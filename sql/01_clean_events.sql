-- До выполнения Python проверяет конфликты: один ключ с разным содержимым.
-- DISTINCT сравнивает все поля и удаляет только точные копии.
CREATE TEMP TABLE clean_events AS
SELECT DISTINCT
    experiment_id, scenario, event_id, user_id,
    event_name, event_ts, listing_id
FROM raw_events;
