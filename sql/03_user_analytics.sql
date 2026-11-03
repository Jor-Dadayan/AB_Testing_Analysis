-- Аналитическая таблица: ровно одна строка на пользователя.
DROP TABLE IF EXISTS user_analytics;

CREATE TABLE user_analytics AS
SELECT
    u.user_id,
    g.group_name,
    u.signup_ts,
    CAST(u.signup_ts AS DATE)                                  AS signup_date,
    CAST(EXTRACT(HOUR FROM u.signup_ts) AS INTEGER)            AS signup_hour,
    CAST(EXTRACT(ISODOW FROM u.signup_ts) AS INTEGER) - 1      AS day_of_week,
    CASE WHEN EXTRACT(ISODOW FROM u.signup_ts) >= 6 THEN 1 ELSE 0 END AS is_weekend,
    u.device_type,
    u.country,
    u.traffic_source,
    CASE WHEN u.is_new_user THEN 1 ELSE 0 END                  AS is_new_user,
    u.age,
    CASE WHEN u.age < 25 THEN '18-24'
         WHEN u.age < 35 THEN '25-34'
         WHEN u.age < 45 THEN '35-44'
         WHEN u.age < 55 THEN '45-54'
         ELSE '55+' END                                        AS age_group,
    u.sessions_prior_30d,
    u.cart_items,
    u.cart_value,
    CASE WHEN u.cart_value < 30 THEN 'low'
         WHEN u.cart_value < 80 THEN 'mid'
         ELSE 'high' END                                       AS cart_segment,
    CASE WHEN o.converted THEN 1 ELSE 0 END                    AS converted,
    o.revenue,
    o.purchase_ts,
    ROUND(CAST(EXTRACT(EPOCH FROM (o.purchase_ts - u.signup_ts)) / 3600.0
               AS NUMERIC), 2)                                 AS hours_to_purchase
FROM users AS u
JOIN experiment_groups AS g ON g.group_id = u.group_id
JOIN user_outcomes AS o     ON o.user_id  = u.user_id;

ALTER TABLE user_analytics ADD PRIMARY KEY (user_id);
CREATE INDEX ix_user_analytics_group ON user_analytics (group_name);
ANALYZE user_analytics;
