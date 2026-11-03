-- Перенос данных из stg_users в боевые таблицы. Запускается внутри транзакции.

-- 1. Пользователи: новые вставляются, существующие обновляются (ON CONFLICT).
INSERT INTO users (user_id, group_id, signup_ts, device_type, country, traffic_source,
                   is_new_user, age, sessions_prior_30d, cart_items, cart_value)
SELECT s.user_id, g.group_id, s.signup_ts, s.device_type, s.country, s.traffic_source,
       s.is_new_user, s.age, s.sessions_prior_30d, s.cart_items,
       CAST(s.cart_value AS NUMERIC(12, 2))
FROM stg_users AS s
JOIN experiment_groups AS g ON g.group_name = s.group_name
ON CONFLICT (user_id) DO UPDATE SET
    group_id = EXCLUDED.group_id,
    signup_ts = EXCLUDED.signup_ts,
    device_type = EXCLUDED.device_type,
    country = EXCLUDED.country,
    traffic_source = EXCLUDED.traffic_source,
    is_new_user = EXCLUDED.is_new_user,
    age = EXCLUDED.age,
    sessions_prior_30d = EXCLUDED.sessions_prior_30d,
    cart_items = EXCLUDED.cart_items,
    cart_value = EXCLUDED.cart_value;

-- 2. Результаты (после users, из-за внешнего ключа).
INSERT INTO user_outcomes (user_id, converted, revenue, purchase_ts)
SELECT user_id, converted, CAST(revenue AS NUMERIC(12, 2)), purchase_ts
FROM stg_users
ON CONFLICT (user_id) DO UPDATE SET
    converted = EXCLUDED.converted,
    revenue = EXCLUDED.revenue,
    purchase_ts = EXCLUDED.purchase_ts;

-- 3. Удаляем пользователей, которых нет в новом файле (outcomes удалятся каскадом).
DELETE FROM users AS u
WHERE NOT EXISTS (SELECT 1 FROM stg_users AS s WHERE s.user_id = u.user_id);
