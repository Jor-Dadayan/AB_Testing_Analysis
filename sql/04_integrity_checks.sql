-- Каждая проверка возвращает число нарушений. Ожидается 0 везде.
SELECT 'users_without_outcome' AS check_name, COUNT(*) AS violations
FROM users AS u
LEFT JOIN user_outcomes AS o ON o.user_id = u.user_id
WHERE o.user_id IS NULL
UNION ALL
SELECT 'users_missing_in_analytics', COUNT(*)
FROM users AS u
LEFT JOIN user_analytics AS a ON a.user_id = u.user_id
WHERE a.user_id IS NULL
UNION ALL
SELECT 'buyers_without_revenue', COUNT(*)
FROM user_outcomes WHERE converted AND revenue <= 0
UNION ALL
SELECT 'non_buyers_with_revenue', COUNT(*)
FROM user_outcomes WHERE NOT converted AND revenue > 0
UNION ALL
SELECT 'purchase_before_signup', COUNT(*)
FROM users AS u
JOIN user_outcomes AS o ON o.user_id = u.user_id
WHERE o.purchase_ts < u.signup_ts
UNION ALL
SELECT 'empty_experiment_groups', COUNT(*)
FROM experiment_groups AS g
LEFT JOIN users AS u ON u.group_id = g.group_id
WHERE u.user_id IS NULL;
