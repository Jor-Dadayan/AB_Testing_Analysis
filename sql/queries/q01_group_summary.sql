-- Задача: главное сравнение A/B. Пользователи, покупатели, конверсия, выручка по группам.
-- Конструкция: JOIN трёх нормализованных таблиц + GROUP BY. AVG от 0/1 равен доле.
-- Результат: 2 строки (control, treatment).
SELECT
    g.group_name,
    COUNT(*)                                                       AS users,
    SUM(CASE WHEN o.converted THEN 1 ELSE 0 END)                   AS buyers,
    ROUND(AVG(CASE WHEN o.converted THEN 1.0 ELSE 0.0 END), 4)     AS conversion,
    ROUND(AVG(o.revenue), 2)                                       AS revenue_per_user,
    ROUND(SUM(o.revenue)
          / NULLIF(SUM(CASE WHEN o.converted THEN 1 ELSE 0 END), 0), 2) AS avg_order_value,
    ROUND(AVG(CASE WHEN o.revenue = 0 THEN 1.0 ELSE 0.0 END), 4)   AS zero_revenue_share
FROM users AS u
JOIN experiment_groups AS g ON g.group_id = u.group_id
JOIN user_outcomes AS o     ON o.user_id  = u.user_id
GROUP BY g.group_name
ORDER BY g.group_name;
