-- Запрос выполняется на SQLite 3. Используются стандартные конструкции SQL.
-- Входные события предварительно проверены и очищены от дублей.
-- Знаменатель основной метрики включает всех назначенных пользователей.
WITH searches AS (
    SELECT a.user_id, e.session_id, MIN(e.event_time) AS search_time
    FROM assignments a JOIN events e ON a.user_id = e.user_id
    WHERE e.event_name = 'search'
      AND e.event_time >= a.assigned_at AND e.event_time <= a.window_end
    GROUP BY a.user_id, e.session_id
), views AS (
    SELECT s.user_id, s.session_id, e.listing_id, MIN(e.event_time) AS view_time
    FROM searches s JOIN events e
      ON s.user_id = e.user_id AND s.session_id = e.session_id
    JOIN assignments a ON a.user_id = e.user_id
    WHERE e.event_name = 'listing_view' AND e.event_time >= s.search_time
      AND e.event_time <= a.window_end
    GROUP BY s.user_id, s.session_id, e.listing_id
), ordered_contacts AS (
    SELECT DISTINCT v.user_id
    FROM views v JOIN events e ON v.user_id = e.user_id
      AND v.session_id = e.session_id AND v.listing_id = e.listing_id
    JOIN assignments a ON a.user_id = e.user_id
    WHERE e.event_name = 'contact' AND e.event_time >= v.view_time
      AND e.event_time <= a.window_end
), outcomes AS (
    SELECT a.user_id,
           MAX(CASE WHEN e.event_name = 'contact' THEN 1 ELSE 0 END) AS contacted,
           MAX(CASE WHEN e.event_name = 'complaint' THEN 1 ELSE 0 END) AS complained
    FROM assignments a LEFT JOIN events e ON a.user_id = e.user_id
      AND e.event_time >= a.assigned_at AND e.event_time <= a.window_end
    GROUP BY a.user_id
)
SELECT a.*, o.contacted, o.complained,
       CASE WHEN s.user_id IS NULL THEN 0 ELSE 1 END AS searched,
       CASE WHEN v.user_id IS NULL THEN 0 ELSE 1 END AS viewed,
       CASE WHEN c.user_id IS NULL THEN 0 ELSE 1 END AS funnel_contacted,
       SUM(o.contacted) OVER (PARTITION BY a.variant ORDER BY a.assigned_at, a.user_id
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_contacts
FROM assignments a JOIN outcomes o ON a.user_id = o.user_id
-- GROUP BY позволяет SQLite использовать индексы для промежуточных агрегатов.
-- Подзапросы с DISTINCT приводили к вложенным сканированиям в проверенном плане.
LEFT JOIN (SELECT user_id FROM searches GROUP BY user_id) s ON a.user_id = s.user_id
LEFT JOIN (SELECT user_id FROM views GROUP BY user_id) v ON a.user_id = v.user_id
LEFT JOIN ordered_contacts c ON a.user_id = c.user_id;
