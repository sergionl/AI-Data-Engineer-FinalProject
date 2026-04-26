USE finalproject;

-- 1. Conteo total de detecciones por clase
SELECT
    class_name,
    COUNT(*) AS total_detecciones
FROM detections
GROUP BY class_name
ORDER BY total_detecciones DESC;

-- 2. Cantidad de personas detectadas por video
SELECT
    source_id,
    COUNT(*) AS total_personas
FROM detections
WHERE source_type = 'video'
  AND class_name = 'person'
GROUP BY source_id
ORDER BY total_personas DESC;

-- 3. Área promedio del bounding box por clase
SELECT
    class_name,
    AVG(area_pixels) AS area_promedio
FROM detections
GROUP BY class_name
ORDER BY area_promedio DESC;

-- 4. Colores dominantes más frecuentes por clase
SELECT
    class_name,
    dominant_color_name,
    COUNT(*) AS total
FROM detections
GROUP BY class_name, dominant_color_name
ORDER BY class_name, total DESC;

-- 5. Cantidad de objetos detectados por ventana de 10 segundos en videos
SELECT
    source_id,
    window_10s,
    COUNT(*) AS total_objetos
FROM detections
WHERE source_type = 'video'
GROUP BY source_id, window_10s
ORDER BY source_id, window_10s;