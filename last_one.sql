CREATE TABLE IF NOT EXISTS last_one AS
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP FROM "202504-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP FROM "202505-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP FROM "202506-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP FROM "202507-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP FROM "202508-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP FROM "202509-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP FROM "202510-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP FROM "202511-divvy";