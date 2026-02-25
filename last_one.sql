CREATE TABLE IF NOT EXISTS last_one AS
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP FROM "202512-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP FROM "202601-divvy";