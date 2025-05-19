CREATE TABLE IF NOT EXISTS merged AS
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202011-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202501-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202208-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202205-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202310-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202109-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202104-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202404-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202409-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202303-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202407-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202107-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202206-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202502-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202012-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202004-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202009-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202210-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202305-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202308-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202111-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202401-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202101-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202411-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202412-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202102-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202402-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202112-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202306-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202203-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202007-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202403-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202307-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202103-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202212-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202006-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202202-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202201-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202005-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202211-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202008-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202410-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202304-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202110-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202309-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202106-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202312-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202302-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202406-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202207-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202503-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202010-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202204-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202405-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202301-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202408-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202108-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202311-divvy"
UNION ALL
SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP, ended_at::TIMESTAMP, start_station_name::TEXT, start_station_id::TEXT, end_station_name::TEXT, end_station_id::TEXT, start_lat::DOUBLE PRECISION, start_lng::DOUBLE PRECISION, end_lat::DOUBLE PRECISION, end_lng::DOUBLE PRECISION, member_casual::TEXT FROM "202105-divvy";