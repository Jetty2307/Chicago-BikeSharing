WITH incoming AS (
SELECT ride_id::TEXT AS ride_id, rideable_type::TEXT AS rideable_type, started_at::TIMESTAMP AS started_at, 0 AS source_priority FROM "202512-divvy"
UNION ALL
SELECT ride_id::TEXT AS ride_id, rideable_type::TEXT AS rideable_type, started_at::TIMESTAMP AS started_at, 1 AS source_priority FROM "202601-divvy"
), deduplicated AS (
    SELECT DISTINCT ON (ride_id) ride_id, rideable_type, started_at
    FROM incoming
    ORDER BY ride_id, source_priority DESC,
             started_at DESC NULLS LAST, rideable_type DESC NULLS LAST
)
INSERT INTO public.merged AS target (ride_id, rideable_type, started_at)
SELECT ride_id, rideable_type, started_at FROM deduplicated WHERE TRUE
ON CONFLICT (ride_id) DO UPDATE
SET rideable_type = EXCLUDED.rideable_type,
    started_at = EXCLUDED.started_at
WHERE (target.rideable_type, target.started_at)
      IS DISTINCT FROM (EXCLUDED.rideable_type, EXCLUDED.started_at);
