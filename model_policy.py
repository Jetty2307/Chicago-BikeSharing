"""Model promotion based on comparable validation scores and data freshness."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


REFRESH_MONTHS = 6
MAX_REFRESH_DEGRADATION = 0.10


@dataclass(frozen=True)
class PromotionDecision:
    passed: bool
    reason: str
    refresh_required: bool


def decide_promotion(candidate_rmse, baseline_rmse, candidate_data_end,
                     baseline_data_end: Optional[str], refresh_months=REFRESH_MONTHS,
                     max_refresh_degradation=MAX_REFRESH_DEGRADATION):
    if not np.isfinite(candidate_rmse) or candidate_rmse < 0:
        return PromotionDecision(False, 'invalid_candidate_metric', False)
    if baseline_rmse is None:
        return PromotionDecision(True, 'first_model', False)
    if not np.isfinite(baseline_rmse) or baseline_rmse < 0:
        raise ValueError('Invalid baseline RMSE')
    refresh = bool(baseline_data_end and pd.Timestamp(candidate_data_end) >=
                   pd.Timestamp(baseline_data_end) + pd.DateOffset(months=refresh_months))
    if candidate_rmse <= baseline_rmse:
        return PromotionDecision(True, 'quality', refresh)
    if refresh and candidate_rmse <= baseline_rmse * (1 + max_refresh_degradation):
        return PromotionDecision(True, 'freshness', True)
    return PromotionDecision(False, 'refresh_quality_floor' if refresh else 'quality', refresh)
