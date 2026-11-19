-- =============================================================================
-- FinGuard Stage 1B: 03_depth1_2_repayment_behavior.sql
-- Table Family: train_applprev_1_* + train_applprev_2
-- Purpose:
--   1. Aggregate internal loan history & repayment behavior (Group 2)
--   2. Calculate previous approval, rejection, and cancellation counts & rates
--   3. Extract historical internal DPD metrics (max, mean, tolerance)
--   4. Aggregate previous loan terms: amounts, annuities, tenors, payments
--   5. Integrate depth-2 credit card blocking & status events from applprev_2
-- Scope: All 1,526,659 case_ids
-- =============================================================================

CREATE OR REPLACE TEMP TABLE applprev_1_agg AS
SELECT
    case_id,

    -- Application counts & outcomes
    COUNT(*) AS prev_appl_total_cnt,
    COUNT(CASE WHEN status_219L = 'A' THEN 1 END) AS prev_appl_approved_cnt,
    COUNT(CASE WHEN status_219L IN ('D', 'K', 'R') THEN 1 END) AS prev_appl_rejected_cnt,
    COUNT(CASE WHEN status_219L = 'T' THEN 1 END) AS prev_appl_canceled_cnt,

    -- DPD & Delinquency in previous applications
    COALESCE(MAX(actualdpd_943P), 0) AS prev_appl_max_dpd,
    AVG(actualdpd_943P) AS prev_appl_mean_dpd,
    COALESCE(MAX(maxdpdtolerance_577P), 0) AS prev_appl_max_dpd_tolerance,

    -- Financial amounts & commitments
    SUM(CASE WHEN status_219L = 'A' THEN credamount_590A ELSE 0 END) AS prev_appl_total_approved_amount,
    MAX(credamount_590A) AS prev_appl_max_credamount,
    SUM(annuity_853A) AS prev_appl_total_annuity,
    MAX(annuity_853A) AS prev_appl_max_annuity,
    MAX(downpmt_134A) AS prev_appl_max_downpayment,
    SUM(pmtnum_8L) AS prev_appl_total_pmt_count,
    AVG(tenor_203L) AS prev_appl_avg_tenor,
    SUM(COALESCE(outstandingdebt_522A, currdebt_94A, 0)) AS prev_appl_outstanding_debt

FROM read_parquet('D:/data analysis/FinGuard/data/parquet_files/train/train_applprev_1_*.parquet')
GROUP BY case_id;

-- Aggregate depth-2 card blocking & status
CREATE OR REPLACE TEMP TABLE applprev_2_agg AS
SELECT
    case_id,
    COUNT(*) AS prev_card_records_cnt,
    COUNT(CASE WHEN cacccardblochreas_147M IS NOT NULL THEN 1 END) AS prev_card_blocked_cnt,
    COUNT(DISTINCT credacc_cards_status_52L) AS prev_card_distinct_status_cnt
FROM read_parquet('D:/data analysis/FinGuard/data/parquet_files/train/train_applprev_2.parquet')
GROUP BY case_id;

-- Combine into repayment features table
CREATE OR REPLACE TEMP TABLE repayment_features AS
SELECT
    b.case_id,

    -- Previous application volume and decision rates
    CAST(COALESCE(ap1.prev_appl_total_cnt, 0) AS FLOAT) AS prev_appl_total_cnt,
    CAST(COALESCE(ap1.prev_appl_approved_cnt, 0) AS FLOAT) AS prev_appl_approved_cnt,
    CAST(COALESCE(ap1.prev_appl_rejected_cnt, 0) AS FLOAT) AS prev_appl_rejected_cnt,
    CAST(COALESCE(ap1.prev_appl_approved_cnt, 0)::FLOAT / NULLIF(COALESCE(ap1.prev_appl_total_cnt, 0), 0) AS FLOAT) AS prev_appl_approval_rate,
    CAST(COALESCE(ap1.prev_appl_rejected_cnt, 0)::FLOAT / NULLIF(COALESCE(ap1.prev_appl_total_cnt, 0), 0) AS FLOAT) AS prev_appl_rejection_rate,

    -- Previous DPD behavior
    CAST(COALESCE(ap1.prev_appl_max_dpd, 0) AS FLOAT) AS prev_appl_max_dpd,
    CAST(COALESCE(ap1.prev_appl_mean_dpd, 0) AS FLOAT) AS prev_appl_mean_dpd,
    CAST(COALESCE(ap1.prev_appl_max_dpd_tolerance, 0) AS FLOAT) AS prev_appl_max_dpd_tolerance,

    -- Previous financial commitments
    CAST(COALESCE(ap1.prev_appl_total_approved_amount, 0) AS FLOAT) AS prev_appl_total_approved_amount,
    CAST(COALESCE(ap1.prev_appl_max_credamount, 0) AS FLOAT) AS prev_appl_max_credamount,
    CAST(COALESCE(ap1.prev_appl_total_annuity, 0) AS FLOAT) AS prev_appl_total_annuity,
    CAST(COALESCE(ap1.prev_appl_max_annuity, 0) AS FLOAT) AS prev_appl_max_annuity,
    CAST(COALESCE(ap1.prev_appl_max_downpayment, 0) AS FLOAT) AS prev_appl_max_downpayment,
    CAST(COALESCE(ap1.prev_appl_total_pmt_count, 0) AS FLOAT) AS prev_appl_total_pmt_count,
    CAST(COALESCE(ap1.prev_appl_avg_tenor, 0) AS FLOAT) AS prev_appl_avg_tenor,
    CAST(COALESCE(ap1.prev_appl_outstanding_debt, 0) AS FLOAT) AS prev_appl_outstanding_debt,

    -- Depth-2 Card behaviors
    CAST(COALESCE(ap2.prev_card_records_cnt, 0) AS FLOAT) AS prev_card_records_cnt,
    CAST(COALESCE(ap2.prev_card_blocked_cnt, 0) AS FLOAT) AS prev_card_blocked_cnt

FROM base_static_features b
LEFT JOIN applprev_1_agg ap1
    ON b.case_id = ap1.case_id
LEFT JOIN applprev_2_agg ap2
    ON b.case_id = ap2.case_id;
