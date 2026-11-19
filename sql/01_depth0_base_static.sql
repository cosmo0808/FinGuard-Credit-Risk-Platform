-- =============================================================================
-- FinGuard Stage 1B: 01_depth0_base_static.sql
-- Table Family: train_base + train_static_0_* + train_static_cb_0
-- Purpose:
--   1. Extract base keys: case_id, date_decision, MONTH, WEEK_NUM, target, split_name
--   2. Extract Core Affordability raw terms: credamount_770A, annuity_780A, maininc_215A
--   3. Apply zero-inflation guards: NULLIF(maininc_215A, 0)
--   4. Calculate Base PTI = annuity_780A / NULLIF(maininc_215A, 0) and Loan-to-Income
--   5. Extract external score & credit bureau query velocity features
-- Scope: All 1,526,659 case_ids
-- =============================================================================

CREATE OR REPLACE TEMP TABLE base_static_features AS
SELECT
    -- Base identifiers and temporal split tags
    b.case_id,
    b.date_decision,
    b.MONTH,
    b.WEEK_NUM,
    b.target,
    CASE 
        WHEN b.WEEK_NUM <= 64             THEN 'Train_Dev'
        WHEN b.WEEK_NUM BETWEEN 65 AND 77 THEN 'OOT_Valid'
        WHEN b.WEEK_NUM BETWEEN 78 AND 91 THEN 'OOT_Monitor'
        ELSE                                   'Out_of_Range'
    END AS split_name,

    -- Core Affordability & Loan Terms
    CAST(s.credamount_770A AS FLOAT) AS credamount_770A,
    CAST(s.annuity_780A AS FLOAT) AS annuity_780A,
    CAST(s.maininc_215A AS FLOAT) AS maininc_215A,
    CAST(s.annuitynextmonth_57A AS FLOAT) AS annuitynextmonth_57A,
    CAST(s.disbursedcredamount_1113A AS FLOAT) AS disbursedcredamount_1113A,

    -- Ratios guarded against zero-inflation & null denominators
    CAST(s.annuity_780A / NULLIF(s.maininc_215A, 0) AS FLOAT) AS pti,
    CAST(s.credamount_770A / NULLIF(s.maininc_215A, 0) AS FLOAT) AS loan_to_income,

    -- Existing debt & payment capacity (from static depth 0)
    CAST(s.currdebt_22A AS FLOAT) AS static_currdebt,
    CAST(s.currdebtcredtyperange_828A AS FLOAT) AS static_currdebt_by_credtype,
    CAST(s.totaldebt_9A AS FLOAT) AS static_totaldebt,
    CAST(s.lastapprcredamount_781A AS FLOAT) AS static_last_approved_amount,
    CAST(s.lastrejectcredamount_222A AS FLOAT) AS static_last_rejected_amount,

    -- Credit Bureau Query Counts & Velocity (from static_cb_0)
    CAST(scb.days30_165L AS FLOAT) AS bureau_queries_30d,
    CAST(scb.days90_310L AS FLOAT) AS bureau_queries_90d,
    CAST(scb.days120_123L AS FLOAT) AS bureau_queries_120d,
    CAST(scb.days180_256L AS FLOAT) AS bureau_queries_180d,
    CAST(scb.days360_512L AS FLOAT) AS bureau_queries_360d,
    CAST(scb.numberofqueries_373L AS FLOAT) AS bureau_total_queries,
    CAST(scb.days30_165L / NULLIF(scb.days360_512L / 12.0, 0) AS FLOAT) AS bureau_recent_query_velocity,

    -- Static Repayment & DPD History (Group 2 from depth-0)
    CAST(s.actualdpdtolerance_344P AS FLOAT) AS static_dpd_with_tolerance,
    CAST(s.avgdbddpdlast24m_3658932P AS FLOAT) AS static_avg_dbd_dpd_last24m,
    CAST(s.avgdbddpdlast3m_4187120P AS FLOAT) AS static_avg_dbd_dpd_last3m,
    CAST(s.avgdbddpdlast3m_4187120P - s.avgdbddpdlast24m_3658932P AS FLOAT) AS static_dpd_short_vs_long_trend,
    CAST(s.avgdpdtolclosure24_3658938P AS FLOAT) AS static_avg_dpd_tol_closure24m,
    CAST(s.avgmaxdpdlast9m_3716943P AS FLOAT) AS static_avg_max_dpd_last9m,
    CAST(s.daysoverduetolerancedd_3976961L AS FLOAT) AS static_days_overdue_tol,
    CAST(s.avginstallast24m_3658937A AS FLOAT) AS static_avg_instal_last24m,
    CAST(s.maxinstallast24m_3658928A AS FLOAT) AS static_max_instal_last24m,
    CAST(s.totinstallast1m_4525188A AS FLOAT) AS static_tot_instal_last1m,
    CAST(s.numinstpaidearly_338L AS FLOAT) AS static_num_inst_paid_early,
    CAST(s.numinstpaidlate1d_3546852L AS FLOAT) AS static_num_inst_paid_late_1d,
    CAST(s.pctinstlsallpaidlat10d_839L AS FLOAT) AS static_pct_inst_paid_late_10d,
    CAST(s.pctinstlsallpaidearl3d_427L AS FLOAT) AS static_pct_inst_paid_early_3d,
    CAST(s.cntincpaycont9m_3716944L AS FLOAT) AS static_cnt_incoming_pmts_9m,
    CAST(s.cntpmts24_3658933L AS FLOAT) AS static_cnt_pmts_months_24m,
    CAST(s.avgoutstandbalancel6m_4187114A AS FLOAT) AS static_avg_outstand_balance_6m,
    CAST(s.maxoutstandbalancel12m_4187113A AS FLOAT) AS static_max_outstand_balance_12m,

    -- Contact & Application Stability Proxies (Group 4 from depth-0)
    CAST(s.applications30d_658L AS FLOAT) AS app_cnt_last_30d,
    CAST(s.applicationcnt_361L AS FLOAT) AS app_cnt_same_email,
    CAST(s.applicationscnt_1086L AS FLOAT) AS app_cnt_same_phone,
    CAST(s.applicationscnt_867L AS FLOAT) AS app_cnt_same_mobile,
    CAST(s.applicationscnt_629L AS FLOAT) AS app_cnt_same_employer_7d,
    CAST(s.applicationscnt_464L AS FLOAT) AS app_cnt_same_employer_phone_30d,
    CAST(s.clientscnt_1022L AS FLOAT) AS clients_sharing_mobile_cnt,
    CAST(s.clientscnt_304L AS FLOAT) AS clients_sharing_phone_cnt,
    CAST(s.clientscnt_100L AS FLOAT) AS clients_matching_empl_and_client_phone,
    CAST(s.clientscnt12m_3712952L AS FLOAT) AS clients_same_mobile_12m

FROM read_parquet('D:/data analysis/FinGuard/data/parquet_files/train/train_base.parquet') b
LEFT JOIN read_parquet('D:/data analysis/FinGuard/data/parquet_files/train/train_static_0_*.parquet') s
    ON b.case_id = s.case_id
LEFT JOIN read_parquet('D:/data analysis/FinGuard/data/parquet_files/train/train_static_cb_0.parquet') scb
    ON b.case_id = scb.case_id;
