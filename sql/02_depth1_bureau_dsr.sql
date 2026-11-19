-- =============================================================================
-- FinGuard Stage 1B: 02_depth1_bureau_dsr.sql
-- Table Family: train_credit_bureau_a_1_*
-- Purpose:
--   1. Aggregate credit bureau contract history strictly at case_id level
--   2. Distinguish active vs. closed contracts based on verified column semantics:
--      - Active: monthlyinstlamount_332A, outstandingamount_362A, totaloutstanddebtvalue_39A, dateofcredend_289D
--      - Closed: monthlyinstlamount_674A, outstandingamount_354A, totaloutstanddebtvalue_668A, dateofcredend_353D
--   3. Compute existing_bureau_monthly_debt (ACTIVE contracts ONLY)
--   4. Calculate Total_DSR_Proxy = (COALESCE(active_bureau_monthly_debt, 0) + annuity_780A) / NULLIF(maininc_215A, 0)
--   5. Engineer Group 1 features: contract counts, active debt stock, overdue amounts, max DPD
-- Scope: All 1,526,659 case_ids
-- =============================================================================

CREATE OR REPLACE TEMP TABLE bureau_agg AS
SELECT
    case_id,

    -- Contract counts
    COUNT(*) AS bureau_total_contracts_cnt,
    COUNT(CASE 
        WHEN monthlyinstlamount_332A IS NOT NULL 
          OR outstandingamount_362A > 0 
          OR totaloutstanddebtvalue_39A > 0 
          OR dateofcredend_289D IS NOT NULL 
        THEN 1 
    END) AS bureau_active_contracts_cnt,

    COUNT(CASE 
        WHEN monthlyinstlamount_674A IS NOT NULL 
          OR outstandingamount_354A IS NOT NULL 
          OR totaloutstanddebtvalue_668A IS NOT NULL 
          OR dateofcredend_353D IS NOT NULL 
        THEN 1 
    END) AS bureau_closed_contracts_cnt,

    -- Active Monthly Debt Commitment (Strictly for Total DSR)
    SUM(CASE 
        WHEN monthlyinstlamount_332A IS NOT NULL AND monthlyinstlamount_332A > 0 
        THEN monthlyinstlamount_332A 
        ELSE 0 
    END) AS active_bureau_monthly_debt,

    -- Active Debt Stock & Outstanding Balances
    SUM(COALESCE(totaloutstanddebtvalue_39A, outstandingamount_362A, 0)) AS bureau_active_total_debt,
    MAX(COALESCE(totaloutstanddebtvalue_39A, outstandingamount_362A, 0)) AS bureau_active_max_debt,
    SUM(COALESCE(totalamount_996A, 0)) AS bureau_active_total_credit_amt,

    -- Closed Debt Stock & Totals
    SUM(COALESCE(totaloutstanddebtvalue_668A, outstandingamount_354A, 0)) AS bureau_closed_total_debt,
    SUM(COALESCE(totalamount_6A, 0)) AS bureau_closed_total_credit_amt,

    -- Bureau Delinquency & DPD (Active vs. Closed)
    COALESCE(MAX(dpdmax_139P), 0) AS bureau_active_max_dpd,
    COALESCE(MAX(dpdmax_757P), 0) AS bureau_closed_max_dpd,
    SUM(COALESCE(overdueamount_659A, 0)) AS bureau_active_overdue_amt,
    MAX(COALESCE(overdueamountmax_155A, 0)) AS bureau_active_max_overdue_amt,
    MAX(COALESCE(numberofoverdueinstls_725L, 0)) AS bureau_active_overdue_instls_max,
    MAX(COALESCE(prolongationcount_599L, 0)) AS bureau_active_prolongation_cnt

FROM read_parquet('D:/data analysis/FinGuard/data/parquet_files/train/train_credit_bureau_a_1_*.parquet')
GROUP BY case_id;

-- Integrate with base affordability fields for Total DSR computation
CREATE OR REPLACE TEMP TABLE bureau_features AS
SELECT
    b.case_id,

    -- Contract counts and structure
    CAST(COALESCE(agg.bureau_total_contracts_cnt, 0) AS FLOAT) AS bureau_total_contracts_cnt,
    CAST(COALESCE(agg.bureau_active_contracts_cnt, 0) AS FLOAT) AS bureau_active_contracts_cnt,
    CAST(COALESCE(agg.bureau_closed_contracts_cnt, 0) AS FLOAT) AS bureau_closed_contracts_cnt,
    CAST(COALESCE(agg.bureau_active_contracts_cnt, 0)::FLOAT / NULLIF(COALESCE(agg.bureau_total_contracts_cnt, 0), 0) AS FLOAT) AS bureau_active_contract_ratio,

    -- Core DSR & Affordability Features
    CAST(COALESCE(agg.active_bureau_monthly_debt, 0) AS FLOAT) AS active_bureau_monthly_debt,
    CAST(CASE WHEN COALESCE(agg.active_bureau_monthly_debt, 0) > 0 THEN 1 ELSE 0 END AS FLOAT) AS has_active_bureau_debt,
    
    -- Total DSR Proxy: (Active Bureau Monthly Debt + New Annuity) / Income
    CAST((COALESCE(agg.active_bureau_monthly_debt, 0) + b.annuity_780A) / NULLIF(b.maininc_215A, 0) AS FLOAT) AS total_dsr_proxy,

    -- Total Debt Burden to Income: (Active Bureau Debt Stock + New Credit Amount) / Income
    CAST(COALESCE(agg.bureau_active_total_debt, 0) AS FLOAT) AS bureau_active_total_debt,
    CAST(COALESCE(agg.bureau_active_max_debt, 0) AS FLOAT) AS bureau_active_max_debt,
    CAST(COALESCE(agg.bureau_closed_total_debt, 0) AS FLOAT) AS bureau_closed_total_debt,
    CAST((COALESCE(agg.bureau_active_total_debt, 0) + b.credamount_770A) / NULLIF(b.maininc_215A, 0) AS FLOAT) AS total_debt_to_income,

    -- Bureau Delinquency features
    CAST(COALESCE(agg.bureau_active_max_dpd, 0) AS FLOAT) AS bureau_active_max_dpd,
    CAST(COALESCE(agg.bureau_closed_max_dpd, 0) AS FLOAT) AS bureau_closed_max_dpd,
    CAST(COALESCE(agg.bureau_active_overdue_amt, 0) AS FLOAT) AS bureau_active_overdue_amt,
    CAST(COALESCE(agg.bureau_active_max_overdue_amt, 0) AS FLOAT) AS bureau_active_max_overdue_amt,
    CAST(COALESCE(agg.bureau_active_overdue_instls_max, 0) AS FLOAT) AS bureau_active_overdue_instls_max,
    CAST(COALESCE(agg.bureau_active_prolongation_cnt, 0) AS FLOAT) AS bureau_active_prolongation_cnt

FROM base_static_features b
LEFT JOIN bureau_agg agg
    ON b.case_id = agg.case_id;
