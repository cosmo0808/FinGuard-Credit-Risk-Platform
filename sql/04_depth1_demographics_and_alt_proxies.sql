-- =============================================================================
-- FinGuard Stage 1B: 04_depth1_demographics_and_alt_proxies.sql
-- Table Family: train_person_1 (primary applicant num_group1=0) + train_tax_registry_a_1 + train_debitcard_1
-- Purpose:
--   1. Extract Group 3 (Application & Demographic Data) for primary applicant
--   2. Calculate exact age at decision and employment tenure
--   3. Explicitly include sensitive attributes for subsequent Fair Lending audit
--   4. Aggregate Group 4 (Alternative / Non-Traditional Proxies):
--      - Tax registry income verification & records
--      - Debit card activity, turnover velocity, and balances
-- Scope: All 1,526,659 case_ids
-- =============================================================================

-- Aggregate Tax Registry
CREATE OR REPLACE TEMP TABLE tax_agg AS
SELECT
    case_id,
    COUNT(*) AS tax_records_cnt,
    AVG(amount_4527230A) AS tax_avg_amount,
    MAX(amount_4527230A) AS tax_max_amount
FROM read_parquet('D:/data analysis/FinGuard/data/parquet_files/train/train_tax_registry_a_1.parquet')
GROUP BY case_id;

-- Aggregate Debit Card Activity
CREATE OR REPLACE TEMP TABLE debitcard_agg AS
SELECT
    case_id,
    COUNT(*) AS debit_records_cnt,
    AVG(last180dayaveragebalance_704A) AS debit_avg_bal_180d,
    AVG(last180dayturnover_1134A) AS debit_avg_turnover_180d,
    AVG(last30dayturnover_651A) AS debit_avg_turnover_30d
FROM read_parquet('D:/data analysis/FinGuard/data/parquet_files/train/train_debitcard_1.parquet')
GROUP BY case_id;

-- Filter primary applicant in person_1 (num_group1 = 0)
CREATE OR REPLACE TEMP TABLE primary_person AS
SELECT
    case_id,
    birth_259D,
    empl_employedfrom_271D,
    incometype_1044T,
    education_927M,
    housingtype_772L,
    familystate_447L,
    childnum_185L,
    contaddr_smempladdr_334L,
    role_1084L,
    gender_992L
FROM read_parquet('D:/data analysis/FinGuard/data/parquet_files/train/train_person_1.parquet')
WHERE num_group1 = 0;

-- Combine demographics and alternative proxies
CREATE OR REPLACE TEMP TABLE demo_and_alt_features AS
SELECT
    b.case_id,

    -- Group 3: Application & Demographic Data
    CAST((TRY_CAST(b.date_decision AS DATE) - TRY_CAST(p.birth_259D AS DATE)) / 365.25 AS FLOAT) AS person_age,
    CAST((TRY_CAST(b.date_decision AS DATE) - TRY_CAST(p.empl_employedfrom_271D AS DATE)) / 365.25 AS FLOAT) AS person_employment_years,
    p.incometype_1044T AS person_income_type,
    p.education_927M AS person_education_level,
    p.housingtype_772L AS person_housing_type,
    p.familystate_447L AS person_family_status,
    CAST(p.childnum_185L AS FLOAT) AS person_children_cnt,
    CAST(p.contaddr_smempladdr_334L AS FLOAT) AS person_is_contact_same_as_empl,
    p.role_1084L AS person_role_type,

    -- Sensitive Attributes (Retained strictly for Fair Lending Audits in Stage 4/5)
    p.gender_992L AS person_gender_sensitive,
    CASE 
        WHEN (TRY_CAST(b.date_decision AS DATE) - TRY_CAST(p.birth_259D AS DATE)) / 365.25 < 25 THEN '<25'
        WHEN (TRY_CAST(b.date_decision AS DATE) - TRY_CAST(p.birth_259D AS DATE)) / 365.25 < 35 THEN '25-34'
        WHEN (TRY_CAST(b.date_decision AS DATE) - TRY_CAST(p.birth_259D AS DATE)) / 365.25 < 50 THEN '35-49'
        WHEN (TRY_CAST(b.date_decision AS DATE) - TRY_CAST(p.birth_259D AS DATE)) / 365.25 >= 50 THEN '50+'
        ELSE 'Unknown'
    END AS person_age_band_sensitive,

    -- Group 4: Alternative / Non-Traditional Data Proxies (Tax Registry)
    CAST(CASE WHEN tax.tax_records_cnt > 0 THEN 1 ELSE 0 END AS FLOAT) AS has_tax_registry_record,
    CAST(COALESCE(tax.tax_records_cnt, 0) AS FLOAT) AS tax_registry_records_cnt,
    CAST(COALESCE(tax.tax_avg_amount, 0) AS FLOAT) AS tax_registry_avg_amount,
    CAST(COALESCE(tax.tax_max_amount, 0) AS FLOAT) AS tax_registry_max_amount,
    CAST(COALESCE(tax.tax_avg_amount, 0) / NULLIF(b.maininc_215A, 0) AS FLOAT) AS tax_verified_income_ratio,

    -- Group 4: Alternative / Non-Traditional Data Proxies (Debit Card)
    CAST(CASE WHEN dc.debit_records_cnt > 0 THEN 1 ELSE 0 END AS FLOAT) AS has_debitcard,
    CAST(COALESCE(dc.debit_records_cnt, 0) AS FLOAT) AS debit_records_cnt,
    CAST(COALESCE(dc.debit_avg_bal_180d, 0) AS FLOAT) AS debit_avg_bal_180d,
    CAST(COALESCE(dc.debit_avg_turnover_180d, 0) AS FLOAT) AS debit_avg_turnover_180d,
    CAST(COALESCE(dc.debit_avg_turnover_30d, 0) AS FLOAT) AS debit_avg_turnover_30d,
    CAST(COALESCE(dc.debit_avg_turnover_30d, 0) / NULLIF(COALESCE(dc.debit_avg_turnover_180d, 0) / 6.0, 0) AS FLOAT) AS debit_turnover_velocity

FROM base_static_features b
LEFT JOIN primary_person p
    ON b.case_id = p.case_id
LEFT JOIN tax_agg tax
    ON b.case_id = tax.case_id
LEFT JOIN debitcard_agg dc
    ON b.case_id = dc.case_id;
