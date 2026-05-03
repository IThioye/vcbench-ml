import json
import ast
import pandas as pd
import numpy as np
from typing import Any

# ─────────────────────────────────────────────
# Ordinal mappings
# ─────────────────────────────────────────────

COMPANY_SIZE_ORDER = {
    "": 0,
    "1-10 employees": 1,
    "11-50 employees": 2,
    "51-200 employees": 3,
    "201-500 employees": 4,
    "501-1000 employees": 5,
    "1001-5000 employees": 6,
    "5000+ employees": 7,
}

VALUATION_ORDER = {
    "": 0,
    "undisclosed": 0,
    "<50m": 1,
    "50m - 150m": 2,
    "150m - 500m": 3,
    ">500m": 4,
}

AMOUNT_RAISED_ORDER = {
    "": 0,
    "undisclosed": 0,
    "<10m": 1,
    "10m - 50m": 2,
    "50m - 150m": 3,
    "150m - 500m": 4,
    ">500m": 5,
}

DURATION_MIDPOINTS = {
    "": 0.0,
    "<1": 0.5,
    "1-2": 1.5,
    "2-3": 2.5,
    "3-4": 3.5,
    "4-5": 4.5,
    "5-10": 7.5,
    "10+": 12.0,
}

EXECUTIVE_ROLES = {
    "ceo", "cto", "coo", "cfo", "cpo", "cso",
    "founder", "co-founder", "president",
    "executive chairman", "managing director", "general partner",
}

VC_ROLES = {
    "venture capitalist", "vc", "investor", "angel investor",
    "partner", "general partner", "principal",
}

POSTGRAD_DEGREES = {"md", "phd", "dphil", "mba", "ms", "msc", "ma", "llm", "jd", "dmd", "dds"}
UNDERGRAD_DEGREES = {"bs", "ba", "bsc", "beng", "btech", "be", "ab"}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def safe_parse(value: Any) -> list:
    """Parse a JSON/Python-literal string into a list. Returns [] on failure."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                result = parser(value)
                return result if isinstance(result, list) else []
            except Exception:
                continue
    return []


def normalise(s) -> str:
    """Lowercase + strip. Safely handles None, NaN (float), and non-strings."""
    if s is None:
        return ""
    if isinstance(s, float):   # catches NaN and any float passed by mistake
        return ""
    return str(s).strip().lower()


def parse_qs_rank(rank_str: str) -> int:
    """Convert QS rank string to int. Lower = better. Missing = 9999."""
    s = normalise(rank_str)
    if not s:
        return 9999
    if "200+" in s or s == "200+":
        return 200
    try:
        return int(s)
    except ValueError:
        return 9999


def ordinal_lookup(value: str, mapping: dict) -> int:
    return mapping.get(normalise(value), 0)


def duration_to_years(duration: str) -> float:
    return DURATION_MIDPOINTS.get(normalise(duration), 0.0)


# ─────────────────────────────────────────────
# Per-field feature extractors
# ─────────────────────────────────────────────

def education_features(educations_raw: Any) -> dict:
    edu_list = safe_parse(educations_raw)

    degrees = [normalise(e.get("degree", "")) for e in edu_list]
    fields  = [normalise(e.get("field", ""))  for e in edu_list]
    ranks   = [parse_qs_rank(e.get("qs_ranking", "")) for e in edu_list]

    best_rank = min(ranks) if ranks else 9999

    return {
        "edu_count":               len(edu_list),
        "has_md":                  int(any(d == "md"  for d in degrees)),
        "has_phd":                 int(any(d in {"phd", "dphil"} for d in degrees)),
        "has_mba":                 int(any(d == "mba" for d in degrees)),
        "has_postgrad":            int(any(d in POSTGRAD_DEGREES for d in degrees)),
        "has_undergrad":           int(any(d in UNDERGRAD_DEGREES for d in degrees)),
        "best_qs_rank":            best_rank,
        "has_top10_school":        int(best_rank <= 10),
        "has_top50_school":        int(best_rank <= 50),
        "has_top100_school":       int(best_rank <= 100),
        "edu_missing_degree_count": sum(1 for d in degrees if not d),
    }


def jobs_features(jobs_raw: Any, founder_industry: str) -> dict:
    job_list = safe_parse(jobs_raw)
    fin_industry = normalise(founder_industry)

    roles      = [normalise(j.get("role", ""))         for j in job_list]
    sizes      = [normalise(j.get("company_size", "")) for j in job_list]
    industries = [normalise(j.get("industry", ""))     for j in job_list]
    durations  = [duration_to_years(j.get("duration", "")) for j in job_list]

    size_tiers = [ordinal_lookup(s, COMPANY_SIZE_ORDER) for s in sizes]

    ceo_years = sum(
        dur for role, dur in zip(roles, durations)
        if "ceo" in role or "chief executive" in role
    )

    industry_match_count = sum(
        1 for ind in industries if ind and ind == fin_industry
    )

    return {
        "jobs_count":               len(job_list),
        "total_experience_years":   round(sum(durations), 2),
        "avg_job_duration":         round(np.mean(durations), 2) if durations else 0.0,
        "longest_tenure":           max(durations) if durations else 0.0,
        "ceo_experience_count":     sum(1 for r in roles if "ceo" in r),
        "has_ceo_experience":       int(any("ceo" in r for r in roles)),
        "ceo_total_years":          round(ceo_years, 2),
        "has_vc_experience":        int(any(r in VC_ROLES or "venture" in r for r in roles)),
        "has_executive_experience": int(any(r in EXECUTIVE_ROLES for r in roles)),
        "max_company_size_tier":    max(size_tiers) if size_tiers else 0,
        "avg_company_size_tier":    round(np.mean(size_tiers), 2) if size_tiers else 0.0,
        "industry_match_job_count": industry_match_count,
        "industry_match_ratio":     round(industry_match_count / len(job_list), 2) if job_list else 0.0,
    }


def ipo_features(ipos_raw: Any) -> dict:
    ipo_list = safe_parse(ipos_raw)

    val_tiers    = [ordinal_lookup(i.get("valuation_usd", ""),    VALUATION_ORDER)    for i in ipo_list]
    raised_tiers = [ordinal_lookup(i.get("amount_raised_usd", ""), AMOUNT_RAISED_ORDER) for i in ipo_list]

    return {
        "ipo_count":          len(ipo_list),
        "has_ipo":            int(len(ipo_list) > 0),
        "ipo_max_val_tier":   max(val_tiers)    if val_tiers    else 0,
        "ipo_max_raised_tier":max(raised_tiers) if raised_tiers else 0,
        "ipo_any_large":      int(any(t >= 4 for t in val_tiers)),   # >500M
    }


def acquisition_features(acqs_raw: Any) -> dict:
    acq_list = safe_parse(acqs_raw)

    price_tiers  = [ordinal_lookup(a.get("price_usd", ""), VALUATION_ORDER) for a in acq_list]
    well_known   = [bool(a.get("acquired_by_well_known", False))             for a in acq_list]
    undisclosed  = [normalise(a.get("price_usd", "")) == "undisclosed"       for a in acq_list]

    return {
        "acquisition_count":        len(acq_list),
        "has_acquisition":          int(len(acq_list) > 0),
        "acq_max_price_tier":       max(price_tiers) if price_tiers else 0,
        "acq_any_large":            int(any(t >= 4 for t in price_tiers)),   # >500M
        "acq_any_well_known_buyer": int(any(well_known)),
        "acq_undisclosed_count":    sum(undisclosed),
    }


def industry_features(industry: str) -> dict:
    ind = normalise(industry)
    return {
        "industry_is_missing": int(not ind),
        # Add one-hot flags for your most common industries, e.g.:
        "industry_is_biotech":  int("biotech" in ind or "nanotechnology" in ind),
        "industry_is_software": int("software" in ind or "saas" in ind),
        "industry_is_fintech":  int("fintech" in ind or "financial" in ind),
        "industry_is_health":   int("health" in ind or "medical" in ind or "pharma" in ind),
    }


# ─────────────────────────────────────────────
# Master function: single record → feature dict
# ─────────────────────────────────────────────

def extract_features(record: dict) -> dict:
    """
    Takes one raw founder record (dict) and returns a flat feature dict.
    Includes numeric features and a raw 'text_summary' for TF-IDF.
    """
    industry = record.get("industry", "")

    features = {}
    features.update(education_features(record.get("educations_json")))
    features.update(jobs_features(record.get("jobs_json"), industry))
    features.update(ipo_features(record.get("ipos")))
    features.update(acquisition_features(record.get("acquisitions")))
    features.update(industry_features(industry))
    
    # Add raw text for TF-IDF
    features["text_summary"] = extract_text_summary(record)

    return features

def extract_text_summary(record: dict) -> str:
    """Concatenate roles, companies, degrees, and institutions for NLP."""
    edu_list = safe_parse(record.get("educations_json"))
    job_list = safe_parse(record.get("jobs_json"))
    
    parts = []
    for j in job_list:
        parts.append(str(j.get("role", "")))
        parts.append(str(j.get("company_name", "")))
    for e in edu_list:
        parts.append(str(e.get("degree", "")))
        parts.append(str(e.get("field", "")))
        parts.append(str(e.get("institution_name", "")))
        
    return " ".join([p for p in parts if p]).lower()


# ─────────────────────────────────────────────
# DataFrame-level helper
# ─────────────────────────────────────────────

def build_feature_dataframe(records: list[dict]) -> pd.DataFrame:
    """
    Converts a list of raw founder records into a feature DataFrame.
    Includes the target column `success` when present.
    """
    rows = []
    for rec in records:
        feat = extract_features(rec)
        if "success" in rec:
            feat["success"] = rec["success"]
        if "founder_uuid" in rec:
            feat["founder_uuid"] = rec["founder_uuid"]
        rows.append(feat)

    df = pd.DataFrame(rows)

    # Enforce numeric dtypes (safety net), excluding text
    non_numeric = ["founder_uuid", "text_summary"]
    numeric_cols = [c for c in df.columns if c not in non_numeric]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    
    # Ensure text_summary is string
    if "text_summary" in df.columns:
        df["text_summary"] = df["text_summary"].astype(str).fillna("")

    return df

