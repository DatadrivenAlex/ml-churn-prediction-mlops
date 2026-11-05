"""Column names and allowed categories shared by the generator, model, and API."""

TARGET = "churn"
ID_COLUMN = "customer_id"

CONTRACTS = ("month_to_month", "one_year", "two_year")
INTERNET_SERVICES = ("dsl", "fiber", "none")
PAYMENT_METHODS = ("electronic_check", "mailed_check", "bank_transfer", "credit_card")
ADDON_VALUES = ("yes", "no", "no_internet")
REGIONS = ("north", "south", "east", "west")

RAW_NUMERIC = (
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets_90d",
    "avg_monthly_gb",
    "satisfaction_score",
)
RAW_BOOLEAN = (
    "paperless_billing",
    "senior_citizen",
    "partner",
    "dependents",
    "autopay",
)
RAW_CATEGORICAL = (
    "contract",
    "internet_service",
    "payment_method",
    "online_security",
    "tech_support",
    "streaming_tv",
    "region",
)
REQUIRED_COLUMNS = RAW_NUMERIC + RAW_BOOLEAN + RAW_CATEGORICAL

# Inputs monitored for population stability. Same fields the API accepts.
DRIFT_FEATURES = REQUIRED_COLUMNS
