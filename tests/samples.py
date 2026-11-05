"""Small customer payloads shared by API tests."""


def customer_payload(**overrides) -> dict:
    payload = {
        "customer_id": "C000001",
        "tenure_months": 36,
        "contract": "two_year",
        "internet_service": "dsl",
        "payment_method": "credit_card",
        "monthly_charges": 55.0,
        "total_charges": 1900.0,
        "paperless_billing": False,
        "senior_citizen": False,
        "partner": True,
        "dependents": False,
        "online_security": "yes",
        "tech_support": "yes",
        "streaming_tv": "no",
        "support_tickets_90d": 0,
        "avg_monthly_gb": 90.0,
        "satisfaction_score": 5,
        "autopay": True,
        "region": "north",
    }
    payload.update(overrides)
    return payload


def risky_payload() -> dict:
    return customer_payload(
        customer_id="C999999",
        tenure_months=2,
        contract="month_to_month",
        internet_service="fiber",
        payment_method="electronic_check",
        monthly_charges=110.0,
        total_charges=150.0,
        paperless_billing=True,
        senior_citizen=True,
        partner=False,
        online_security="no",
        tech_support="no",
        streaming_tv="yes",
        support_tickets_90d=6,
        avg_monthly_gb=20.0,
        satisfaction_score=1,
        autopay=False,
        region="south",
    )
