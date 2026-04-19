import pandas as pd

def format_currency(value):
    return f"${value:,.2f}"

def get_risk_color(score):
    if score > 0.7:
        return "red"
    elif score > 0.3:
        return "orange"
    else:
        return "green"
