# core/formatter.py
def format_currency(value):
    return f"Rp{value:,.0f}" if value else "N/A"

def format_percent(value):
    return f"{value:.2%}" if value else "N/A"

def get_color(value, reverse=False):
    if value is None:
        return ""
    if reverse:
        return "negative" if value > 0 else "positive"
    return "positive" if value > 0 else "negative"
