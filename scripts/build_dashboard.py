"""Write a self-contained dashboard with the experiment data inlined.

The output opens in any browser with no server, no database, and no network,
which makes it a safe fallback for a presentation.
"""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashboard_queries import collect

TEMPLATE = "dashboard/template.html"
OUTPUT = "dashboard/dashboard.html"


def main():
    data = collect()
    data["built"] = datetime.now().strftime("%d %B %Y at %H:%M")

    html = open(TEMPLATE).read()
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    html = html.replace("__LIVE__", "false")

    with open(OUTPUT, "w") as f:
        f.write(html)

    size = os.path.getsize(OUTPUT) / 1024
    print(f"{OUTPUT}  ({size:.0f} KB)")
    print(f"  {data['corpus']['rows']:,} records, "
          f"{data['corpus']['experiments']} experiments, "
          f"{len(data['timeline']):,} daily points")


if __name__ == "__main__":
    main()
