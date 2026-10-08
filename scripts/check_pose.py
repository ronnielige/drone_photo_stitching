import json
rows = json.load(open(r"D:\lr\project\无人机航拍拼接\mysolution\out\metadata.json"))
def have(k):
    return sum(1 for r in rows if (r.get('maker') or {}).get(k) not in (None, ""))
for k in ["dji:RelativeAltitude","dji:GimbalYawDegree","dji:GimbalPitchDegree",
          "dji:GimbalRollDegree","dji:FlightYawDegree","dji:FlightPitchDegree","dji:FlightRollDegree"]:
    print(k, have(k))
nolat = sum(1 for r in rows if r.get('lat') is None)
print("rows with no gps:", nolat, "total", len(rows))
import collections
missing = [r['name'] for r in rows if not (r.get('maker') or {}).get('dji:GimbalYawDegree')]
print("first 10 lacking gimbal yaw:", missing[:10])
# check whether mission grid: imager yaw roughly -90 (rows) or flight direction
yaws = [(r['name'][-20:-4], (r['maker'] or {}).get('dji:GimbalYawDegree')) for r in rows if (r['maker'] or {}).get('dji:GimbalYawDegree')]
import statistics
ys = [float(y) for _, y in yaws]
print("gimbal yaw count", len(ys), "min", min(ys), "max", max(ys), "median", statistics.median(ys))
from collections import Counter
print("yaw histogram first:", Counter(round(y/30)*30 for y in ys).most_common())