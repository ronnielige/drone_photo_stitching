import json
rows = json.load(open(r"D:\lr\project\无人机航拍拼接\mysolution\out\metadata.json"))
for r in rows[:2]:
    print("==", r['name'], "lat=%.6f lon=%.6f relalt=%s" % (r['lat'], r['lon'], (r['maker'] or {}).get('dji:RelativeAltitude')))
    for k, v in (r.get('maker') or {}).items():
        print("   ", k, "=", v)
    print("focal_mm", r['focal_mm'], "fnum", r['fnum'], "shutter", r['shutter'], "sn", r['sn'])
# stats
import statistics
reld = sorted(float(m.get('dji:RelativeAltitude') or 0) for m in rows)
print("relative altitude min/med/max:", min(reld), statistics.median(reld), max(reld))
yaws = [m.get('dji:GimbalYawDegree') for m in rows]
print("unique gimbal yaw values:", sorted(set(yaws)))
pitchs = sorted(set(m.get('dji:GimbalPitchDegree') for m in rows))
print("gimbal pitch set:", pitchs)
rolls = sorted(set(m.get('dji:GimbalRollDegree') for m in rows))
print("gimbal roll set:", rolls)