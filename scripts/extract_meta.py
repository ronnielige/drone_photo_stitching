import glob, os, re, json
import piexif

SRC = r"D:\lr\project\无人机航拍拼接\xiangjia1_186m"

def parse_ratio(v):
    # piexif GPS value tuple ((num,den),...) -> degrees
    if not v:
        return None
    def cv(x):
        return x[0]/x[1] if isinstance(x, (tuple, list)) else x
    deg, mn, sc = cv(v[0]), cv(v[1]), cv(v[2])
    return deg + mn/60.0 + sc/3600.0

def extract_xmp_dji(data):
    txt = data  # raw bytes
    out = {}
    pat = re.compile(rb'([A-Za-z0-9]+):(Yaw|Pitch|Roll|AbsoluteAltitude|RelativeAltitude|FlightYawDegree|FlightPitchDegree|FlightRollDegree|GimbalYawDegree|GimbalPitchDegree|GimbalRollDegree|GpsLongitude|GpsLatitude|GPSLatitude|GPSLongitude)="([^"]*)"')
    for m in pat.finditer(txt):
        key = m.group(1).decode() + ":" + m.group(2).decode()
        out[key] = m.group(3).decode('utf-8', 'ignore')
    return out

rows = []
files = sorted(glob.glob(os.path.join(SRC, "*.JPG")))
for f in files:
    name = os.path.basename(f)
    try:
        data = open(f, 'rb').read()
        ex = piexif.load(f)
        g = ex['GPS']
        lat = parse_ratio(g[2]); latref = g.get(1, b'')
        lon = parse_ratio(g[4]); lonref = g.get(3, b'')
        if latref in (b'S',): lat = -lat
        if lonref in (b'W',): lon = -lon
        alt = g[6][0]/g[6][1] if g.get(6) else None
        exif = ex['Exif']
        ff = exif.get(33434)  # exposure
        fnum = exif.get(33437)
        focal = exif.get(37386)
        w = exif.get(40962); h = exif.get(40963)
        iso = exif.get(34855)
        exp = (ff[0]/ff[1] if isinstance(ff, tuple) and ff else ff)
        foc = (focal[0]/1000.0 if isinstance(focal, tuple) and focal else focal)
        fn  = (fnum[0]/fnum[1] if isinstance(fnum, tuple) and fnum else fnum)
        sn = exif.get(42033)
        maker = extract_xmp_dji(data)
        rows.append(dict(name=name, width=w, height=h, lat=lat, lon=lon, alt=alt,
                         focal_mm=foc, fnum=fn, shutter=exp, iso=iso, sn=sn.decode() if sn else None,
                         maker=maker))
    except Exception as e:
        rows.append(dict(name=name, error=str(e)))

with open(r"D:\lr\project\无人机航拍拼接\mysolution\out\metadata.json", "w") as fh:
    json.dump(rows, fh, indent=1)

print("parsed", len(rows), "images")
for r in rows[:3]:
    print(r['name'], r.get('lat'), r.get('lon'), r.get('alt'), r.get('focal_mm'),
          'yaw=', (r.get('maker') or {}).get('DJI:Yaw'))
print("first row maker sample keys:", list(rows[0].get('maker', {}).items())[:6])