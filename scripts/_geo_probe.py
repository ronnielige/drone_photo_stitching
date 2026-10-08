import tifffile, os
BASE = r"D:\lr\project\无人机航拍拼接"
for key, path in [("MINE final mosaic","mysolution/out/mosaic_geotiff.tif"),
                  ("MINE 5cm ortho","mysolution/out/xiangjia_ortho_mine.tif"),
                  ("DJI","xiangjia1/map/result.tif")]:
    fp = os.path.join(BASE, path)
    try:
        t = tifffile.TiffFile(fp)
        p = t.pages[0]
        gsd = None; tie=None
        try: gsd = p.tags["ModelPixelScale"].value.tolist()
        except: pass
        try: tie = p.tags["ModelTiepoint"].value.tolist()
        except: pass
        print(key, "| page.shape(h,w,ch)=", p.shape, "| dtype=", p.dtype, "| gsd(x,y)=", gsd, "| tie=", tie)
    except Exception as e:
        print(key, "ERR", repr(e))