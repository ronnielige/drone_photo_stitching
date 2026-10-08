import tifffile
p = r"D:\lr\project\无人机航拍拼接\xiangjia1\map\result.tif"
with tifffile.TiffFile(p) as tf:
    for i, page in enumerate(tf.pages):
        print("page", i, "shape", page.shape, "dtype", page.dtype, "SAMPLEFORMAT", page.tags.get('SampleFormat'))
    # series
    print("series:", tf.series[0].shape)
    geo = tf.pages[0].tags.get('ModelTransformation') or tf.pages[0].tags.get('GeoTiePoints') or tf.pages[0].tags.get('ModelPixelScale')
    print("geo tags present:", [t for t in ('ModelPixelScale','ModelTransformation','GeoTiePoints','GeoAsciiParams') if t in tf.pages[0].tags])
    for t in ('ModelPixelScale','ModelTiepoint','GeoAsciiParams','ModelTransformation'):
        if t in tf.pages[0].tags:
            print(t, tf.pages[0].tags[t].value)
    if 'ModelTiepoint' in tf.pages[0].tags:
        print("tiepoint", tf.pages[0].tags['ModelTiepoint'].value)