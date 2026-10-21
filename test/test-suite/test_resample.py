# vim: set fileencoding=utf-8 :
import pytest

import pyvips
from helpers import *


# Run a function expecting a complex image on a two-band image
def run_cmplx(fn, image):
    if image.format == pyvips.BandFormat.FLOAT:
        new_format = pyvips.BandFormat.COMPLEX
    elif image.format == pyvips.BandFormat.DOUBLE:
        new_format = pyvips.BandFormat.DPCOMPLEX
    else:
        raise pyvips.Error("run_cmplx: not float or double")

    # tag as complex, run, revert tagging
    cmplx = image.copy(bands=1, format=new_format)
    cmplx_result = fn(cmplx)

    return cmplx_result.copy(bands=2, format=image.format)


def to_polar(image):
    """Transform image coordinates to polar.

    The image is transformed so that it is wrapped around a point in the
    centre. Vertical straight lines become circles or segments of circles,
    horizontal straight lines become radial spokes.
    """
    # xy image, zero in the centre, scaled to fit image to a circle
    xy = pyvips.Image.xyz(image.width, image.height)
    xy -= [image.width / 2.0, image.height / 2.0]
    scale = min(image.width, image.height) / float(image.width)
    xy *= 2.0 / scale

    # to polar, scale vertical axis to 360 degrees
    index = run_cmplx(lambda x: x.polar(), xy)
    index *= [1, image.height / 360.0]

    return image.mapim(index)


def to_rectangular(image):
    """Transform image coordinates to rectangular.

    The image is transformed so that it is unwrapped from a point in the
    centre. Circles or segments of circles become vertical straight lines,
    radial lines become horizontal lines.
    """
    # xy image, vertical scaled to 360 degrees
    xy = pyvips.Image.xyz(image.width, image.height)
    xy *= [1, 360.0 / image.height]

    # to rect, scale to image rect
    index = run_cmplx(lambda x: x.rect(), xy)
    scale = min(image.width, image.height) / float(image.width)
    index *= scale / 2.0
    index += [image.width / 2.0, image.height / 2.0]

    return image.mapim(index)


class TestResample:
    def test_affine(self):
        im = pyvips.Image.new_from_file(JPEG_FILE)

        # vsqbs is non-interpolatory, don't test this way
        for name in ["nearest", "bicubic", "bilinear", "nohalo", "lbb"]:
            x = im
            interpolate = pyvips.Interpolate.new(name)
            for i in range(4):
                x = x.affine([0, 1, 1, 0], interpolate=interpolate)

            assert (x - im).abs().max() == 0

    def test_reduce(self):
        im = pyvips.Image.new_from_file(JPEG_FILE)
        # cast down to 0-127, the smallest range, so we aren't messed up by
        # clipping
        im = im.cast(pyvips.BandFormat.CHAR)

        for fac in [1, 1.1, 1.5, 1.999]:
            for fmt in all_formats:
                for kernel in ["nearest", "linear",
                               "cubic", "mitchell", "lanczos2",
                               "lanczos3", "mks2013", "mks2021"]:
                    x = im.cast(fmt)
                    r = x.reduce(fac, fac, kernel=kernel)
                    d = abs(r.avg() - im.avg())
                    assert d < 2

        # try constant images ... should not change the constant
        for const in [0, 1, 2, 254, 255]:
            im = (pyvips.Image.black(10, 10) + const).cast("uchar")
            for kernel in ["nearest", "linear",
                           "cubic", "mitchell", "lanczos2",
                           "lanczos3", "mks2013", "mks2021"]:
                # print "testing kernel =", kernel
                # print "testing const =", const
                shr = im.reduce(2, 2, kernel=kernel)
                d = abs(shr.avg() - im.avg())
                assert d == 0

        # https://github.com/libvips/libvips/issues/4864
        if have("ppmload"):
            im = pyvips.Image.new_from_buffer(b'P6\n2 2\n255\n'
                                              b'\xff\x00\x00' b'\x00\xff\x00'
                                              b'\x00\x00\xff' b'\xff\xff\x00', "")
            im2 = im.reduceh(1.5, kernel="nearest")
            assert im2.width == 1

    def test_resize(self):
        im = pyvips.Image.new_from_file(JPEG_FILE)
        im2 = im.resize(0.25)
        # in py3, round() does not round to nearest in the obvious way, so we
        # have to do it by hand
        assert im2.width == int(im.width / 4.0 + 0.5)
        assert im2.height == int(im.height / 4.0 + 0.5)

        # test geometry rounding corner case
        im = pyvips.Image.black(100, 1)
        x = im.resize(0.5)
        assert x.width == 50
        assert x.height == 1

        # test whether we use double-precision calculations in reduce{h,v}
        im = pyvips.Image.black(1600, 1000)
        x = im.resize(10.0 / im.width)
        assert x.width == 10
        assert x.height == 6

        # test round-up option of shrink
        im = pyvips.Image.black(2049 - 2, 2047 - 2, bands=3)
        im = im.embed(1, 1, 2049, 2047,
                      extend=pyvips.Extend.BACKGROUND,
                      background=[255, 0, 0])
        for scale in [8, 9.4, 16]:
            x = im.resize(1 / scale, vscale=1 / scale)

            for point in ([(round(x.width / 2), 0),
                           (x.width - 1, round(x.height / 2)),
                           (round(x.width / 2), x.height - 1),
                           (0, round(x.height / 2))]):
                y = x(*point)[0]
                assert y != 0

    def test_shrink(self):
        im = pyvips.Image.new_from_file(JPEG_FILE)
        im2 = im.shrink(4, 4)
        # in py3, round() does not round to nearest in the obvious way, so we
        # have to do it by hand
        assert im2.width == int(im.width / 4.0 + 0.5)
        assert im2.height == int(im.height / 4.0 + 0.5)
        assert abs(im.avg() - im2.avg()) < 1

        im2 = im.shrink(2.5, 2.5)
        assert im2.width == int(im.width / 2.5 + 0.5)
        assert im2.height == int(im.height / 2.5 + 0.5)
        assert abs(im.avg() - im2.avg()) < 1

        # https://github.com/libvips/libvips/issues/4864
        if have("ppmload"):
            im = pyvips.Image.new_from_buffer(b'P6\n2 2\n255\n'
                                              b'\xff\x00\x00' b'\x00\xff\x00'
                                              b'\x00\x00\xff' b'\xff\xff\x00', "")
            im2 = im.shrinkh(2)
            assert im2.width == 1
            assert abs(im.avg() - im2.avg()) < 1

    def test_thumbnail(self):
        im = pyvips.Image.thumbnail(JPEG_FILE, 100)

        assert im.height == 100
        assert im.bands == 3
        assert im.bands == 3

        # the average shouldn't move too much
        im_orig = pyvips.Image.new_from_file(JPEG_FILE)
        assert abs(im_orig.avg() - im.avg()) < 1

        # make sure we always get the right width
        for height in range(440, 1, -13):
            im = pyvips.Image.thumbnail(JPEG_FILE, height)
            assert im.height == height

        # should fit one of width or height
        im = pyvips.Image.thumbnail(JPEG_FILE, 100, height=300)
        assert im.width == 100
        assert im.height != 300
        im = pyvips.Image.thumbnail(JPEG_FILE, 300, height=100)
        assert im.width != 300
        assert im.height == 100

        # with @crop, should fit both width and height
        im = pyvips.Image.thumbnail(JPEG_FILE, 100,
                                    height=300, crop=True)
        assert im.width == 100
        assert im.height == 300

        im1 = pyvips.Image.thumbnail(JPEG_FILE, 100)
        with open(JPEG_FILE, 'rb') as f:
            buf = f.read()
        im2 = pyvips.Image.thumbnail_buffer(buf, 100)
        assert abs(im1.avg() - im2.avg()) < 1

        # should be able to thumbnail many-page tiff
        im = pyvips.Image.thumbnail(OME_FILE, 100)
        assert im.width == 100
        assert im.height == 38

        # should be able to thumbnail individual pages from many-page tiff
        im1 = pyvips.Image.thumbnail(OME_FILE + "[page=0]", 100)
        assert im1.width == 100
        assert im1.height == 38
        im2 = pyvips.Image.thumbnail(OME_FILE + "[page=1]", 100)
        assert im2.width == 100
        assert im2.height == 38
        assert (im1 - im2).abs().max() != 0

        # should be able to thumbnail entire many-page tiff as a toilet-roll
        # image
        im = pyvips.Image.thumbnail(OME_FILE + "[n=-1]", 100)
        assert im.width == 100
        assert im.height == 570

        # should be able to thumbnail a single-page tiff in a buffer
        im1 = pyvips.Image.thumbnail(TIF_FILE, 100)
        with open(TIF_FILE, 'rb') as f:
            buf = f.read()
        im2 = pyvips.Image.thumbnail_buffer(buf, 100)
        assert abs(im1.avg() - im2.avg()) < 1

        # linear shrink should work on rgba images
        if have("ppmload"):
            im1 = pyvips.Image.thumbnail(RGBA_FILE, 64, linear=True)
            im2 = pyvips.Image.new_from_file(RGBA_CORRECT_FILE)
            assert abs(im1.flatten(background=255).avg() - im2.avg()) < 1

        # thumbnailing a 16-bit image should always make an 8-bit image
        rgb16_buffer = pyvips.Image \
                .new_from_file(JPEG_FILE) \
                .colourspace("rgb16") \
                .write_to_buffer(".png")
        thumb = pyvips.Image.thumbnail_buffer(rgb16_buffer, 128)
        assert thumb.format == "uchar"

        if have("heifload"):
            # this image is orientation 6 ... thumbnail should flip it
            im = pyvips.Image.new_from_file(AVIF_FILE)
            thumb = pyvips.Image.thumbnail(AVIF_FILE, 100)

            # thumb should be portrait
            assert thumb.width < thumb.height
            assert thumb.height == 100

    def test_thumbnail_kernel(self):
        # test that the kernel option works
        im = pyvips.Image.thumbnail(JPEG_FILE, 100, kernel="nearest")
        assert im.height == 100

        im = pyvips.Image.thumbnail(JPEG_FILE, 100, kernel="linear")
        assert im.height == 100

        im = pyvips.Image.thumbnail(JPEG_FILE, 100, kernel="cubic")
        assert im.height == 100

        im = pyvips.Image.thumbnail(JPEG_FILE, 100, kernel="lanczos2")
        assert im.height == 100

        im = pyvips.Image.thumbnail(JPEG_FILE, 100, kernel="lanczos3")
        assert im.height == 100

        # different kernels should produce different results
        im_nearest = pyvips.Image.thumbnail(JPEG_FILE, 100, kernel="nearest")
        im_lanczos3 = pyvips.Image.thumbnail(JPEG_FILE, 100, kernel="lanczos3")
        assert (im_nearest - im_lanczos3).abs().max() > 0

        # test that kernel works with thumbnail_buffer
        with open(JPEG_FILE, 'rb') as f:
            buf = f.read()
        im_buf = pyvips.Image.thumbnail_buffer(buf, 100, kernel="cubic")
        assert im_buf.height == 100

        # test that kernel works with thumbnail_image
        im_orig = pyvips.Image.new_from_file(JPEG_FILE)
        im_thumb = im_orig.thumbnail_image(100, kernel="cubic")
        assert im_thumb.height == 100

    def test_resize_upsize_kernel(self):
        im = pyvips.Image.new_from_file(JPEG_FILE)

        for scale in [1.5, 2, 3.7]:
            a = im.resize(scale, kernel="cubic")
            b = im.resize(scale, kernel="mitchell")
            assert a.width == b.width
            assert a.height == b.height
            assert (a - b).abs().max() > 0

    def test_resize_upsize_fallback(self):
        # there is no lanczos interpolator, so enlarging with lanczos3 falls
        # back to bicubic ... this pins the current behaviour, it does not
        # endorse it; delete this test when a lanczos interpolator arrives
        im = pyvips.Image.new_from_file(JPEG_FILE)
        a = im.resize(2, kernel="cubic")
        b = im.resize(2, kernel="lanczos3")
        assert (a - b).abs().max() == 0

    def test_resize_upsize_constant(self):
        # constant images must survive every format path ... compare
        # against the cast image, since char clips 254 and 255 to 127
        def check(im):
            value = im.avg()
            out = im.resize(2.5, kernel="mitchell")
            assert out.width == 25
            assert out.height == 25
            if im.format in complex_formats:
                assert abs(out.avg() - value) < 1e-9
            elif im.format in float_formats:
                assert abs(out.min() - value) < 1e-9
                assert abs(out.max() - value) < 1e-9
            else:
                assert out.min() == value
                assert out.max() == value

        for const in [0, 1, 2, 254, 255]:
            for fmt in all_formats:
                check((pyvips.Image.black(10, 10) + const).cast(fmt))

        # negative constants for the signed double-table paths ... char is
        # left out: its fixed-point path floors negatives, for cubic too
        for fmt, type_min in [("short", -32768), ("int", -2147483648)]:
            for const in [-1, -2, -254, -255, type_min]:
                check((pyvips.Image.black(10, 10) + const).cast(fmt))

    def test_resize_upsize_reference(self):
        # Mitchell-Netravali cubic filter
        def cubic_filter(x, B, C):
            ax = abs(x)
            if ax <= 1:
                return ((12 - 9 * B - 6 * C) * ax ** 3 +
                        (-18 + 12 * B + 6 * C) * ax ** 2 +
                        (6 - 2 * B)) / 6
            if ax <= 2:
                return ((-B - 6 * C) * ax ** 3 +
                        (6 * B + 30 * C) * ax ** 2 +
                        (-12 * B - 48 * C) * ax +
                        (8 * B + 24 * C)) / 6
            return 0.0

        kernels = {"cubic": (0.0, 0.5), "mitchell": (1 / 3, 1 / 3)}
        width = 11
        impulse = 5

        # double: table-free path, any scale, tight tolerance
        for kernel, (B, C) in kernels.items():
            for scale in [1.5, 2, 3.7]:
                im = pyvips.Image.black(width, 1).cast("double")
                im = im.draw_rect(1.0, impulse, 0, 1, 1)
                out = im.resize(scale, vscale=1, kernel=kernel)
                assert out.width == int(width * scale + 0.5)
                for X in range(out.width):
                    # centre convention: output pixel X samples input x
                    x = (X + 0.5) / scale - 0.5
                    if x - 2 < 0 or x + 2 > width - 1:
                        continue
                    expected = cubic_filter(x - impulse, B, C)
                    assert abs(out(X, 0)[0] - expected) < 1e-9

        # uchar: fixed-point path, offsets at scale 2 fall on the 1/64 table
        # grid, values clip at 0 ... the observed error is 0, 1 is headroom
        for kernel, (B, C) in kernels.items():
            im = pyvips.Image.black(width, 1)
            im = im.draw_rect(255, impulse, 0, 1, 1)
            out = im.resize(2, vscale=1, kernel=kernel)
            for X in range(out.width):
                x = (X + 0.5) / 2 - 0.5
                if x - 2 < 0 or x + 2 > width - 1:
                    continue
                expected = round(255 * cubic_filter(x - impulse, B, C))
                expected = min(max(expected, 0), 255)
                assert abs(out(X, 0)[0] - expected) <= 1

    def test_resize_upsize_interpolate(self):
        interpolate = pyvips.Interpolate.new("mitchell")
        im = (pyvips.Image.black(10, 10) + 128).cast("uchar")
        out = im.affine([2, 0, 0, 2], interpolate=interpolate,
                        extend="copy")
        assert out.width == 20
        assert out.height == 20
        assert out.min() == 128
        assert out.max() == 128

    def test_thumbnail_upsize_kernel(self):
        im = pyvips.Image.new_from_file(JPEG_FILE)
        width = 2 * im.width
        height = 2 * im.height

        # thumbnail output is sequential, and t1 is read twice
        t1 = pyvips.Image.thumbnail(JPEG_FILE, width, height=height,
                                    size="up", kernel="mitchell")
        t1 = t1.copy_memory()
        t2 = pyvips.Image.thumbnail(JPEG_FILE, width, height=height,
                                    size="up", kernel="cubic")
        assert t1.width == width
        assert t2.width == width
        assert (t1 - t2).abs().max() > 0

        # thumbnail enlarges with resize, so the two must agree
        r = im.resize(2, kernel="mitchell")
        assert (t1 - r).abs().max() == 0

    def test_thumbnail_icc(self):
        im = pyvips.Image.thumbnail(JPEG_FILE_XYB, 442, output_profile="srgb")

        assert im.width == 290
        assert im.height == 442
        assert im.bands == 3

        # the colour distance should not deviate too much
        # (i.e. the embedded profile should not be ignored)
        im_orig = pyvips.Image.new_from_file(JPEG_FILE)
        assert im_orig.de00(im).max() < 10

    # this has caused a few bugs in the past ,,,
    def test_thumbnail_uhdr_linear(self):
        im = pyvips.Image.thumbnail(UHDR_FILE, 128, linear=True)

        assert im.width == 128
        assert im.bands == 3

    def test_similarity(self):
        im = pyvips.Image.new_from_file(JPEG_FILE)
        im2 = im.similarity(angle=90)
        im3 = im.affine([0, -1, 1, 0])
        # rounding in calculating the affine transform from the angle stops
        # this being exactly true
        assert (im2 - im3).abs().max() < 50

    def test_similarity_scale(self):
        im = pyvips.Image.new_from_file(JPEG_FILE)
        im2 = im.similarity(scale=2)
        im3 = im.affine([2, 0, 0, 2])
        assert (im2 - im3).abs().max() == 0

    # added in 8.7
    def test_rotate(self):
        if have("rotate"):
            im = pyvips.Image.new_from_file(JPEG_FILE)
            im2 = im.rotate(90)
            im3 = im.affine([0, -1, 1, 0])
            # rounding in calculating the affine transform from the angle stops
            # this being exactly true
            assert (im2 - im3).abs().max() < 50

    def test_mapim(self):
        im = pyvips.Image.new_from_file(JPEG_FILE)

        p = to_polar(im)
        r = to_rectangular(p)

        # the left edge (which is squashed to the origin) will be badly
        # distorted, but the rest should not be too bad
        a = r.crop(50, 0, im.width - 50, im.height).gaussblur(2)
        b = im.crop(50, 0, im.width - 50, im.height).gaussblur(2)
        assert (a - b).abs().max() < 50

        # this was a bug at one point, strangely, if executed with debug
        # enabled
        mp = pyvips.Image.xyz(im.width, im.height)
        interp = pyvips.Interpolate.new('bicubic')
        assert im.mapim(mp, interpolate=interp).avg() == im.avg()


if __name__ == '__main__':
    pytest.main()
