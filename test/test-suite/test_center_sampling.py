# vim: set fileencoding=utf-8 :
"""
Test center sampling behavior for resize operations.

Uses impulse response (single white pixel) to verify that vips resize
aligns output pixel centers with input positions according to:
    output = (input + 0.5) * scale - 0.5
"""

import pytest
import pyvips


class TestCenterSampling:
    """Test center sampling alignment using impulse response.

    Creates images with a single white pixel (impulse), resizes them,
    and verifies the centroid of the output matches the expected
    position from the center sampling formula.
    """

    @staticmethod
    def create_impulse(width, height, impulse_x, impulse_y):
        """Create a black image with a single white pixel at (impulse_x, impulse_y)."""
        img = pyvips.Image.black(width, height)
        img = img.draw_rect(255, impulse_x, impulse_y, 1, 1, fill=True)
        return img.cast("uchar")

    @staticmethod
    def find_centroid(image, row=None, col=None):
        """Find centroid along one axis.

        If row is specified, find x-centroid along that row.
        If col is specified, find y-centroid along that column.
        Returns weighted average position by pixel intensity.
        """
        if row is not None:
            size = image.width
            get_val = lambda i: max(0, image(i, row)[0])
        elif col is not None:
            size = image.height
            get_val = lambda i: max(0, image(col, i)[0])
        else:
            raise ValueError("Must specify row or col")

        total_weight = 0.0
        weighted_sum = 0.0

        for i in range(size):
            val = get_val(i)
            total_weight += val
            weighted_sum += i * val

        if total_weight == 0:
            return None
        return weighted_sum / total_weight

    @staticmethod
    def expected_position(input_pos, scale):
        """Calculate expected output position.

        Formula: output = (input + 0.5) * scale - 0.5
        """
        return (input_pos + 0.5) * scale - 0.5

    def test_uniform_upscale(self):
        """Test that uniform upscaling aligns pixel centers."""
        width, height = 100, 20
        impulse_x, impulse_y = 50, 10

        for scale in [2.0, 3.0, 4.0]:
            img = self.create_impulse(width, height, impulse_x, impulse_y)
            resized = img.resize(scale)

            expected_x = self.expected_position(impulse_x, scale)
            expected_y = self.expected_position(impulse_y, scale)

            measured_x = self.find_centroid(resized, row=int(expected_y + 0.5))
            measured_y = self.find_centroid(resized, col=int(expected_x + 0.5))

            assert measured_x is not None, f"No impulse found at scale={scale}"
            # Tolerance of 0.15 allows for minor interpolation effects
            # The bug causes exactly +0.5 output pixel shift
            assert abs(measured_x - expected_x) < 0.15, \
                f"scale={scale}: x centroid at {measured_x:.3f}, expected {expected_x:.3f}"
            assert abs(measured_y - expected_y) < 0.15, \
                f"scale={scale}: y centroid at {measured_y:.3f}, expected {expected_y:.3f}"

    def test_nonuniform_upscale(self):
        """Test non-uniform upscaling (different scales per dimension)."""
        width, height = 100, 20
        impulse_x, impulse_y = 50, 10

        test_cases = [
            (2.0, 3.0),
            (3.0, 2.0),
            (1.5, 4.0),
        ]

        for hscale, vscale in test_cases:
            img = self.create_impulse(width, height, impulse_x, impulse_y)
            resized = img.resize(hscale, vscale=vscale)

            expected_x = self.expected_position(impulse_x, hscale)
            expected_y = self.expected_position(impulse_y, vscale)

            measured_x = self.find_centroid(resized, row=int(expected_y + 0.5))
            measured_y = self.find_centroid(resized, col=int(expected_x + 0.5))

            assert measured_x is not None, \
                f"No impulse found at hscale={hscale}, vscale={vscale}"
            assert abs(measured_x - expected_x) < 0.15, \
                f"hscale={hscale}, vscale={vscale}: x centroid at {measured_x:.3f}, expected {expected_x:.3f}"
            assert abs(measured_y - expected_y) < 0.15, \
                f"hscale={hscale}, vscale={vscale}: y centroid at {measured_y:.3f}, expected {expected_y:.3f}"

    def test_fractional_upscale(self):
        """Test fractional scale factors."""
        width, height = 100, 20
        impulse_x, impulse_y = 50, 10

        for scale in [1.5, 2.5, 3.5]:
            img = self.create_impulse(width, height, impulse_x, impulse_y)
            resized = img.resize(scale)

            expected_x = self.expected_position(impulse_x, scale)
            measured_x = self.find_centroid(resized, row=int(self.expected_position(impulse_y, scale) + 0.5))

            assert measured_x is not None, f"No impulse found at scale={scale}"
            assert abs(measured_x - expected_x) < 0.15, \
                f"scale={scale}: centroid at {measured_x:.3f}, expected {expected_x:.3f}"

    def test_off_center_impulse(self):
        """Test impulse not at image center."""
        width, height = 100, 20
        impulse_y = 10
        scale = 2.0

        for impulse_x in [25, 33, 67, 75]:
            img = self.create_impulse(width, height, impulse_x, impulse_y)
            resized = img.resize(scale)

            expected_x = self.expected_position(impulse_x, scale)
            measured_x = self.find_centroid(resized, row=int(self.expected_position(impulse_y, scale) + 0.5))

            assert measured_x is not None, f"No impulse found at impulse_x={impulse_x}"
            assert abs(measured_x - expected_x) < 0.15, \
                f"impulse_x={impulse_x}: centroid at {measured_x:.3f}, expected {expected_x:.3f}"

    def test_downscale_preserves_position(self):
        """Test that downscaling preserves relative position."""
        width, height = 200, 40
        impulse_x, impulse_y = 100, 20
        scale = 0.5

        img = self.create_impulse(width, height, impulse_x, impulse_y)
        resized = img.resize(scale)

        # For downscale, expected position is simply scaled
        # (vips_reduce handles center sampling internally)
        expected = impulse_x * scale
        measured = self.find_centroid(resized, row=int(impulse_y * scale))

        assert measured is not None, f"No impulse found at scale={scale}"
        # Higher tolerance for downscale due to averaging effects
        assert abs(measured - expected) < 1.0, \
            f"scale={scale}: centroid at {measured:.3f}, expected {expected:.3f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
