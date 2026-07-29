import extensibility

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.utils.coord import Coord, coord_to_index_list

from extensibility import InputTensor, OutputTensor, foreach


@extensibility.register("collie_box_postprocess")
struct CollieBoxPostprocess:
    """Transform, clip, blend, and validate one tracker bounding box."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=DType.float32, rank=2, ...],
        record: InputTensor[dtype=DType.float32, rank=2, ...],
        config: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        @parameter
        @always_inline
        def compute[width: Int](idx: Coord) -> SIMD[DType.float32, width]:
            var indices = coord_to_index_list(idx)
            var row = indices[0]
            var column = indices[1]

            var x = record[row, 0][0]
            var y = record[row, 1][0]
            var box_width = record[row, 2][0]
            var box_height = record[row, 3][0]
            var a00 = record[row, 4][0]
            var a01 = record[row, 5][0]
            var a02 = record[row, 6][0]
            var a10 = record[row, 7][0]
            var a11 = record[row, 8][0]
            var a12 = record[row, 9][0]
            var frame_width = record[row, 10][0]
            var frame_height = record[row, 11][0]

            var minimum_size = config[0][0]
            var minimum_scale = config[1][0]
            var maximum_scale = config[2][0]
            var maximum_step_fraction = config[3][0]
            var new_box_weight = config[4][0]

            var x2 = x + box_width
            var y2 = y + box_height
            var tx0 = a00 * x + a01 * y + a02
            var ty0 = a10 * x + a11 * y + a12
            var tx1 = a00 * x2 + a01 * y + a02
            var ty1 = a10 * x2 + a11 * y + a12
            var tx2 = a00 * x2 + a01 * y2 + a02
            var ty2 = a10 * x2 + a11 * y2 + a12
            var tx3 = a00 * x + a01 * y2 + a02
            var ty3 = a10 * x + a11 * y2 + a12

            var left = min(min(tx0, tx1), min(tx2, tx3))
            var top = min(min(ty0, ty1), min(ty2, ty3))
            var right = max(max(tx0, tx1), max(tx2, tx3))
            var bottom = max(max(ty0, ty1), max(ty2, ty3))

            var zero: Scalar[DType.float32] = 0.0
            var clipped_left = max(zero, min(frame_width, left))
            var clipped_top = max(zero, min(frame_height, top))
            var clipped_right = max(zero, min(frame_width, right))
            var clipped_bottom = max(zero, min(frame_height, bottom))
            var clipped_width = clipped_right - clipped_left
            var clipped_height = clipped_bottom - clipped_top

            var scale = sqrt(a00 * a00 + a10 * a10)
            var old_center_x = x + box_width * 0.5
            var old_center_y = y + box_height * 0.5
            var new_center_x = clipped_left + clipped_width * 0.5
            var new_center_y = clipped_top + clipped_height * 0.5
            var delta_x = new_center_x - old_center_x
            var delta_y = new_center_y - old_center_y
            var center_step = sqrt(delta_x * delta_x + delta_y * delta_y)
            var frame_diagonal = sqrt(
                frame_width * frame_width + frame_height * frame_height
            )

            var valid: Scalar[DType.float32] = 1.0
            if scale < minimum_scale or scale > maximum_scale:
                valid = 0.0
            if clipped_width < minimum_size or clipped_height < minimum_size:
                valid = 0.0
            if center_step > frame_diagonal * maximum_step_fraction:
                valid = 0.0

            var previous_weight = 1.0 - new_box_weight
            var blended_x = (
                previous_weight * x + new_box_weight * clipped_left
            )
            var blended_y = (
                previous_weight * y + new_box_weight * clipped_top
            )
            var blended_width = (
                previous_weight * box_width
                + new_box_weight * clipped_width
            )
            var blended_height = (
                previous_weight * box_height
                + new_box_weight * clipped_height
            )

            var result = center_step
            if column == 0:
                result = blended_x
            elif column == 1:
                result = blended_y
            elif column == 2:
                result = blended_width
            elif column == 3:
                result = blended_height
            elif column == 4:
                result = valid
            elif column == 5:
                result = scale
            return SIMD[DType.float32, width](result)

        foreach[compute, target=target, simd_width=1](output, ctx)
