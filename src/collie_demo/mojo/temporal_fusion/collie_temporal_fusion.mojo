import extensibility

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.utils.coord import Coord, coord_to_index_list

from extensibility import InputTensor, OutputTensor, foreach


@extensibility.register("collie_temporal_fusion")
struct CollieTemporalFusion:
    """Fuse YOLO measurements with bounded motion prediction for three fruits."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=DType.float32, rank=2, ...],
        records: InputTensor[dtype=DType.float32, rank=2, ...],
        config: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        @parameter
        @always_inline
        def compute[width: Int](idx: Coord) -> SIMD[DType.float32, width]:
            var indices = coord_to_index_list(idx)
            var row = indices[0]
            var column = indices[1]

            var previous_x = records[row, 0][0]
            var previous_y = records[row, 1][0]
            var previous_width = records[row, 2][0]
            var previous_height = records[row, 3][0]
            var velocity_x = records[row, 4][0]
            var velocity_y = records[row, 5][0]
            var velocity_width = records[row, 6][0]
            var velocity_height = records[row, 7][0]
            var measurement_x = records[row, 8][0]
            var measurement_y = records[row, 9][0]
            var measurement_width = records[row, 10][0]
            var measurement_height = records[row, 11][0]
            var frame_width = records[row, 12][0]
            var frame_height = records[row, 13][0]
            var zero: Scalar[DType.float32] = 0.0
            var one: Scalar[DType.float32] = 1.0
            var minimum_delta: Scalar[DType.float32] = 0.005
            var maximum_delta: Scalar[DType.float32] = 0.25
            var delta_s = max(
                minimum_delta,
                min(maximum_delta, records[row, 14][0]),
            )
            var confidence = max(zero, min(one, records[row, 15][0]))
            var measurement_valid = records[row, 16][0] >= 0.5
            var active = records[row, 17][0] >= 0.5

            var minimum_alpha = config[0][0]
            var maximum_alpha = config[1][0]
            var velocity_momentum = config[2][0]
            var prediction_decay = config[3][0]
            var maximum_step_fraction = config[4][0]
            var minimum_size = config[5][0]

            var predicted_x = previous_x + velocity_x * delta_s
            var predicted_y = previous_y + velocity_y * delta_s
            var predicted_width = previous_width + velocity_width * delta_s
            var predicted_height = previous_height + velocity_height * delta_s

            var predicted_center_x = predicted_x + predicted_width * 0.5
            var predicted_center_y = predicted_y + predicted_height * 0.5
            var measurement_center_x = measurement_x + measurement_width * 0.5
            var measurement_center_y = measurement_y + measurement_height * 0.5
            var center_dx = measurement_center_x - predicted_center_x
            var center_dy = measurement_center_y - predicted_center_y
            var center_step = sqrt(center_dx * center_dx + center_dy * center_dy)
            var frame_diagonal = sqrt(
                frame_width * frame_width + frame_height * frame_height
            )

            var use_measurement = (
                active
                and measurement_valid
                and measurement_width >= minimum_size
                and measurement_height >= minimum_size
                and center_step <= frame_diagonal * maximum_step_fraction
            )
            var alpha: Scalar[DType.float32] = 0.0
            var fused_x = predicted_x
            var fused_y = predicted_y
            var fused_width = predicted_width
            var fused_height = predicted_height
            var next_velocity_x = velocity_x * prediction_decay
            var next_velocity_y = velocity_y * prediction_decay
            var next_velocity_width = velocity_width * prediction_decay
            var next_velocity_height = velocity_height * prediction_decay

            if use_measurement:
                alpha = (
                    minimum_alpha
                    + confidence * (maximum_alpha - minimum_alpha)
                )
                fused_x = predicted_x + alpha * (measurement_x - predicted_x)
                fused_y = predicted_y + alpha * (measurement_y - predicted_y)
                fused_width = (
                    predicted_width
                    + alpha * (measurement_width - predicted_width)
                )
                fused_height = (
                    predicted_height
                    + alpha * (measurement_height - predicted_height)
                )
                var observed_velocity_x = (
                    (measurement_x - previous_x) / delta_s
                )
                var observed_velocity_y = (
                    (measurement_y - previous_y) / delta_s
                )
                var observed_velocity_width = (
                    (measurement_width - previous_width) / delta_s
                )
                var observed_velocity_height = (
                    (measurement_height - previous_height) / delta_s
                )
                var observed_weight = 1.0 - velocity_momentum
                next_velocity_x = (
                    velocity_momentum * velocity_x
                    + observed_weight * observed_velocity_x
                )
                next_velocity_y = (
                    velocity_momentum * velocity_y
                    + observed_weight * observed_velocity_y
                )
                next_velocity_width = (
                    velocity_momentum * velocity_width
                    + observed_weight * observed_velocity_width
                )
                next_velocity_height = (
                    velocity_momentum * velocity_height
                    + observed_weight * observed_velocity_height
                )
            else:
                var previous_center_x = previous_x + previous_width * 0.5
                var previous_center_y = previous_y + previous_height * 0.5
                var prediction_dx = predicted_center_x - previous_center_x
                var prediction_dy = predicted_center_y - previous_center_y
                center_step = sqrt(
                    prediction_dx * prediction_dx
                    + prediction_dy * prediction_dy
                )

            var clipped_left = max(zero, min(frame_width, fused_x))
            var clipped_top = max(zero, min(frame_height, fused_y))
            var clipped_right = max(
                zero,
                min(frame_width, fused_x + fused_width),
            )
            var clipped_bottom = max(
                zero,
                min(frame_height, fused_y + fused_height),
            )
            var clipped_width = max(zero, clipped_right - clipped_left)
            var clipped_height = max(zero, clipped_bottom - clipped_top)
            var valid: Scalar[DType.float32] = 0.0
            if (
                active
                and frame_width > 0.0
                and frame_height > 0.0
                and clipped_width >= minimum_size
                and clipped_height >= minimum_size
            ):
                valid = 1.0

            var result = center_step
            if column == 0:
                result = clipped_left
            elif column == 1:
                result = clipped_top
            elif column == 2:
                result = clipped_width
            elif column == 3:
                result = clipped_height
            elif column == 4:
                result = next_velocity_x
            elif column == 5:
                result = next_velocity_y
            elif column == 6:
                result = next_velocity_width
            elif column == 7:
                result = next_velocity_height
            elif column == 8:
                result = valid
            elif column == 9:
                result = 0.0
                if use_measurement:
                    result = 1.0
            elif column == 10:
                result = alpha
            return SIMD[DType.float32, width](result)

        foreach[compute, target=target, simd_width=1](output, ctx)
