def extract_patches(
    input_tensor,
    target_tensor,
    patch_height,
    patch_width,
    stride,
):
    _, image_height, image_width = input_tensor.shape
    patches = []

    for row_index in range(0, image_height - patch_height + 1, stride):
        for column_index in range(0, image_width - patch_width + 1, stride):
            input_patch = input_tensor[
                :,
                row_index : row_index + patch_height,
                column_index : column_index + patch_width,
            ]

            target_patch = target_tensor[
                :,
                row_index : row_index + patch_height,
                column_index : column_index + patch_width,
            ]

            patches.append(
                (
                    input_patch,
                    target_patch,
                    row_index,
                    column_index,
                )
            )

    return patches
