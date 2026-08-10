class ImageTrimBatch:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "images": ("IMAGE",),
                "trim_count": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "trim_batch"
    CATEGORY = "image"

    def trim_batch(self, images, trim_count):
        # Get the batch size
        batch_size = images.shape[0]
        
        # Ensure we don't trim more images than we have
        trim_count = min(trim_count, batch_size)
        
        # If trim_count is 0 or we're trimming all images, handle edge cases
        if trim_count == 0:
            return (images,)
        elif trim_count >= batch_size:
            # Return an empty tensor with the same dimensions except batch size = 0
            # This maintains compatibility with ComfyUI's expected tensor structure
            empty_batch = images[:0]  # Creates tensor with shape (0, height, width, channels)
            return (empty_batch,)
        
        # Trim the first trim_count images
        trimmed_images = images[trim_count:]
        
        return (trimmed_images,)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "ImageTrimBatch": ImageTrimBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageTrimBatch": "Image Trim Batch"
}