from PIL import Image
import io

def process_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return {
        "message": "Image processed",
        "size": image.size
    }
 