import qrcode
from PIL import Image

# The URL you want to encode
url = "https://192.168.8.11:8000"

# Generate QR code
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=20,  # Controls the size of the QR code
    border=4,     # Standard border size
)
qr.add_data(url)
qr.make(fit=True)

# Create the image (perfect square)
img = qr.make_image(fill_color="black", back_color="white").convert('RGB')

# Save as PNG
img.save("qr_code.png")
print("QR code saved as qr_code.png")