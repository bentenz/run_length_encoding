import cv2
import json
import numpy as np

class RLECompression:
    def __init__(self, image_path):
        # Load gambar RGB
        self.image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.image is None:
            raise ValueError(f"Gagal load gambar: {image_path}")
        self.height, self.width, self.channels = self.image.shape

        # Encode per channel
        self.encoded_data = []
        for i in range(self.channels):
            channel_pixels = self.image[:, :, i].flatten().tolist()
            self.encoded_data.append(self.encode(channel_pixels))

    @staticmethod
    def encode(data):
        if not data:
            return []
        encoded = []
        current_val = data[0]
        count = 1
        for i in range(1, len(data)):
            if data[i] == current_val:
                count += 1
            else:
                encoded.append([current_val, count])
                current_val = data[i]
                count = 1
        encoded.append([current_val, count])
        return encoded

    def save_to_json(self, path):
        # JSON single-line, tapi per channel
        payload = {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "data": self.encoded_data  # sudah per channel
        }
        with open(path, 'w') as f:
            json.dump(payload, f)  # tanpa indent

# ============================
# Contoh penggunaan console
# ============================
if __name__ == "__main__":
    file_path = "myphoto.png"  # ganti sesuai nama file
    compressor = RLECompression(file_path)

    # Info singkat
    print(f"Gambar: {file_path}")
    print(f"Ukuran: {compressor.width}x{compressor.height}, Channels: {compressor.channels}")
    print(f"Jumlah channel yang di-RLE: {len(compressor.encoded_data)}")

    # Simpan ke JSON single-line
    compressor.save_to_json("compressed.json")
    print("RLE per channel disimpan ke compressed.json")
