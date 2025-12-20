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

    @staticmethod
    def decode(encoded):
        pixels = []
        for value, count in encoded:
            pixels.extend([value] * count)
        return pixels

    def get_image(self):
        decoded_channels = [np.array(self.decode(ch), dtype=np.uint8) for ch in self.encoded_data]
        img = np.stack([ch.reshape(self.height, self.width) for ch in decoded_channels], axis=-1)
        return img

    def save_to_json(self, path):
        payload = {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "data": self.encoded_data  # per channel, single line
        }
        with open(path, 'w') as f:
            json.dump(payload, f)

# ============================
# MAIN CONSOLE
# ============================
if __name__ == "__main__":
    file_path = "image/PhotoTraces_Free_RAW_Photos_01_Manhattan_Skyline.dng"  # ganti sesuai nama file
    compressor = RLECompression(file_path)

    # Info singkat
    print(f"Gambar: {file_path}")
    print(f"Ukuran: {compressor.width}x{compressor.height}, Channels: {compressor.channels}")
    print(f"Jumlah channel yang di-RLE: {len(compressor.encoded_data)}")

    # Preview beberapa pasangan pertama per channel
    for idx, ch in enumerate(compressor.encoded_data):
        print(f"\n--- Channel {idx} (RLE preview 20 pasangan pertama) ---")
        for pair in ch[:20]:
            print(pair, end=' ')
        if len(ch) > 20:
            print(f"... ({len(ch)} pasangan total)")

    # Simpan ke JSON
    compressor.save_to_json("compressed.json")
    print("\nRLE per channel disimpan ke compressed.json")

    # Dekompres dan verifikasi (opsional)
    decompressed_img = compressor.get_image()
    print(f"Gambar dekompresi berhasil, shape: {decompressed_img.shape}")
