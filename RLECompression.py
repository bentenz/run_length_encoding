import cv2
import numpy as np
import struct
import json
import io
import base64
import os

class RLECompression:
    def __init__(self, file, grayscale=False):
        self.file = file
        self.grayscale = grayscale

        self.file.seek(0)
        file_bytes = self.file.read()
        np_img = np.frombuffer(file_bytes, np.uint8)

        if grayscale:
            self.image = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
            self.height, self.width = self.image.shape
            self.channels = 1
        else:
            self.image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            self.height, self.width, self.channels = self.image.shape

        if self.image is None:
            raise ValueError("Gagal decode gambar. Pastikan formatnya benar.")

        self.encoded_data = []
        if grayscale:
            self.pixels = self.image.flatten().tolist()
            self.encoded_data.append(self.encode(self.pixels))
        else:
            self.pixels = []
            for i in range(self.channels):
                channel_pixels = self.image[:, :, i].flatten().tolist()
                self.pixels.append(channel_pixels)
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
                encoded.append((current_val, count))
                current_val = data[i]
                count = 1
        encoded.append((current_val, count))
        return encoded

    @staticmethod
    def decode(encoded):
        pixels = []
        for value, count in encoded:
            pixels.extend([value] * count)
        return pixels

    def save_to_binary(self, output_path):
        with open(output_path, 'wb') as f:
            f.write(struct.pack('<III', self.height, self.width, self.channels))
            for channel in self.encoded_data:
                f.write(struct.pack('<I', len(channel)))
            for channel in self.encoded_data:
                for value, count in channel:
                    f.write(struct.pack('<BI', value, count))

    def save_to_json(self, output_path):
        payload = {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "grayscale": self.grayscale,
            "data": [
                [[v, c] for (v, c) in channel]
                for channel in self.encoded_data
            ]
        }
        with open(output_path, 'w') as f:
            json.dump(payload, f)

    def load_from_json(self, input_path):
        with open(input_path, 'r') as f:
            payload = json.load(f)
        self.width = payload["width"]
        self.height = payload["height"]
        self.channels = payload["channels"]
        self.grayscale = payload.get("grayscale", False)
        self.encoded_data = [
            [(v, c) for (v, c) in channel]
            for channel in payload["data"]
        ]

    def get_image(self):
        if self.grayscale:
            pixels = self.decode(self.encoded_data[0])
            return np.array(pixels, dtype=np.uint8).reshape(self.height, self.width)
        decoded_channels = []
        for chan in self.encoded_data:
            pixels = self.decode(chan)
            decoded_channels.append(np.array(pixels, dtype=np.uint8))
        img = np.stack([ch.reshape(self.height, self.width) for ch in decoded_channels], axis=-1)
        return img

    def original_to_base64(self):
        _, buffer = cv2.imencode('.png', self.image)
        return base64.b64encode(buffer).decode('utf-8')

    def to_base64(self):
        img = self.get_image()
        _, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode('utf-8')

    def get_compression_stats(self):
        """
        Statistik kompresi dengan persentase pengurangan ukuran file
        """
        # Ukuran file asli
        self.file.seek(0)
        original_bytes = self.file.read()
        original_size = len(original_bytes)

        # Ukuran JSON RLE
        payload = {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "grayscale": self.grayscale,
            "data": [
                [[v, c] for (v, c) in channel]
                for channel in self.encoded_data
            ]
        }
        json_bytes = json.dumps(payload).encode('utf-8')
        json_size = len(json_bytes)

        # Ukuran Binary RLE
        binary_buffer = io.BytesIO()
        binary_buffer.write(struct.pack('<III', self.height, self.width, self.channels))
        for channel in self.encoded_data:
            binary_buffer.write(struct.pack('<I', len(channel)))
        for channel in self.encoded_data:
            for value, count in channel:
                binary_buffer.write(struct.pack('<BI', value, count))
        binary_size = len(binary_buffer.getvalue())

        # Hitung persentase pengurangan ukuran
        json_reduction = round((original_size - json_size) / original_size * 100, 2)
        binary_reduction = round((original_size - binary_size) / original_size * 100, 2)

        # Hitung compression ratio juga
        json_ratio = round(original_size / json_size, 2) if json_size > 0 else 0
        binary_ratio = round(original_size / binary_size, 2) if binary_size > 0 else 0

        stats = {
            "original_size": original_size,
            "json_size": json_size,
            "binary_size": binary_size,
            "json_reduction": json_reduction,  # Persentase pengurangan
            "binary_reduction": binary_reduction,  # Persentase pengurangan
            "json_ratio": json_ratio,  # Compression ratio
            "binary_ratio": binary_ratio  # Compression ratio
        }

        return stats
