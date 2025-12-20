import cv2
import numpy as np
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
        
        arr = np.array(data, dtype=np.uint8)
        n = len(arr)
        
        if n == 0:
            return []
        
        changes = np.where(arr[1:] != arr[:-1])[0] + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [n]])
        counts = ends - starts
        values = arr[starts]
        
        return list(zip(values.tolist(), counts.tolist()))

    @staticmethod
    def decode(encoded):
        pixels = []
        for value, count in encoded:
            pixels.extend([value] * count)
        return pixels

    def save_to_binary(self, output_path):
        with open(output_path, 'wb') as f:
            f.write(self.height.to_bytes(4, 'little'))
            f.write(self.width.to_bytes(4, 'little'))
            f.write(self.channels.to_bytes(4, 'little'))
            for channel in self.encoded_data:
                run_count = sum(1 + (count - 1) // 65535 for _, count in channel)
                f.write(run_count.to_bytes(4, 'little'))
            for channel in self.encoded_data:
                for value, count in channel:
                    while count > 65535:
                        f.write(bytes([value, 255]) + (65535).to_bytes(2, 'little'))
                        count -= 65535
                    if count <= 254:
                        f.write(bytes([value, count]))
                    else:
                        f.write(bytes([value, 255]) + count.to_bytes(2, 'little'))

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

    def grayscale_to_base64(self):
        if self.grayscale:
            gray_img = self.image
        else:
            gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, buffer = cv2.imencode('.png', gray_img)
        return base64.b64encode(buffer).decode('utf-8')

    def get_grayscale_encoded(self):
        if self.grayscale:
            return self.encoded_data[0]
        else:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            gray_pixels = gray_image.flatten().tolist()
            return self.encode(gray_pixels)

    def get_compression_stats(self):
        self.file.seek(0)
        original_bytes = self.file.read()
        file_size = len(original_bytes)
        raw_size = self.height * self.width * self.channels

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

        binary_buffer = io.BytesIO()
        binary_buffer.write(self.height.to_bytes(4, 'little'))
        binary_buffer.write(self.width.to_bytes(4, 'little'))
        binary_buffer.write(self.channels.to_bytes(4, 'little'))
        for channel in self.encoded_data:
            run_count = sum(1 + (count - 1) // 65535 for _, count in channel)
            binary_buffer.write(run_count.to_bytes(4, 'little'))
        for channel in self.encoded_data:
            for value, count in channel:
                while count > 65535:
                    binary_buffer.write(bytes([value, 255]) + (65535).to_bytes(2, 'little'))
                    count -= 65535
                if count <= 254:
                    binary_buffer.write(bytes([value, count]))
                else:
                    binary_buffer.write(bytes([value, 255]) + count.to_bytes(2, 'little'))
        binary_size = len(binary_buffer.getvalue())

        json_reduction = round((file_size - json_size) / file_size * 100, 2)
        binary_reduction = round((file_size - binary_size) / file_size * 100, 2)
        json_ratio = round(file_size / json_size, 2) if json_size > 0 else 0
        binary_ratio = round(file_size / binary_size, 2) if binary_size > 0 else 0

        stats = {
            "original_size": file_size,
            "file_size": file_size,
            "raw_size": raw_size,
            "json_size": json_size,
            "binary_size": binary_size,
            "json_reduction": json_reduction,
            "binary_reduction": binary_reduction,
            "json_ratio": json_ratio,
            "binary_ratio": binary_ratio
        }

        if not self.grayscale:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            gray_pixels = gray_image.flatten().tolist()
            gray_encoded = self.encode(gray_pixels)
            
            gray_payload = {
                "width": self.width,
                "height": self.height,
                "channels": 1,
                "grayscale": True,
                "data": [[[v, c] for (v, c) in gray_encoded]]
            }
            gray_json_size = len(json.dumps(gray_payload).encode('utf-8'))
            
            gray_binary_buffer = io.BytesIO()
            gray_binary_buffer.write(self.height.to_bytes(4, 'little'))
            gray_binary_buffer.write(self.width.to_bytes(4, 'little'))
            gray_binary_buffer.write((1).to_bytes(4, 'little'))
            run_count = sum(1 + (count - 1) // 65535 for _, count in gray_encoded)
            gray_binary_buffer.write(run_count.to_bytes(4, 'little'))
            for value, count in gray_encoded:
                while count > 65535:
                    gray_binary_buffer.write(bytes([value, 255]) + (65535).to_bytes(2, 'little'))
                    count -= 65535
                if count <= 254:
                    gray_binary_buffer.write(bytes([value, count]))
                else:
                    gray_binary_buffer.write(bytes([value, 255]) + count.to_bytes(2, 'little'))
            gray_binary_size = len(gray_binary_buffer.getvalue())
            
            stats["gray_json_size"] = gray_json_size
            stats["gray_binary_size"] = gray_binary_size
            stats["gray_json_reduction"] = round((file_size - gray_json_size) / file_size * 100, 2)
            stats["gray_binary_reduction"] = round((file_size - gray_binary_size) / file_size * 100, 2)

        return stats
