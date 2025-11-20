from flask import Flask, render_template, request, send_file, abort, url_for
from PIL import Image
import io
import json
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# ===================== RLE JSON =====================
def rle_encode_row(row):
    encoded = []
    count = 1
    for i in range(1, len(row)):
        if row[i] == row[i-1]:
            count += 1
        else:
            encoded.append([row[i-1], count])
            count = 1
    encoded.append([row[-1], count])
    return encoded

def compress_image_rle(image):
    img = image.convert("L")  # grayscale
    pixels = list(img.getdata())
    width, height = img.size
    rle_data = []
    index = 0
    for _ in range(height):
        row = pixels[index:index+width]
        rle_row = rle_encode_row(row)
        rle_data.append(rle_row)
        index += width
    return rle_data, width, height

def rle_decode(rle_data, width, height):
    pixels = []
    for row in rle_data:
        for value, count in row:
            pixels.extend([value]*count)
    img = Image.new("L", (width, height))
    img.putdata(pixels)
    return img

# ===================== RLE BINARY =====================
def rle_encode_row_binary(row):
    encoded = bytearray()
    count = 1
    for i in range(1, len(row)):
        if row[i] == row[i-1] and count < 255:
            count += 1
        else:
            encoded.append(row[i-1])
            encoded.append(count)
            count = 1
    encoded.append(row[-1])
    encoded.append(count)
    return encoded

def compress_image_rle_binary(image):
    img = image.convert("L")
    pixels = list(img.getdata())
    width, height = img.size
    rle_bin = bytearray()
    index = 0
    for _ in range(height):
        row = pixels[index:index+width]
        rle_bin.extend(rle_encode_row_binary(row))
        index += width
    return rle_bin, width, height

def rle_decode_binary(rle_bin, width, height):
    pixels = []
    for i in range(0, len(rle_bin), 2):
        value = rle_bin[i]
        count = rle_bin[i+1]
        pixels.extend([value]*count)
    img = Image.new("L", (width, height))
    img.putdata(pixels)
    return img

# ===================== UTILITY =====================
def image_to_base64(img, resize=None):
    if resize:
        img = img.resize(resize)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ===================== ROUTES =====================
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/compress', methods=['POST'])
def compress():
    file = request.files.get('image')
    if not file:
        abort(400, description="File gambar tidak ditemukan.")
    try:
        img = Image.open(file.stream)
    except Exception:
        abort(400, description="File yang diupload bukan gambar yang valid.")

    # --- RLE JSON ---
    rle_result, width, height = compress_image_rle(img)
    output_obj = {"width": width, "height": height, "data": rle_result}
    output_json = json.dumps(output_obj)

    # --- RLE Binary ---
    rle_bin, _, _ = compress_image_rle_binary(img)

    # --- Statistik ukuran ---
    file.stream.seek(0)
    original_size = len(file.read())
    compressed_size_json = len(output_json.encode('utf-8'))
    compressed_size_bin = len(rle_bin)
    compression_ratio_json = round(original_size / compressed_size_json, 2)
    compression_ratio_bin = round(original_size / compressed_size_bin, 2)

    # --- Preview gambar ---
    original_base64 = image_to_base64(img, resize=(300, int(300*height/width)))
    decompressed_img = rle_decode(rle_result, width, height)
    decompressed_base64 = image_to_base64(decompressed_img, resize=(300, int(300*height/width)))

    # --- Sample 5 baris pixel asli ---
    sample_rows = []
    for i in range(min(5, height)):
        index_start = i * width
        index_end = index_start + width
        sample_rows.append(list(img.getdata())[index_start:index_end])

    # --- Simpan file RLE ---
    with open('compressed.rle', 'w', encoding='utf-8') as f:
        f.write(output_json)
    with open('compressed.rlebin', 'wb') as f:
        f.write(rle_bin)

    chart_data = json.dumps([original_size, compressed_size_json, compressed_size_bin])

    return render_template(
        "result.html",
        original_base64=original_base64,
        decompressed_base64=decompressed_base64,
        original_size=original_size,
        compressed_size_json=compressed_size_json,
        compressed_size_bin=compressed_size_bin,
        compression_ratio_json=compression_ratio_json,
        compression_ratio_bin=compression_ratio_bin,
        download_url_json=url_for('download_rle'),
        download_url_bin=url_for('download_rlebin'),
        chart_data=chart_data,
        rle_result=rle_result,
        sample_rows=sample_rows
    )

# ===================== DOWNLOAD ROUTES =====================
@app.route('/download')
def download_rle():
    try:
        return send_file('compressed.rle', as_attachment=True, download_name='compressed.rle', mimetype='application/json')
    except Exception:
        abort(404, description="File RLE belum tersedia.")

@app.route('/download_bin')
def download_rlebin():
    try:
        return send_file('compressed.rlebin', as_attachment=True, download_name='compressed.rlebin', mimetype='application/octet-stream')
    except Exception:
        abort(404, description="File RLE Binary belum tersedia.")

# ===================== MAIN =====================
if __name__ == "__main__":
    app.run(debug=True)
