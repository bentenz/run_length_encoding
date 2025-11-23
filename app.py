from flask import Flask, render_template, request, send_file, abort, url_for
import io
import json
from RLECompression import RLECompression

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

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
        compressor = RLECompression(file.stream)
        compressor.save_to_binary('mycompressed.rle')
        compressor.save_to_json('mycompressed.json')

    except Exception:
        abort(400, description="File yang diupload bukan gambar yang valid.")

    # --- Statistik ukuran ---
    stats = compressor.get_compression_stats()

       # Sekarang Anda bisa akses:
    original_size = stats['original_size']
    json_size = stats['json_size']
    binary_size = stats['binary_size']
    json_ratio = stats['json_ratio']  # Compression ratio JSON
    binary_ratio = stats['binary_ratio']

    # --- Preview gambar ---
    original_base64 = compressor.original_to_base64()
    decompressed_base64 = compressor.to_base64()

    original_pixels = compressor.image.flatten() if compressor.grayscale else compressor.image.reshape(-1, compressor.channels)
    sample_rows_original = []
    for i in range(min(5, compressor.height)):
        start = i * compressor.width
        end = start + compressor.width
        if compressor.grayscale:
            sample_rows_original.append(original_pixels[start:end].tolist())
        else:
            row = original_pixels[start:end, :]
            sample_rows_original.append(row.tolist())


    sample_rows_rle = []
    encoded = compressor.encoded_data 
    for i in range(len(encoded)):
        sample_rows_rle.append(encoded[i])  

    chart_data = json.dumps([original_size, json_size, binary_size])

    return render_template(
        "result.html",
        original_base64=original_base64,
        decompressed_base64=decompressed_base64,
        original_size=original_size,
        compressed_size_json=json_size,
        compressed_size_bin=binary_size,
        compression_ratio_json=json_ratio,
        compression_ratio_bin=binary_ratio,
        download_url_json=url_for('download_rle'),
        download_url_bin=url_for('download_rlebin'),
        chart_data=chart_data,
        rle_result=sample_rows_rle,
        sample_rows=sample_rows_original
    )

# ===================== DOWNLOAD ROUTES =====================
@app.route('/download')
def download_rle():
    try:
        return send_file('mycompressed.json', as_attachment=True, download_name='mycompressed.json', mimetype='application/json')
    except Exception:
        abort(404, description="File RLE belum tersedia.")

@app.route('/download_bin')
def download_rlebin():
    try:
        return send_file('mycompressed.rle', as_attachment=True, download_name='mycompressed.rlebin', mimetype='application/octet-stream')
    except Exception:
        abort(404, description="File RLE Binary belum tersedia.")

# ===================== MAIN =====================
if __name__ == "__main__":
    app.run(debug=True)
