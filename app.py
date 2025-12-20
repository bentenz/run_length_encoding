from flask import Flask, render_template, request, send_file, abort, url_for
import io
import json
from RLECompression import RLECompression

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

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

    stats = compressor.get_compression_stats()
    original_size = stats['original_size']
    json_size = stats['json_size']
    binary_size = stats['binary_size']
    json_ratio = stats['json_ratio']
    binary_ratio = stats['binary_ratio']
    json_reduction = stats['json_reduction']
    binary_reduction = stats['binary_reduction']
    gray_json_size = stats.get('gray_json_size', 0)
    gray_binary_size = stats.get('gray_binary_size', 0)
    gray_json_reduction = stats.get('gray_json_reduction', 0)
    gray_binary_reduction = stats.get('gray_binary_reduction', 0)

    original_base64 = compressor.original_to_base64()
    decompressed_base64 = compressor.to_base64()
    grayscale_base64 = compressor.grayscale_to_base64()

    max_pixels = 10000
    if compressor.grayscale:
        sample_blue = compressor.image.flatten()[:max_pixels].tolist()
        sample_green = []
        sample_red = []
    else:
        sample_blue = compressor.image[:, :, 0].flatten()[:max_pixels].tolist()
        sample_green = compressor.image[:, :, 1].flatten()[:max_pixels].tolist()
        sample_red = compressor.image[:, :, 2].flatten()[:max_pixels].tolist()

    rle_blue = compressor.encoded_data[0] if len(compressor.encoded_data) > 0 else []
    rle_green = compressor.encoded_data[1] if len(compressor.encoded_data) > 1 else []
    rle_red = compressor.encoded_data[2] if len(compressor.encoded_data) > 2 else []
    rle_grayscale = compressor.get_grayscale_encoded()

    chart_data = json.dumps([original_size, json_size, binary_size, gray_json_size, gray_binary_size])

    return render_template(
        "result.html",
        original_base64=original_base64,
        decompressed_base64=decompressed_base64,
        grayscale_base64=grayscale_base64,
        original_size=original_size,
        compressed_size_json=json_size,
        compressed_size_bin=binary_size,
        compression_ratio_json=json_reduction,
        compression_ratio_bin=binary_reduction,
        gray_json_size=gray_json_size,
        gray_binary_size=gray_binary_size,
        gray_json_reduction=gray_json_reduction,
        gray_binary_reduction=gray_binary_reduction,
        width=compressor.width,
        height=compressor.height,
        download_url_json=url_for('download_rle'),
        download_url_bin=url_for('download_rlebin'),
        chart_data=chart_data,
        rle_blue=rle_blue,
        rle_green=rle_green,
        rle_red=rle_red,
        rle_grayscale=rle_grayscale,
        sample_blue=sample_blue,
        sample_green=sample_green,
        sample_red=sample_red
    )

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

if __name__ == "__main__":
    app.run(debug=True)
