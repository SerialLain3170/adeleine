import io
import argparse
import time
import base64
import numpy as np
import cv2 as cv

from PIL import Image
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, jsonify
from reference.inference import ReferenceInferer
from atari.inference import AtariInferer
from flatten.inference import FlatInferer
from point.inference import PointInferer


def main(args):
    app = Flask(__name__,
                static_url_path="/static",
                static_folder="./static")

    if args.ref is not None:
        ref_infer = ReferenceInferer(str(args.ref))
    if args.scribble is not None:
        atari_infer = AtariInferer(str(args.atari))
    if args.flat is not None:
        flat_infer = FlatInferer(str(args.flat))
    if args.point is not None:
        point_infer = PointInferer(str(args.point))

    title = args.title
    ref_base = args.ref_base
    scrib_base = args.scrib_base

    def resize(img: np.array, limit: int) -> (np.array, int, int):
        h, w = img.shape[0], img.shape[1]
        if w > limit:
            scale = float(limit / w)
            h = int(h * scale)
            w = limit

            img = cv.resize(img, (w, h), interpolation=cv.INTER_LANCZOS4)

        return img, h, w

    def decode(stream, limit: int) -> (np.array, int, int):
        array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv.imdecode(array, 1)
        img, h, w = resize(img, limit)

        return img, h, w

    def base64decode(json_data, limit: int) -> (np.array, int, int):
        json_data = json_data.replace("data:image/png;base64,", "")
        json_data = json_data.replace(" ", "+")

        img_binary = base64.b64decode(json_data)
        img = np.frombuffer(img_binary, dtype=np.uint8)
        img = cv.imdecode(img, cv.IMREAD_COLOR)
        img, h, w = resize(img, limit)

        return img, h, w

    def base64encode(imgarray: np.array):
        pillow_object = Image.fromarray(imgarray.astype(np.uint8))
        byte = io.BytesIO()
        pillow_object.save(byte, 'PNG')

        byte = byte.getvalue()
        base64_encoded = base64.b64encode(byte)
        base64_encoded = base64_encoded.decode('ascii')

        return base64_encoded

    @app.route('/')
    def index():
        return render_template('index.html', title=title)

    @app.route('/reference', methods=['GET', 'POST'])
    def ref():
        if request.method == 'POST':
            line, h, w = base64decode(request.json['image_line'], limit=ref_base)
            ref, _, _ = base64decode(request.json['image_ref'], limit=ref_base)

            y, line, ref = ref_infer(line, ref)
            y = base64encode(y)
            return jsonify({'data': y, 'h': h, 'w': w})

        else:
            y = 255 * np.ones((ref_base, ref_base, 3)).astype(np.uint8)
            y = base64encode(y)

            return render_template('reference.html', title=title, y=y, h=ref_base, w=ref_base)

    @app.route('/reference/upload', methods=['GET', 'POST'])
    def ref_upload():
        if request.method == 'POST':
            line, h, w = base64decode(request.json['image'], limit=ref_base)
            line = line[:, :, ::-1]
            line = base64encode(line)

            return jsonify({'data': line, 'h': h, 'w': w})

        elif request.method == 'GET':
            return redirect('/reference')

    @app.route('/scribble', methods=["GET", "POST"])
    def scribble():
        if request.method == "POST":
            atari, h, w = base64decode(request.json["data"], limit=scrib_base)
            filename = request.json["name"].split("/")[-1]
            line = cv.imread("static/" + filename)

            y, line, atari = atari_infer(line, atari)
            y = base64encode(y)
            return jsonify({"data": y, "h": h, "w": w})
        else:
            y = 255 * np.ones((scrib_base, scrib_base, 3)).astype(np.uint8)
            y = base64encode(y)
            return render_template('scribble.html', title=title, scrib="first.png", y=y, h=scrib_base, w=scrib_base)

    @app.route('/scribble/upload', methods=['POST'])
    def scribble_upload():
        if request.files['image_line']:
            stream = request.files['image_line'].stream
            line, h, w = decode(stream, limit=scrib_base)
            y = 255 * np.ones((h, w, 3)).astype(np.uint8)
            y = base64encode(y)
            new_name = "tmp_" + str(time.time()) + ".png"
            cv.imwrite("static/" + new_name, line)

        return render_template('scribble.html', title=title, scrib=new_name, y=y, h=str(h), w=str(w))

    @app.route('/flatten', methods=['GET', 'POST'])
    def flatten():
        if request.method == 'POST':
            atari, h, w = base64decode(request.json['image'], limit=scrib_base)
            filename = request.json['name'].split('/')[-1]
            line = cv.imread('static/' + filename)

            y, line, atari = flat_infer(line, atari)
            y = base64encode(y)
            return jsonify({'data': y, 'h': h, 'w': w})

        elif request.method == 'GET':
            y = 255 * np.ones((scrib_base, scrib_base, 3)).astype(np.uint8)
            y = base64encode(y)
            return render_template('flatten.html', title=title, scrib='first.png', y=y, h=scrib_base, w=scrib_base)

    @app.route('/flatten/upload', methods=['POST'])
    def flatten_upload():
        if request.method == 'POST':
            stream = request.files['image_line'].stream
            line, h, w = decode(stream, limit=scrib_base)
            y = 255 * np.ones((h, w, 3)).astype(np.uint8)
            y = base64encode(y)
            new_name = 'tmp_' + str(time.time()) + '.png'
            cv.imwrite('static/' + new_name, line)

            return render_template('flatten.html', title=title, scrib=new_name, y=y, h=str(h), w=str(w))

        elif request.method == 'GET':
            return redirect('/flatten')

    @app.route('/point', methods=['GET', 'POST'])
    def point():
        if request.method == 'POST':
            atari, h, w = base64decode(request.json['image'], limit=scrib_base)
            point, _, _ = base64decode(request.json['point'], limit=scrib_base)
            place, _, _ = base64decode(request.json['place'], limit=scrib_base)
            filename = request.json['name'].split('/')[-1]
            line = cv.imread('static/' + filename)

            y = point_infer(line, point, place)
            y = base64encode(y)
            return jsonify({'data': y, 'h': h, 'w': w})

        elif request.method == 'GET':
            y = 255 * np.ones((scrib_base, scrib_base, 3)).astype(np.uint8)
            y = base64encode(y)
            return render_template('point.html', title=title, scrib='first.png', y=y, h=scrib_base, w=scrib_base)

    @app.route('/point/upload', methods=['POST'])
    def point_upload():
        if request.method == 'POST':
            stream = request.files['image_line'].stream
            line, h, w = decode(stream, limit=scrib_base)
            y = 255 * np.ones((h, w, 3)).astype(np.uint8)
            y = base64encode(y)
            new_name = 'tmp_' + str(time.time()) + '.png'
            cv.imwrite('static/' + new_name, line)

            return render_template('point.html', title=title, scrib=new_name, y=y, h=str(h), w=str(w))

        elif request.method == 'GET':
            return redirect('/point')

    app.debug = True
    app.run(host="0.0.0.0", port=34848)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adeleine")
    parser.add_argument("--ref", type=Path, help="pretrained file for reference")
    parser.add_argument("--scribble", type=Path, help="pretrained file for scribble")
    parser.add_argument("--flat", type=Path, help="pretrained file for flat")
    parser.add_argument("--point", type=Path, help="pretrained file for point")
    parser.add_argument("--title", type=str, default="Adeleine", help="title")
    parser.add_argument("--ref_base", type=int, default=384, help="first height for reference image")
    parser.add_argument("--scrib_base", type=int, default=512, help="first height for point")
    args = parser.parse_args()
    main(args)
