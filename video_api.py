from flask import Flask, request, jsonify
import json
import os
import video_demo_json


app = Flask(__name__)


@app.route('/ptg-rules-api/egoism-hoi', methods=['POST'])
def process():
    print("Post request received. Processing...")
    # Extract parameters from the request
    content = request.json
    print("Request content:", content)
    image_path = content.get('image_path', None)
    video_path = content.get('video_path', None)
    start_time_sec = content.get('start_time_sec', 0)
    end_time_sec = content.get('end_time_sec', None)
    desired_fps = content.get('desired_fps', 1)
    save_dir = content.get('save_dir', "../../output/")

    result = video_demo_json.main(image_path=image_path, video_path=video_path,
                                  start_time_sec=start_time_sec, end_time_sec=end_time_sec,
                                  desired_fps=desired_fps, save_dir=save_dir)

    return result

if __name__ == "__main__":
    app.run(debug=True)
