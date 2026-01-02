import cv2
import time
from PIL import Image
from google import genai
from google.genai import types
from google.oauth2 import service_account
from moviepy import VideoFileClip, concatenate_videoclips

# oauth, genai stuff
filename = "service_account_file.json"
credentials = service_account.Credentials.from_service_account_file(
    filename, scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
client = genai.Client(
    vertexai    = True,
    credentials = credentials,
    project     = "cael-workspace-berkeley",
    location    = "us-central1"
)

# prompt engineering
def video_gen(image_path, video_path=None, prompt_add=None, args_input=None):
    prompt = f"""
Shoot a drone fly-by movie starring this tree. The camera should capture a full 360 degree view around the tree (The first frame is the given image facing the front of the tree. Then circle the drone to the RIGHT to show a frame facing the back of the tree. Make the drone fly-by as fast as needed to finish a full 360 degree view and make the drone fly level with the ground. Make sure every angle of the tree is shown so infer surroundings as needed. MOST IMPORTANTLY: make the drone fly far/wide enough to fully capture all parts of the tree, trunk, and canopy: the video edge must not cut off any organ of the tree so infer non-visible parts as needed. MORE IMPORTANTLY: fly the drone to the RIGHT, ignore any obstacles fly through them, just pan to the RIGHT.
"""
    # video gen options:
    if args_input is None:
        arg = input(f"""Video generation options:
( ) Uses {image_path} in default image - video gen
(1) Uses {image_path} as both first and last frame
(2) Uses last frame of {video_path} as first frame
(3) Does (2), then uses {image_path} as last frame""")
    else: arg = str(args_input)

    # crop image to 16x9
    img = Image.open(image_path)
    w, h = img.size
    ratio = 16/9 if w > h else 9/16
    if w / h > ratio:
        new_w, new_h = int(h * ratio), h
    else:
        new_w, new_h = w, int(w / ratio)

    # save cropped image
    l = (w - new_w) // 2
    t = (h - new_h) // 2
    crop_path = f"crop_{image_path}"
    img.crop((l, t, l+new_w, t+new_h)).save(crop_path)

    # init video frames
    first_frame = types.Image.from_file(location=crop_path)
    last_frame = None
    if arg == "1" or arg == "3":
        last_frame = first_frame

    # (2,3) Uses last frame of {VIDEO_PATH} as first frame
    if arg == "2" or arg == "3":
        cap = cv2.VideoCapture(video_path)
        frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_cnt - 1)
        _, buffer = cv2.imencode(".png", cap.read()[1])
        first_frame = types.Image(image_bytes=buffer.tobytes(), mime_type="image/png")

    # video gen options:
    config = types.GenerateVideosConfig(
        # params we can edit
        aspect_ratio        = "16:9" if w > h else "9:16",
        last_frame          = last_frame,
        duration_seconds    = 8,
        generate_audio      = False,
        negative_prompt     = "drones",
        reference_images    = None,
        resolution          = "1080p",
        seed                = None,
        # ones we can't edit
        compression_quality = "optimized",
        enhance_prompt      = True,
        fps                 = 24,
        http_options        = None,
        mask                = None,
        number_of_videos    = 1,
        output_gcs_uri      = None,
        person_generation   = None,
        pubsub_topic        = None,
    )

    # Generate video with Veo 3.1 using the image.
    operation = client.models.generate_videos(
        model = "veo-3.1-generate-preview",
        prompt= prompt,
        image = first_frame,
        config= config,
    )

    # Poll the operation status until the video is ready.
    cnt = 0
    while not operation.done:
        print(f"Waiting (ETA {120-cnt}s)")
        time.sleep(10)
        cnt += 10
        operation = client.operations.get(operation)
    
    if operation.error:
        print(f"Video generation errored: {operation.error}")
    elif operation.response:
        # Download the video.
        video = operation.response.generated_videos[0].video
        video.save(f"{time.ctime()}.mp4")
        print(f"Generated video saved to {time.ctime()}.mp4")
        return True
    else:
        print(f"Operation finished but returned no response")

# prompt engineering
if __name__ == "__main__":
    for i in [1, 8]:
        video_gen(image_path=f"{i}.png", video_path=f"{i}.mp4", args_input=3)

    def concatenate(path1, path2, pathout):
        clip1 = VideoFileClip(path1)
        clip2 = VideoFileClip(path2).subclip(0, 7.5)
        final = concatenate_videoclips([clip1, clip2])
        final.write_videofile(pathout)

    for i in [1, 2, 3]:
        concatenate(f"{i}.mp4", f"{i}{i}.mp4", f"out{i}.mp4")
