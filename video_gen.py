import cv2
import time
from PIL import Image
from google import genai
from google.genai import types
from google.oauth2 import service_account

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
    Shoot a smooth drone fly-by movie starring this tree. The drone should capture a full 360 degree view of the tree (The first frame should be the uploaded image facing the front of the tree. Then circle the drone around to show a frame facing the back of the tree. Keep circling in the same direction to face the front again for the final frame).
    Make the drone fly-by as fast as needed to finish the full 360 degree circle, and make the drone fly parallel to the ground (no vertical camera movement). Make sure every angle of the tree is captured (infer any details needed about surroundings). Make sure the entire tree and movie is reanimated """
    if prompt_add is None:
        prompt += "with boosted, saturated, vibrant color grading."
    else:
        prompt += prompt_add

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
    styles = [
        "like a high-contrast dramatic shadowy noir film",
        "like a Wizard-Oz vibrant Technicolor stage play",
        "like a film-grainy sepia-color old western film",
        "like a glitchy-VHS chromatic-abr vaporwave film",
        "like a neon-lit rainy city techy cyperbunk film",
        "like a telephoto-lens hi-bokeh NatGeo IMAX film",
        "like a hand-drawn cel-shade Studio Ghibli anime",
        "like a plastic-feel stop-motion claymation film",
        "like a gothic victorian puppety Tim Burton film",
        "like a gold-lit childish whimsical fantasy film",
        "like a swirling Van Gogh starry nights painting",
        "like a surrealist melting Salvador Dali painting",
    ]
    # for style in styles:
    #     video_gen(image_path="", video_path="", prompt_add=style, args_input=0)

    angles = [
        None,
        "with the camera lying as low as possible from the POV of an ant on the ground looking up to the tree",
        "with the camera raised high from the POV of a bird circling the tree canopy looking down at the tree",
        "with the camera circling as far away from the tree as possible, giving a wide angle shot of the tree",
        "with the camera flying into/close to the tree as possible, giving a close-up of the trunk and canopy",
    ]
    for angle in angles:
        video_gen(image_path="", video_path="", prompt_add=angle, args_input=0)