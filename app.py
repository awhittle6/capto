from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import TextToVideoZeroPipeline
import torch
import uuid
import os
import subprocess

app = FastAPI()

# Load the pipeline (model and scheduler)
# Adjust model name to the chosen one
pipeline = TextToVideoZeroPipeline.from_pretrained(
    "genmo/mochi-1-preview", 
    torch_dtype=torch.float16,
    revision="fp16",
) 

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_video(request: PromptRequest):
    prompt = request.prompt
    try:
        # Generate frames from text prompt
        # The pipeline may return a series of PIL images or raw tensors.
        # Check the model's documentation for exact usage.
        frames = pipeline(prompt)
        
        # frames should be a list of PIL images; save them as temporary files.
        output_dir = f"outputs_{uuid.uuid4()}"
        os.makedirs(output_dir, exist_ok=True)
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = os.path.join(output_dir, f"frame_{i:05d}.png")
            frame.save(frame_path)
            frame_paths.append(frame_path)

        # Use ffmpeg to stitch images into a video
        video_path = os.path.join(output_dir, "output.mp4")
        # Adjust fps if needed, for example 8 fps:
        cmd = [
            "ffmpeg", "-y", "-framerate", "8", "-i", 
            os.path.join(output_dir, "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", video_path
        ]
        subprocess.run(cmd, check=True)

        # Return the path or a downloadable URL (in production, serve this file via a CDN or storage)
        return {"video_url": f"/videos/{os.path.basename(video_path)}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
