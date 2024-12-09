from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("NimVideo/cogvideox-2b-img2vid")

prompt = "Astronaut in a jungle"
image = pipe(prompt).images[0]