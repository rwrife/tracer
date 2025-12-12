# Tracer

## Requirements
- Python
- Web Cam

## Installation
```pip install -r ./requirements.txt```

## Usage

1. Print the frame_layout.svg file to paper, it should nearly fill an 8x11" sheet of paper, or run `python generate_frame_with_tags.py` to create a new layout file.
2. Place object inside the printed frame.
3. Run `python trace_to_svg`, if you have multiple cameras press `TAB` to toggle between cameras (this may take a few seconds to initialize the other camera), point the camera at the printed out layout file, press `SPACE` to capture the outline of the SVG.
