#!/usr/bin/env python3

import os
import numpy as np
import argparse

# Define the base directory for raw NPZ files relative to this script's location
# Assumes this script might be in a 'utils' or 'tools' folder,
# and 'downloaded_data' is in the parent project directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Adjust this path if your script is located elsewhere relative to the data
DEFAULT_RAW_NPZ_DIR = os.path.join(SCRIPT_DIR, "..", "downloaded_data", "quickdraw_sketchrnn")

def convert_3element_to_5element(sketch_3elem):
    """
    Converts a SketchRNN 3-element sequence [dx, dy, p_eos_only]
    to a 5-element sequence [dx, dy, p0_draw, p1_lift, p2_eos].
    This is a simplified interpretation for consistent processing.
    p0 (pen_down): 1 if not EOS, 0 if EOS (pen is lifted at the very end).
    p1 (pen_up after this point): 1 if next point is EOS (marks end of stroke), else 0.
                                  For single point sketch, p1 is 0.
    p2 (end_of_sketch): from the 3rd element of input.
    """
    num_points = sketch_3elem.shape[0]
    if num_points == 0:
        return np.array([], dtype=np.float32).reshape(0, 5)

    sketch_5elem = np.zeros((num_points, 5), dtype=np.float32)
    sketch_5elem[:, :2] = sketch_3elem[:, :2] # Copy dx, dy
    sketch_5elem[:, 4] = sketch_3elem[:, 2]   # Copy p2 (end_of_sketch)

    for i in range(num_points):
        is_eos_current = sketch_5elem[i, 4] == 1.0
        
        # p0 (pen_down state for *this* point being drawn)
        # If this point is the end of the sketch, pen is considered up for this point's state.
        # Otherwise, pen is down.
        sketch_5elem[i, 2] = 0.0 if is_eos_current else 1.0
        
        # p1 (pen_up state *after* this point, i.e., end of current stroke)
        # If this is the last point, or the next point is the start of a new stroke (which we can't tell from 3-elem),
        # or if this point itself is EOS, then the stroke ends here.
        # For simplicity, if this point is EOS, or if it's the second to last and the last is EOS, stroke ends.
        if is_eos_current:
            sketch_5elem[i, 3] = 1.0 # Pen lifts after EOS point
            if i > 0: # If not the first point, the previous point was the end of a stroke
                sketch_5elem[i-1, 3] = 1.0
                sketch_5elem[i-1, 2] = 0.0 # Pen was down for drawing, now lifted after
        elif i + 1 < num_points and sketch_5elem[i+1, 4] == 1.0: # If next point is EOS
            sketch_5elem[i, 3] = 1.0 # This point is end of stroke
            sketch_5elem[i, 2] = 0.0 # Pen lifts after this stroke
        else:
            sketch_5elem[i, 3] = 0.0 # Stroke continues

    # Final check: if the sketch is just one point and it's EOS
    if num_points == 1 and sketch_5elem[0, 4] == 1.0:
        sketch_5elem[0, 2] = 0.0 # Not drawing
        sketch_5elem[0, 3] = 1.0 # Pen up after this single EOS point

    return sketch_5elem.astype(np.float32)


def inspect_npz(category_name, raw_dir):
    """
    Loads an NPZ file for a given category and prints details of the first sketch.
    """
    npz_file_path = os.path.join(raw_dir, f"{category_name}.npz")

    if not os.path.exists(npz_file_path):
        print(f"Error: NPZ file not found at {npz_file_path}")
        print(f"Please ensure the file for category '{category_name}' is downloaded to the correct raw directory.")
        return

    print(f"--- Inspecting NPZ file: {npz_file_path} ---")
    
    try:
        data = np.load(npz_file_path, encoding='latin1', allow_pickle=True)
        print(f"Keys in NPZ file: {list(data.files)}")

        first_sketch_raw = None
        source_key = None

        for key in ['train', 'valid', 'test']: # Check in this order
            if key in data and len(data[key]) > 0:
                first_sketch_raw = data[key][0]
                source_key = key
                print(f"\nFound first sketch in array: '{source_key}'")
                break
        
        if first_sketch_raw is None:
            print("No sketches found in 'train', 'valid', or 'test' arrays in the NPZ file.")
            return

        print(f"Raw shape of first sketch: {first_sketch_raw.shape}")
        
        sketch_to_print = None
        if first_sketch_raw.shape[1] == 3:
            print("Detected 3-element sketch format. Converting to 5-element for printing.")
            sketch_to_print = convert_3element_to_5element(first_sketch_raw)
        elif first_sketch_raw.shape[1] == 5:
            print("Detected 5-element sketch format.")
            sketch_to_print = first_sketch_raw.astype(np.float32)
        else:
            print(f"Unsupported sketch format with shape {first_sketch_raw.shape}. Cannot print details.")
            return

        if sketch_to_print is not None and sketch_to_print.size > 0:
            print(f"\n--- Entire First Sketch from '{category_name}.npz' (from '{source_key}' array) ---")
            print(f"Shape after potential conversion: {sketch_to_print.shape}")
            print("[dx,   dy,  p0,  p1,  p2]")
            print("-------------------------------")
            for point in sketch_to_print:
                # Formatting for alignment and to show integers for pen states
                print(f"[{point[0]:>5.0f}, {point[1]:>5.0f}, {int(point[2])}, {int(point[3])}, {int(point[4])}]")
        else:
            print("First sketch is empty or could not be processed.")

    except Exception as e:
        print(f"Error loading or processing NPZ file {npz_file_path}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the content of a QuickDraw SketchRNN .npz file.")
    parser.add_argument("category_name", type=str, help="The name of the category to inspect (e.g., 'cat', 'dog').")
    parser.add_argument("--raw_dir", type=str, default=DEFAULT_RAW_NPZ_DIR,
                        help=f"Directory containing the raw .npz files (default: {DEFAULT_RAW_NPZ_DIR})")
    args = parser.parse_args()

    inspect_npz(args.category_name, args.raw_dir)
