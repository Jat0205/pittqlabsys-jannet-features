#!/usr/bin/env python3
# Written by <Jannet Trabelsi>
"""
Preview an ODMR pulsed sequence.

Builds and shows the example sequence WITHOUT creating an ODMRPulsedExperiment,
so no devices are connected. All the sequence work is done by
ODMRSequencePreview (see odmr_sequence_preview.py).

Run this from your project root (the folder that contains `src/`).
"""

from src.Model.experiments.odmr_sequence_preview import ODMRSequencePreview, EXAMPLE_ODMR_SEQUENCE, EXAMPLE_CHIRP_DEER_SEQUENCE


def main():
    # Use the built-in example sequence (same one the experiment uses).
    # Pass your own text with sequence_text=... to preview something else.
    preview = ODMRSequencePreview(sequence_text=EXAMPLE_ODMR_SEQUENCE)

    print("Example ODMR Sequence:")
    print(preview.sequence_text)
    print("\n" + "=" * 50 + "\n")

    if preview.load():
        print("Sequence loaded successfully")
    else:
        print("Failed to load sequence")

    if preview.build():
        print("Scan sequences built successfully")
    else:
        print("Failed to build scan sequences")

    # Open the preview window (blocks until you close it).
    preview.preview(10)


if __name__ == "__main__":
    main()
