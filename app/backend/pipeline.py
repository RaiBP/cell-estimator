config = {
    'image_segmentator': {
        'method': "threshold",
        # whether to segment the phase or the amplitude image
        'image_to_segment': 'amplitude'
        # Put other parameters here in the future
    },
    'classifier': {
        'method': 'tsc'
    }
}
