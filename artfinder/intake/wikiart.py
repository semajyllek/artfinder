
def wikiart_image_first_generator(stream, labels, authority_set):
    """
    Transforms incoming integer labels from Hugging Face into string names
    on-the-fly and drops any artwork not matching the active authority set.
    Fits the exact lowercase field key schema expected by VaultBuilder.
    """
    for idx, item in enumerate(stream):
        artist_id = item['artist']
        # Safely capture string index references
        artist_name = labels[artist_id] if 0 <= artist_id < len(labels) else "Unknown"
        
        # Cross-reference text signatures against our active curation array
        if artist_name.lower().strip() in authority_set:
            tracking_filename = f"wikiart_{idx}.jpg"
            
            yield {
                'visual_id': f"wikiart_{idx}",
                'title': item.get('title', 'Unknown Title'),
                'artist': artist_name,
                'filename': tracking_filename,
                'ImageURL': f"hf://wikiart/{idx}",
                'SourceURL': "https://www.wikiart.org",
                'Source': 'wikiart',
                'image': item['image']
            }
