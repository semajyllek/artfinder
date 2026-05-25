
def wikiart_image_first_generator(stream, labels, authority_set):
    """
    Transforms incoming integer labels from Hugging Face into string names
    with deep verbose telemetry tracking to debug ingestion filtering.
    """
    print(f"\n📢 Telemetry Activated: Scanning stream against an authority set of {len(authority_set)} artists...")
    
    total_scanned = 0
    total_matched = 0
    rejected_samples = set()

    for idx, item in enumerate(stream):
        total_scanned += 1
        artist_id = item['artist']
        artist_name = labels[artist_id] if 0 <= artist_id < len(labels) else "Unknown"
        
        normalized_name = artist_name.lower().strip()
        
        # Periodic Telemetry Pulse
        if total_scanned % 500 == 0:
            print(f"📊 Stream Progress | Scanned: {total_scanned:,} | Matched & Accepted: {total_matched:,}")
            if rejected_samples:
                print(f"   🚫 Sample Rejected Names in this block: {list(rejected_samples)[:5]}")
                rejected_samples.clear()

        if normalized_name in authority_set:
            total_matched += 1
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
        else:
            # Track a few rejections so we can see what the names look like
            if len(rejected_samples) < 10:
                rejected_samples.add(artist_name)
