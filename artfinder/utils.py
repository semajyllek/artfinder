import faiss
from .config import Config


def print_engine_diagnostics(state):
    import faiss
    from artfinder.config import Config
    
    print("🔍 --- ARTFINDER CLOUD SYSTEM DIAGNOSTICS --- 🔍\n")
    
    # 1. Metadata Check
    try:
        from artfinder.vault.builder import load_source_metadata
        df = load_source_metadata(state.bucket)
        valid_records = len(df.dropna(subset=['id']))
        print(f"📄 Metadata Tracking Parquet:")
        print(f"   • Total Logged Artworks: {valid_records:,}")
    except Exception as e:
        print(f"📄 Metadata Tracking Parquet: ⚠️ Error loading - {e}")

    # 2. Raw Vault Check
    try:
        blob = state.bucket.blob(Config.VAULT_PATH)
        if blob.exists():
            blob.download_to_filename("temp_vault.bin")
            vault = faiss.read_index_binary("temp_vault.bin")
            print(f"\n📦 Raw Flat Vault (The Source Vectors):")
            print(f"   • Total Feature Vectors: {vault.ntotal:,}")
            print(f"   • Vector Dimensionality: {vault.d} bits")
            print(f"   • Vector Bytes:          {vault.d // 8} bytes")
        else:
            print(f"\n📦 Raw Flat Vault: ⚠️ Missing from GCS ({Config.VAULT_PATH})")
    except Exception as e:
        print(f"\n📦 Raw Flat Vault: ⚠️ Error - {e}")

    # 3. Production Index Check
    try:
        blob = state.bucket.blob(Config.INDEX_PATH)
        if blob.exists():
            blob.download_to_filename("temp_index.bin")
            index = faiss.read_index_binary("temp_index.bin")
            print(f"\n⚡ Production IVF Index (The Search Engine):")
            print(f"   • Total Clustered Features: {index.ntotal:,}")
            
            # Drill into wrapper to find cluster counts
            core_index = index.index if hasattr(index, 'index') else index
            clusters = core_index.nlist if hasattr(core_index, 'nlist') else "Unknown"
            print(f"   • Total Voronoi Clusters:   {clusters}")
            print(f"   • Vector Dimensionality:    {index.d} bits")
        else:
            print(f"\n⚡ Production IVF Index: ⚠️ Missing from GCS ({Config.INDEX_PATH})")
    except Exception as e:
        print(f"\n⚡ Production IVF Index: ⚠️ Error - {e}")
    print("\n" + "─"*50)
