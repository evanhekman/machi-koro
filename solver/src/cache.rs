use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub fn delta_path(dir: &Path, depth: usize) -> PathBuf {
    dir.join(format!("d{:02}.bin.zst", depth))
}

/// Write `delta` (new entries from one depth pass) to a zstd-compressed flat binary.
/// Format: [u64 count][u64 key, f64 value, ...]
pub fn save_delta(dir: &Path, depth: usize, delta: &HashMap<u64, f64>) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let path = delta_path(dir, depth);
    let file = fs::File::create(&path)?;
    let mut enc = zstd::Encoder::new(file, 9)?;
    enc.write_all(&(delta.len() as u64).to_le_bytes())?;
    for (&k, &v) in delta {
        enc.write_all(&k.to_le_bytes())?;
        enc.write_all(&v.to_le_bytes())?;
    }
    enc.finish()?;
    Ok(())
}

/// Load only the single delta needed to resume at `target_depth`.
/// To compute depth d, only the depth-(d-1) delta is queried at runtime.
/// Returns the frozen map and the highest depth present on disk (to detect cache coverage).
pub fn load_deltas(dir: &Path, target_depth: usize) -> std::io::Result<(HashMap<u64, f64>, usize)> {
    // Find how far the cache extends on disk.
    let mut max_on_disk = 0usize;
    for depth in 1usize.. {
        if !delta_path(dir, depth).exists() { break; }
        max_on_disk = depth;
    }

    if max_on_disk == 0 || target_depth <= 1 {
        return Ok((HashMap::new(), 0));
    }

    // We only need the delta one level below what we're about to compute.
    // If cache already covers target_depth, load that delta to serve the first-turn table.
    let load_depth = target_depth.min(max_on_disk);
    let path = delta_path(dir, load_depth);
    let size_mb = path.metadata().map(|m| m.len() as f64 / 1e6).unwrap_or(0.0);
    let t0 = std::time::Instant::now();

    let file = fs::File::open(&path)?;
    let mut dec = zstd::Decoder::new(file)?;

    let mut len_buf = [0u8; 8];
    dec.read_exact(&mut len_buf)?;
    let count = u64::from_le_bytes(len_buf) as usize;

    let mut frozen = HashMap::with_capacity(count);
    let mut kbuf = [0u8; 8];
    let mut vbuf = [0u8; 8];
    for _ in 0..count {
        dec.read_exact(&mut kbuf)?;
        dec.read_exact(&mut vbuf)?;
        frozen.insert(u64::from_le_bytes(kbuf), f64::from_le_bytes(vbuf));
    }

    println!(
        "  loaded depth {:2}  entries={:>9}  ({:.1} MB  {:.2}s)",
        load_depth, frozen.len(), size_mb, t0.elapsed().as_secs_f64()
    );

    Ok((frozen, max_on_disk))
}
