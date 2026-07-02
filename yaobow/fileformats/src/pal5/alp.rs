//! PAL5 `.alp` terrain alphamap decoder.
//!
//! Each terrain block ships `alphamap_<r>_<c>.alp` alongside the matching
//! `<map>_<r>_<c>.mp` heightfield. The file holds, per terrain patch, a
//! 64×64 per-texel **blend-weight raster** used to splat up to four
//! terrain-texture layers over the patch. The concrete texture ids for the
//! four layer *slots* live in the `.mp` block footer
//! ([`fileformats::pal5::mp::MpFile::texture_ids`]); the weights here say
//! how strongly each slot shows through at each texel.
//!
//! ## File layout
//! ```text
//! 0x0000..0x0800   patch table: 256 entries × 8 bytes
//! 0x0800..EOF      packed per-patch LZO1X-compressed blobs, patch order
//! ```
//! A block is a 16×16 grid of patches (256). The patch at grid index `i`
//! covers block-local cell `(col = i % 16, row = i / 16)` — the same
//! row-major order the `.mp` records use (`index = row*16 + col`).
//!
//! ### Patch table
//! Each entry is intended to be `(u32 blob_offset, u32 blob_len)`, both
//! little-endian absolute. Every shipped file carries the same corruption
//! in entries 2–3 (a spurious high-word length and a nonsense offset).
//! Blobs are physically contiguous in patch order, so we recover ranges
//! robustly: trust an entry's offset only when sane (`>= 0x800`, `< len`);
//! otherwise reconstruct from the previous recovered offset plus the
//! previous length's low 16 bits; take each blob's length as
//! `next_offset - offset`.
//!
//! ### Patch blob — LZO1X stream
//! The whole blob is a single LZO1X-compressed stream (the earlier
//! "uncompressed `width/height/plane_count` header" guess was wrong — those
//! bytes are just the start of the compressed data). It decompresses to
//! `0x5030` bytes:
//! ```text
//! u32 width    = 64
//! u32 height   = 64
//! u32 encoded  // 0..3, number of *extra* blend layers (active layers = +1)
//! u32 pixels[64*64]   // raster-scan, each pixel packs 4 weight bytes
//! …trailing scratch (buffer over-allocation, ignored)
//! ```
//! Each pixel's four little-endian bytes `[b0, b1, b2, b3]` are per-channel
//! weights that **sum to 255**, and map **one-to-one to the four texture
//! slots in order**: byte `b_j` is the blend weight for the `.mp` footer
//! texture slot `j`.
//! ```text
//! slot0 ← b0   slot1 ← b1   slot2 ← b2   slot3 ← b3
//! ```
//! The channels are used in a priority order though: a patch with `encoded +
//! 1` active layers carries its weight in exactly bytes `{b2, b1, b0, b3}[0..=
//! encoded]`. So `b2` is the always-present primary (open ground → slot 2,
//! typically grass), `b1` the first overlay (→ slot 1, e.g. the road's
//! ground), then `b0` (→ slot 0) and `b3` (→ slot 3). A single-layer
//! (`encoded = 0`) patch is **100% `b2` → slot 2**.
//!
//! Single-/base-texture blocks therefore park their full weight on slot 2 even
//! when their only valid footer id is at slot 0; the terrain builder folds
//! weight on unused slots (footer id `-1`) back into the base layer (slot 0),
//! so this resolves correctly.
//!
//! [`AlpPatch::planes`] always holds **four** rasters, `planes[j]` = byte
//! `b_j` = the weight for texture slot `j` (paired with `.mp` footer
//! `texture_ids[j]`).
//!
//! This identity mapping was pinned against the rendered terrain on
//! `kuangfengzhai`: `b2` (open ground, ~78%) is the grass slot (footer slot 2 =
//! `cao002`) and `b1` (the road, ~15%) is the warm-tan ground at footer slot 1
//! (`shan035`) — sampled road colour `(179,149,110)` matches the original's
//! `(157,140,91)`. Note footer slot 3 (`zhuan042`) is genuinely a *blue*-grey
//! texture in the shipped assets (verified by three independent DXT3 decoders);
//! it is only a minor ~0.4% terrain detail here, so routing the road to it
//! (an earlier `[0,3,2,1]` mapping) wrongly painted the path purple. An even
//! earlier `slot0 ← b2` mapping painted the whole village with the slot-0 clay
//! (`jiaotu001`).
//!
//! All offsets/sizes and the codec were derived clean-room from the shipped
//! binaries (no external PAL5 implementation was consulted).

use serde::Serialize;

const TABLE_ENTRIES: usize = 256;
const TABLE_BYTES: usize = TABLE_ENTRIES * 8;

/// Weight-raster edge length (texels per patch edge).
pub const WEIGHT_EDGE: usize = 64;
/// Texels per patch (`64 × 64`).
pub const WEIGHT_TEXELS: usize = WEIGHT_EDGE * WEIGHT_EDGE;
/// Decompressed patch buffer size the game allocates: `(64*64 + 0x40c) * 4`.
const DECOMP_LEN: usize = (WEIGHT_TEXELS + 0x40c) * 4;

/// One decoded patch's blend weights.
#[derive(Debug, Clone, Serialize)]
pub struct AlpPatch {
    /// Number of active blend layers (`encoded + 1`, range 1..=4). Informational
    /// only — the active channels are the byte-priority prefix `[b2, b1, b0,
    /// b3]` (→ slots `[2, 1, 0, 3]`; see module docs); use [`AlpPatch::planes`]
    /// (always four channels, one per texture slot) for the actual weights.
    pub layer_count: u8,
    /// Per-texture-slot 64×64 weight raster: `planes[j]` is the weight for
    /// texture slot `j` (`.mp` footer `texture_ids[j]`), row-major
    /// (`row * 64 + col`), values `0..=255`. Always four entries; for any texel
    /// the four sum to 255. Empty only if the blob failed to decode.
    pub planes: Vec<Vec<u8>>,
}

impl AlpPatch {
    /// Whether this patch actually blends more than one layer.
    pub fn is_multilayer(&self) -> bool {
        self.layer_count > 1
    }
}

/// A decoded `.alp` block: 256 patches in row-major grid order.
#[derive(Debug, Clone, Serialize)]
pub struct AlpFile {
    pub patches: Vec<AlpPatch>,
}

#[derive(thiserror::Error, Debug)]
pub enum AlpError {
    #[error("file too small ({0} bytes)")]
    TooSmall(usize),
}

impl AlpFile {
    /// Decode a raw `alphamap_<r>_<c>.alp` file.
    pub fn read(raw: &[u8]) -> Result<AlpFile, AlpError> {
        if raw.len() < TABLE_BYTES {
            return Err(AlpError::TooSmall(raw.len()));
        }
        let lzo = minilzo_rs::LZO::init().ok();
        let ranges = recover_blob_ranges(raw);
        let patches = ranges
            .into_iter()
            .map(|(off, len)| decode_patch(raw, off, len, lzo.as_ref()))
            .collect();
        Ok(AlpFile { patches })
    }

    /// The decoded patch at block-local cell `(col, row)` (each `0..16`).
    pub fn patch(&self, col: usize, row: usize) -> Option<&AlpPatch> {
        if col >= 16 || row >= 16 {
            return None;
        }
        self.patches.get(row * 16 + col)
    }
}

fn read_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// Recover each patch blob's `(offset, len)` from the (partly corrupt)
/// table, exploiting that blobs are physically contiguous in patch order.
fn recover_blob_ranges(raw: &[u8]) -> Vec<(usize, usize)> {
    let len = raw.len();
    let mut offsets = vec![usize::MAX; TABLE_ENTRIES];
    for p in 0..TABLE_ENTRIES {
        let off = read_u32(raw, p * 8) as usize;
        if (TABLE_BYTES..len).contains(&off) {
            offsets[p] = off;
        }
    }
    // Forward-fill any rejected offset from the previous recovered offset
    // plus that entry's length (the low 16 bits are the real length even
    // in the corrupt entries).
    for p in 0..TABLE_ENTRIES {
        if offsets[p] == usize::MAX {
            let prev = if p == 0 { TABLE_BYTES } else { offsets[p - 1] };
            let prev_len = (read_u32(raw, p.saturating_sub(1) * 8 + 4) & 0xffff) as usize;
            offsets[p] = (prev + prev_len).min(len);
        }
    }
    (0..TABLE_ENTRIES)
        .map(|p| {
            let start = offsets[p];
            let end = if p + 1 < TABLE_ENTRIES {
                offsets[p + 1].min(len)
            } else {
                len
            };
            (start, end.saturating_sub(start))
        })
        .collect()
}

/// Decode one LZO1X-compressed patch blob into its four per-texture-slot
/// weight rasters (`planes[j]` = packed byte `b_j` = weight for slot `j`).
fn decode_patch(raw: &[u8], off: usize, len: usize, lzo: Option<&minilzo_rs::LZO>) -> AlpPatch {
    let empty = AlpPatch {
        layer_count: 0,
        planes: Vec::new(),
    };
    let (Some(lzo), true) = (lzo, len > 0 && off + len <= raw.len()) else {
        return empty;
    };
    let blob = &raw[off..off + len];
    let Ok(decomp) = lzo.decompress(blob, DECOMP_LEN) else {
        return empty;
    };
    // Header: width, height, encoded, then 4096 packed-weight pixels.
    if decomp.len() < 12 + WEIGHT_TEXELS * 4
        || read_u32(&decomp, 0) != WEIGHT_EDGE as u32
        || read_u32(&decomp, 4) != WEIGHT_EDGE as u32
    {
        return empty;
    }
    let encoded = read_u32(&decomp, 8).min(3) as usize;
    let layer_count = (encoded + 1) as u8; // informational; see module docs

    // One raster per texture slot: `planes[j]` is packed byte `b_j` (the
    // weight for footer texture slot `j`). All four are emitted; inactive
    // slots are naturally ~0 in the packed pixels.
    let planes = (0..4)
        .map(|slot| {
            (0..WEIGHT_TEXELS)
                .map(|i| decomp[12 + i * 4 + slot])
                .collect::<Vec<u8>>()
        })
        .collect::<Vec<_>>();

    AlpPatch {
        layer_count,
        planes,
    }
}

/// Map a terrain-texture id to its `TerrainTexture\*.dds` filename (without
/// directory). Ids are the `TextureID` field of `Config/Data/T_textureConfig.ini`
/// (the value stored in `.mp` patch tables). Returns `None` for out-of-range ids.
pub fn terrain_texture_name(id: u8) -> Option<&'static str> {
    TERRAIN_TEXTURES.get(id as usize).copied()
}

/// The 220 PAL5 terrain textures registered in `Config/Data/T_textureConfig.ini`,
/// indexed by the `TextureID` field (which is what `.mp` patch tables store).
/// The section name in that ini is the `TerrainTexture\\<name>.dds` base name.
/// Transcribed clean-room from the shipped `T_textureConfig.ini`.
pub static TERRAIN_TEXTURES: [&str; 220] = [
    "Cl001.dds",
    "Cl002.dds",
    "Cl003.dds",
    "Cl004.dds",
    "Cl005.dds",
    "Cl006.dds",
    "Cl007.dds",
    "Cl008.dds",
    "Cl009.dds",
    "Cl010.dds",
    "Cl011.dds",
    "shan001.dds",
    "shan002.dds",
    "shan003.dds",
    "shan004.dds",
    "shan005.dds",
    "shan006.dds",
    "shan007.dds",
    "shan008.dds",
    "shan009.dds",
    "zhuan001.dds",
    "zhuan002.dds",
    "zhuan003.dds",
    "zhuan004.dds",
    "zhuan005.dds",
    "zhuan006.dds",
    "zhuan007.dds",
    "zhuan008.dds",
    "shan010.dds",
    "shan011.dds",
    "shan012.dds",
    "waicheng_zhuan001.dds",
    "waicheng_zhuan002.dds",
    "waicheng_zhuan003.dds",
    "waicheng_zhuan004.dds",
    "waicheng_water001.dds",
    "waicheng_water002.dds",
    "waicheng_water003.dds",
    "zhuan009.dds",
    "zhuan010.dds",
    "zhuan011.dds",
    "zhuan012.dds",
    "shan013.dds",
    "shan014.dds",
    "shan015.dds",
    "shan016.dds",
    "shan017.dds",
    "shan018.dds",
    "shan019.dds",
    "shan020.dds",
    "shan021.dds",
    "dibiao424.dds",
    "dibiao425.dds",
    "dibiao426.dds",
    "dibiao427.dds",
    "dibiao428.dds",
    "dibiao429.dds",
    "shan022.dds",
    "shan023.dds",
    "shan024.dds",
    "shan025.dds",
    "shan026.dds",
    "shan027.dds",
    "shan028.dds",
    "shan029.dds",
    "shan030.dds",
    "zhuan013.dds",
    "shan031.dds",
    "cao001.dds",
    "sha001.dds",
    "sadi001.dds",
    "tudi001.dds",
    "cao002.dds",
    "shan032.dds",
    "cao003.dds",
    "sadi002.dds",
    "tudi002.dds",
    "cao004.dds",
    "cao005.dds",
    "cao006.dds",
    "cao007.dds",
    "cao008.dds",
    "cao009.dds",
    "cao010.dds",
    "cao011.dds",
    "cao012.dds",
    "cao013.dds",
    "cao014.dds",
    "cao015.dds",
    "cao016.dds",
    "cao017.dds",
    "cao018.dds",
    "Luoye001.dds",
    "Luoye002.dds",
    "Luoye022.dds",
    "shan033.dds",
    "zhuan014.dds",
    "zhuan015.dds",
    "zhuan016.dds",
    "zhuan017.dds",
    "zhuan018.dds",
    "zhuan019.dds",
    "zhuan020.dds",
    "zhuan021.dds",
    "zhuan022.dds",
    "tudi003.dds",
    "tudi004.dds",
    "tudi005.dds",
    "tudi006.dds",
    "tudi007.dds",
    "tudi008.dds",
    "tudi009.dds",
    "tudi010.dds",
    "tudi011.dds",
    "tudi012.dds",
    "tudi013.dds",
    "tudi014.dds",
    "tudi015.dds",
    "tudi016.dds",
    "shan034.dds",
    "shan035.dds",
    "shan036.dds",
    "shan037.dds",
    "shan038.dds",
    "shan039.dds",
    "shan040.dds",
    "jiaotu001.dds",
    "suishi001.dds",
    "suishi002.dds",
    "suishi003.dds",
    "suishi004.dds",
    "sadi003.dds",
    "zhuan023.dds",
    "zhuan024.dds",
    "zhuan025.dds",
    "tudi017.dds",
    "tudi018.dds",
    "tudi019.dds",
    "zhuan026.dds",
    "shan041.dds",
    "cao019.dds",
    "cao020.dds",
    "shiban01.dds",
    "tudi020.dds",
    "tudi021.dds",
    "luoye023.dds",
    "shan042.dds",
    "dibiao430.dds",
    "waicheng_zhuan005.dds",
    "shan043.dds",
    "zhuan027.dds",
    "shan044.dds",
    "zhuan028.dds",
    "zhuan034.dds",
    "zhuan035.dds",
    "tudi022.dds",
    "tudi023.dds",
    "luoye003.dds",
    "luoye004.dds",
    "luoye005.dds",
    "zhuan036.dds",
    "zhuan037.dds",
    "zhuan038.dds",
    "zhuan029.dds",
    "zhuan030.dds",
    "zhuan031.dds",
    "zhuan032.dds",
    "zhuan033.dds",
    "zhuan039.dds",
    "xuedi001.dds",
    "xuedi002.dds",
    "luoye024.dds",
    "luoye025.dds",
    "luoye026.dds",
    "luoye027.dds",
    "shan045.dds",
    "shan046.dds",
    "cao022.dds",
    "ztqshan030.dds",
    "ztqtudi016.dds",
    "ztqtudi022.dds",
    "huacao001.dds",
    "huacao002.dds",
    "huacao003.dds",
    "huacao004.dds",
    "suishi005.dds",
    "suishi006.dds",
    "shan048.dds",
    "plhy001.dds",
    "plhy002.dds",
    "plhy003.dds",
    "plhy004.dds",
    "zhuan040.dds",
    "zhuan041.dds",
    "xsl001.dds",
    "xsl002.dds",
    "tudi024.dds",
    "suishi007.dds",
    "LU-1.dds",
    "LU-2.dds",
    "LU-3.dds",
    "LU-4.dds",
    "shan047.dds",
    "cao023.dds",
    "LU-5.dds",
    "LU-6.dds",
    "zhuan042.dds",
    "zhuan043.dds",
    "zhuan044.dds",
    "zhuan045.dds",
    "zhuan046.dds",
    "tangfucao001.dds",
    "tangfudi002.dds",
    "tudi025.dds",
    "CL012.dds",
    "zhuan047.dds",
    "zhuan048.dds",
    "shiban03.dds",
    "shan049.dds",
    "shan050.dds",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_table_has_expected_anchors() {
        // Indices are the `T_textureConfig.ini` `TextureID` field.
        assert_eq!(TERRAIN_TEXTURES.len(), 220);
        assert_eq!(terrain_texture_name(0), Some("Cl001.dds"));
        assert_eq!(terrain_texture_name(1), Some("Cl002.dds"));
        // kuangfengzhai cobblestone road dominant.
        assert_eq!(terrain_texture_name(133), Some("zhuan024.dds"));
        assert_eq!(terrain_texture_name(215), Some("zhuan047.dds"));
        assert_eq!(terrain_texture_name(219), Some("shan050.dds"));
        assert_eq!(terrain_texture_name(220), None);
    }

    /// Build a synthetic `.alp`: a 256-entry table whose first blob is an
    /// LZO1X stream of a single-layer (`encoded = 0`) 64×64 weight raster
    /// where byte `b2` (the primary/base channel) is 255. Decode and confirm
    /// the table recovery, LZO decode, and per-slot extraction (`planes[2]`).
    #[test]
    fn decodes_single_layer_patch() {
        let mut lzo = minilzo_rs::LZO::init().unwrap();

        let mut decomp = vec![0u8; DECOMP_LEN];
        decomp[0..4].copy_from_slice(&(WEIGHT_EDGE as u32).to_le_bytes());
        decomp[4..8].copy_from_slice(&(WEIGHT_EDGE as u32).to_le_bytes());
        decomp[8..12].copy_from_slice(&0u32.to_le_bytes()); // encoded = 0
        for i in 0..WEIGHT_TEXELS {
            decomp[12 + i * 4 + 2] = 255; // b2 -> texture slot 2
        }
        let blob = lzo.compress(&decomp).unwrap();

        let mut raw = vec![0u8; TABLE_BYTES];
        let off = TABLE_BYTES as u32;
        raw[0..4].copy_from_slice(&off.to_le_bytes());
        raw[4..8].copy_from_slice(&(blob.len() as u32).to_le_bytes());
        let tail = TABLE_BYTES + blob.len();
        for p in 1..TABLE_ENTRIES {
            raw[p * 8..p * 8 + 4].copy_from_slice(&(tail as u32).to_le_bytes());
        }
        raw.extend_from_slice(&blob);

        let alp = AlpFile::read(&raw).expect("decode");
        assert_eq!(alp.patches.len(), 256);
        let p0 = &alp.patches[0];
        assert_eq!(p0.layer_count, 1);
        // Always four per-slot planes; the weight sits on slot 2 (byte b2).
        assert_eq!(p0.planes.len(), 4);
        assert_eq!(p0.planes[2].len(), WEIGHT_TEXELS);
        assert!(p0.planes[2].iter().all(|&w| w == 255));
        assert!(p0.planes[0].iter().all(|&w| w == 0));
        assert!(p0.planes[1].iter().all(|&w| w == 0));
        assert!(p0.planes[3].iter().all(|&w| w == 0));
        assert!(!p0.is_multilayer());
        assert_eq!(alp.patch(0, 0).map(|p| p.layer_count), Some(1));
    }

    #[test]
    fn rejects_truncated_file() {
        assert!(matches!(
            AlpFile::read(&[0u8; 16]),
            Err(AlpError::TooSmall(_))
        ));
    }
}
