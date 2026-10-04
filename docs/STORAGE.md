# Native storage preparation

The default resident spine is the measured **row-int8** representation. `deltafin setup` prepares it automatically at the end of installation by converting the downloaded BF16 checkpoint, so a fresh install runs the default with no separate step. The conversion is resumable, authenticated, and recognizes an already-complete output. Existing installations can produce it explicitly:

```bash
./target/release/deltafin convert-spine-int8
```

Int8 changes resident weights and is therefore still labeled **quantized and non-weight-exact** throughout the CLI. The released MXFP4 routed experts are untouched in every configuration — K3's native representation, all 16 always routed.

The **original BF16 spine remains on disk** as the conversion source and verification authority, and stays selectable explicitly:

```bash
./target/release/deltafin run --spine bf16 --prompt "The capital of France is" --max-new 17
```

While accuracy validation of the quantized default continues, selection never falls back silently: if the int8 spine is missing, `--spine auto` fails with instructions instead of substituting a different representation.

Either spine can be packed into authenticated DFSP files to reduce file discovery and make layer reads contiguous:

```bash
./target/release/deltafin pack-spine --spine int8
./target/release/deltafin pack-spine --spine int8 --verify-only
./target/release/deltafin pack-spine --spine bf16
```

Full local installs may losslessly compact only the MXFP4 scale streams:

```bash
./target/release/deltafin convert-experts-scale4
```

The resumable conversion adds about **40.25 GiB** of sidecars and keeps the raw experts. Activation is atomic only after the complete 82,432-expert corpus validates. Packed expert values are unchanged and scales reconstruct exactly.

Approximate disk footprint on a full install: 1.7 TB raw experts, 107 GB BF16 spine source, 53 GB int8 resident spine, plus optional scale4 sidecars and DFSP packs.

## Spreading reads over several drives

On a 64 GB host every decode pass streams the whole 53 GB int8 spine plus the
routed experts (about 25 GB per token at T=1, more when drafts are verified),
so one SSD's bandwidth sets the pace. Extra drives holding copies of those
files add bandwidth. Example: internal SSD plus Thunderbolt NVMe enclosures.

```bash
# Copy the spine, then the most-used experts, onto a drive (stop at 500 GB).
# Every copy is read back without the page cache and compared by SHA-256.
./target/release/deltafin populate-storage-home --dir /Volumes/NVMe1/deltafin --budget-gb 500

# Run with it. No speeds are needed: each drive's rate is measured as it reads.
K3_STORAGE_HOMES=/Volumes/NVMe1/deltafin,/Volumes/NVMe2/deltafin \
  ./target/release/deltafin run --prompt "The capital of France is" --max-new 17 --stats
```

How it works:
- Each read job is assigned when a worker starts it, so the chunks of one
  large file spread over every drive that holds it.
- All readers (spine, expert demand, expert prefetch) share one load counter
  per drive. Expert placement therefore sees spine traffic, and vice versa.
- A drive that slows down (thermal throttling) receives less work as its
  measured rate drops.
- A drive that fails a read (unplugged, I/O error) sits out with an increasing
  back-off, and the read is retried on another copy. Generation continues.
- `--stats` prints each drive's served bytes, measured rate and failures.

What gets used:
- Copies are admitted at startup only as regular, non-symlink files with
  exactly the primary's length and a modification time no older than the
  primary's. A primary rewritten after the copy marks that copy stale.
- Every read re-validates type and length on the live descriptor.
- Scale4 sidecar records stay on the model root's drive with their digest
  checks. Lazy installs still fetch into `<model>/k3-experts`; homes are
  read-only to the engine.
- A missing or empty home is skipped with a warning. Network shares, FAT-family
  volumes and copies on the model root's own disk are flagged.

Hardware guidance (from emulation; see OPTIMIZATIONS.md 2026-10-03):
- Use one enclosure per Thunderbolt port. On a MacBookPro18,2 each of the
  three ports is its own bus. Hubs and daisy chains share one bus.
- Spine plus the hottest ~30% of experts (about 500 GB per drive) performs
  almost exactly like a full mirror. Spine-only drives (53 GB) still help
  plain decoding but add little to draft verification.
- Format homes as APFS. `populate-storage-home` drops a
  `.metadata_never_index` marker; `sudo mdutil -i off /Volumes/NAME` also
  keeps Spotlight away.
- Prefer enclosures with a heatsink or fan: decode reads are sustained.
