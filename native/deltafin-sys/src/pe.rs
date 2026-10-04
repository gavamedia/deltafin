//! Bounded, non-executing parsing of Windows Portable Executable images.
//!
//! Like the Mach-O and ELF readers in `loader_audit`, this never loads or runs
//! the image it inspects; it reads a byte slice and answers structural
//! questions. It is platform-independent on purpose, so the checks that guard
//! a Windows install can be tested with synthetic images on any host.

use std::fmt;
use std::fs::File;
use std::io;

const DOS_MAGIC: &[u8; 2] = b"MZ";
const PE_SIGNATURE: &[u8; 4] = b"PE\0\0";
/// `e_lfanew`, the offset of the PE signature, lives at this DOS-header offset.
const E_LFANEW_OFFSET: usize = 0x3c;
const COFF_HEADER_BYTES: usize = 20;

pub const MACHINE_I386: u16 = 0x014c;
pub const MACHINE_AMD64: u16 = 0x8664;
pub const MACHINE_ARM64: u16 = 0xaa64;

const CHARACTERISTIC_EXECUTABLE_IMAGE: u16 = 0x0002;
const CHARACTERISTIC_DLL: u16 = 0x2000;
const OPTIONAL_MAGIC_PE32: u16 = 0x010b;
const OPTIONAL_MAGIC_PE32_PLUS: u16 = 0x020b;

/// What the headers of a PE image say it is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Headers {
    pub machine: u16,
    pub characteristics: u16,
    pub pe32_plus: bool,
}

impl Headers {
    pub const fn is_dll(self) -> bool {
        self.characteristics & CHARACTERISTIC_DLL != 0
    }

    pub const fn is_executable_image(self) -> bool {
        self.characteristics & CHARACTERISTIC_EXECUTABLE_IMAGE != 0
    }
}

fn u16_at(bytes: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        bytes.get(offset..offset.checked_add(2)?)?.try_into().ok()?,
    ))
}

fn u32_at(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?,
    ))
}

/// Parse the DOS header, PE signature, COFF header and optional-header magic.
/// `None` for anything that is not a well-formed PE image prefix, including a
/// script, an App Execution Alias stub, or a truncated file.
pub fn parse_headers(bytes: &[u8]) -> Option<Headers> {
    if bytes.get(..2)? != DOS_MAGIC {
        return None;
    }
    let signature = usize::try_from(u32_at(bytes, E_LFANEW_OFFSET)?).ok()?;
    // The signature cannot overlap the 64-byte DOS header.
    if signature < 0x40 || bytes.get(signature..signature.checked_add(4)?)? != PE_SIGNATURE {
        return None;
    }
    let coff = signature + 4;
    let machine = u16_at(bytes, coff)?;
    let optional_size = usize::from(u16_at(bytes, coff + 16)?);
    let characteristics = u16_at(bytes, coff + 18)?;
    let optional = coff + COFF_HEADER_BYTES;
    if optional_size < 2 {
        return None;
    }
    let pe32_plus = match u16_at(bytes, optional)? {
        OPTIONAL_MAGIC_PE32 => false,
        OPTIONAL_MAGIC_PE32_PLUS => true,
        _ => return None,
    };
    Some(Headers {
        machine,
        characteristics,
        pe32_plus,
    })
}

/// Whether the bytes begin a native, runnable program image for a CPU Windows
/// runs: a PE executable (not a DLL) for x86, x86-64 or ARM64.
pub fn is_native_executable(bytes: &[u8]) -> bool {
    parse_headers(bytes).is_some_and(|headers| {
        headers.is_executable_image()
            && !headers.is_dll()
            && matches!(
                headers.machine,
                MACHINE_I386 | MACHINE_AMD64 | MACHINE_ARM64
            )
    })
}

// ---------------------------------------------------------------------------
// Import tables
// ---------------------------------------------------------------------------

/// The most sections a PE file may declare (the format allows 96).
const MAX_SECTIONS: usize = 96;
/// The most import descriptors read from one table. A real image has dozens.
const MAX_IMPORT_DESCRIPTORS: usize = 4_096;
/// The longest DLL name accepted (`MAX_PATH`).
const MAX_DLL_NAME_BYTES: usize = 260;
/// The optional header is at most this large (PE32+ is 240 bytes).
const MAX_OPTIONAL_HEADER_BYTES: usize = 0x400;
/// The longest header prefix read to locate the PE signature.
const MAX_SIGNATURE_OFFSET: u64 = 1 << 20;
const IMPORT_DIRECTORY: usize = 1;
const DELAY_IMPORT_DIRECTORY: usize = 13;
const IMPORT_DESCRIPTOR_BYTES: usize = 20;
const DELAY_IMPORT_DESCRIPTOR_BYTES: usize = 32;
const SECTION_HEADER_BYTES: usize = 40;
/// `dlattrRva`: the delay descriptor's fields are RVAs (every modern linker).
const DELAY_ATTRIBUTE_RVA: u32 = 1;

/// Random-access, bounded reads of an image that may be gigabytes long. The
/// reader is `&mut` so a caller can account for every byte it is asked for.
pub trait ImageSource {
    fn len(&self) -> u64;
    fn read_exact_at(&mut self, buffer: &mut [u8], offset: u64) -> io::Result<()>;
}

/// An in-memory image.
pub struct Bytes<'a>(pub &'a [u8]);

impl ImageSource for Bytes<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_exact_at(&mut self, buffer: &mut [u8], offset: u64) -> io::Result<()> {
        let short = || io::Error::new(io::ErrorKind::UnexpectedEof, "read beyond the end of the image");
        let start = usize::try_from(offset).map_err(|_| short())?;
        let end = start.checked_add(buffer.len()).ok_or_else(short)?;
        buffer.copy_from_slice(self.0.get(start..end).ok_or_else(short)?);
        Ok(())
    }
}

impl ImageSource for File {
    fn len(&self) -> u64 {
        self.metadata().map_or(0, |metadata| metadata.len())
    }

    fn read_exact_at(&mut self, buffer: &mut [u8], offset: u64) -> io::Result<()> {
        crate::fs::FileExt::read_exact_at(&*self, buffer, offset)
    }
}

/// Why an image's import tables could not be trusted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MalformedImage(pub String);

impl fmt::Display for MalformedImage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for MalformedImage {}

fn malformed<T>(why: impl Into<String>) -> Result<T, MalformedImage> {
    Err(MalformedImage(why.into()))
}

/// The DLLs an image loads when it starts and those it may load later.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Imports {
    /// Load-time imports (`IMAGE_DIRECTORY_ENTRY_IMPORT`).
    pub load_time: Vec<String>,
    /// Delay-load imports (`IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT`).
    pub delay_load: Vec<String>,
}

impl Imports {
    pub fn all(&self) -> impl Iterator<Item = &str> {
        self.load_time
            .iter()
            .chain(self.delay_load.iter())
            .map(String::as_str)
    }
}

/// A parsed image: what its headers say it is and which DLLs it names.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Image {
    pub headers: Headers,
    pub imports: Imports,
}

#[derive(Clone, Copy)]
struct Section {
    virtual_address: u32,
    virtual_size: u32,
    raw_size: u32,
    raw_offset: u32,
}

struct Layout {
    headers: Headers,
    sections: Vec<Section>,
    headers_size: u32,
    directories: Vec<(u32, u32)>,
    length: u64,
}

impl Layout {
    /// The file offset of a relative virtual address, and how many bytes of
    /// file data remain in the region that holds it.
    fn locate(&self, rva: u32) -> Result<(u64, usize), MalformedImage> {
        if rva < self.headers_size {
            let offset = u64::from(rva);
            if offset >= self.length {
                return malformed(format!("RVA {rva:#x} maps beyond the end of the file"));
            }
            let in_headers = u64::from(self.headers_size - rva);
            return Ok((offset, in_headers.min(self.length - offset) as usize));
        }
        for section in &self.sections {
            let extent = section.virtual_size.max(section.raw_size);
            let in_section = rva >= section.virtual_address
                && section
                    .virtual_address
                    .checked_add(extent)
                    .is_some_and(|end| rva < end);
            if !in_section {
                continue;
            }
            let inside = rva - section.virtual_address;
            if inside >= section.raw_size {
                return malformed(format!("RVA {rva:#x} lies in uninitialized section data"));
            }
            let offset = u64::from(section.raw_offset) + u64::from(inside);
            if offset >= self.length {
                return malformed(format!("RVA {rva:#x} maps beyond the end of the file"));
            }
            let in_raw = u64::from(section.raw_size - inside);
            return Ok((offset, in_raw.min(self.length - offset) as usize));
        }
        malformed(format!("RVA {rva:#x} is in no section"))
    }
}

fn read_bytes<T: ImageSource + ?Sized>(
    source: &mut T,
    offset: u64,
    length: usize,
    what: &str,
) -> Result<Vec<u8>, MalformedImage> {
    let mut bytes = vec![0_u8; length];
    source
        .read_exact_at(&mut bytes, offset)
        .map_err(|error| MalformedImage(format!("cannot read {what} at {offset:#x}: {error}")))?;
    Ok(bytes)
}

fn le32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]])
}

fn layout<T: ImageSource + ?Sized>(source: &mut T) -> Result<Layout, MalformedImage> {
    let length = source.len();
    let dos = read_bytes(source, 0, 0x40, "the DOS header")?;
    let signature = u64::from(le32(&dos, E_LFANEW_OFFSET));
    let prefix_bytes = signature + 4 + COFF_HEADER_BYTES as u64 + 2;
    if prefix_bytes > length || prefix_bytes > MAX_SIGNATURE_OFFSET {
        return malformed("the PE signature offset is outside the image");
    }
    let prefix = read_bytes(source, 0, prefix_bytes as usize, "the PE headers")?;
    let Some(headers) = parse_headers(&prefix) else {
        return malformed("not a PE image");
    };
    let coff = signature as usize + 4;
    let section_count = usize::from(u16_at(&prefix, coff + 2).expect("bounded prefix"));
    let optional_size = usize::from(u16_at(&prefix, coff + 16).expect("bounded prefix"));
    if section_count == 0 || section_count > MAX_SECTIONS {
        return malformed(format!("implausible section count {section_count}"));
    }
    if !(2..=MAX_OPTIONAL_HEADER_BYTES).contains(&optional_size) {
        return malformed(format!("implausible optional header size {optional_size}"));
    }
    let optional_offset = signature + 4 + COFF_HEADER_BYTES as u64;
    let optional = read_bytes(source, optional_offset, optional_size, "the optional header")?;
    // SizeOfHeaders is at offset 60 in both formats. The data-directory count
    // and table follow the format-specific fixed part.
    let (count_offset, table_offset) = if headers.pe32_plus { (108, 112) } else { (92, 96) };
    if optional.len() < count_offset + 4 {
        return malformed("the optional header is too short for its data directories");
    }
    let headers_size = le32(&optional, 60);
    let declared = le32(&optional, count_offset) as usize;
    let available = (optional.len() - table_offset) / 8;
    let directories = (0..declared.min(16).min(available))
        .map(|index| {
            let at = table_offset + index * 8;
            (le32(&optional, at), le32(&optional, at + 4))
        })
        .collect();
    let sections_offset = optional_offset + optional_size as u64;
    let table = read_bytes(
        source,
        sections_offset,
        section_count * SECTION_HEADER_BYTES,
        "the section table",
    )?;
    let sections = table
        .chunks_exact(SECTION_HEADER_BYTES)
        .map(|entry| Section {
            virtual_size: le32(entry, 8),
            virtual_address: le32(entry, 12),
            raw_size: le32(entry, 16),
            raw_offset: le32(entry, 20),
        })
        .collect();
    Ok(Layout {
        headers,
        sections,
        headers_size,
        directories,
        length,
    })
}

fn dll_name<T: ImageSource + ?Sized>(
    source: &mut T,
    layout: &Layout,
    rva: u32,
) -> Result<String, MalformedImage> {
    if rva == 0 {
        return malformed("an import descriptor has no DLL name");
    }
    let (offset, available) = layout.locate(rva)?;
    // A name is at most MAX_DLL_NAME_BYTES long plus its terminator, and must
    // end inside the data the section actually holds.
    let take = available.min(MAX_DLL_NAME_BYTES + 1);
    let bytes = read_bytes(source, offset, take, "a DLL name")?;
    let Some(end) = bytes.iter().position(|byte| *byte == 0) else {
        return malformed("an unterminated or overlong DLL name");
    };
    let name = &bytes[..end];
    if name.is_empty() || !name.iter().all(|byte| (0x20..0x7f).contains(byte)) {
        return malformed("a DLL name is empty or not printable ASCII");
    }
    Ok(String::from_utf8_lossy(name).into_owned())
}

fn table_names<T: ImageSource + ?Sized>(
    source: &mut T,
    layout: &Layout,
    directory: usize,
    descriptor_bytes: usize,
    name_field: usize,
    delay: bool,
) -> Result<Vec<String>, MalformedImage> {
    let Some(&(rva, size)) = layout.directories.get(directory) else {
        return Ok(Vec::new());
    };
    if rva == 0 || size == 0 {
        return Ok(Vec::new());
    }
    let (first, _) = layout.locate(rva)?;
    let declared = (size as usize / descriptor_bytes).min(MAX_IMPORT_DESCRIPTORS);
    let mut names = Vec::new();
    // One past the declared size: a linker that leaves the terminating zero
    // descriptor out of the directory size is read correctly too.
    for index in 0..=declared {
        let at = first + (index * descriptor_bytes) as u64;
        let descriptor = read_bytes(source, at, descriptor_bytes, "an import descriptor")?;
        if descriptor.iter().all(|byte| *byte == 0) {
            return Ok(names);
        }
        if index == declared {
            break;
        }
        if delay && le32(&descriptor, 0) & DELAY_ATTRIBUTE_RVA == 0 {
            return malformed("a delay-import descriptor uses virtual addresses, not RVAs");
        }
        names.push(dll_name(source, layout, le32(&descriptor, name_field))?);
    }
    malformed("an import table is not terminated within its declared size")
}

/// Read the headers and the DLL names an image imports, bounded and without
/// executing or mapping it. Only the headers, the section table and the import
/// tables are read, however large the image is.
pub fn parse_image<T: ImageSource + ?Sized>(source: &mut T) -> Result<Image, MalformedImage> {
    let layout = layout(source)?;
    let load_time = table_names(
        source,
        &layout,
        IMPORT_DIRECTORY,
        IMPORT_DESCRIPTOR_BYTES,
        12,
        false,
    )?;
    let delay_load = table_names(
        source,
        &layout,
        DELAY_IMPORT_DIRECTORY,
        DELAY_IMPORT_DESCRIPTOR_BYTES,
        4,
        true,
    )?;
    Ok(Image {
        headers: layout.headers,
        imports: Imports {
            load_time,
            delay_load,
        },
    })
}

/// Read just the import names of an in-memory image.
pub fn parse_imports(bytes: &[u8]) -> Result<Imports, MalformedImage> {
    parse_image(&mut Bytes(bytes)).map(|image| image.imports)
}

/// Synthetic PE images for tests, here and in the crates that audit images.
/// They are structurally valid, never runnable.
#[cfg(any(test, feature = "testing"))]
pub mod testing {
    use super::*;

    const SIGNATURE: usize = 0x80;
    const HEADERS_SIZE: u32 = 0x200;
    const SECTION_RVA: u32 = 0x1000;

    /// A PE image with one section holding the import tables.
    #[derive(Clone, Debug)]
    pub struct SyntheticImage {
        pub machine: u16,
        pub characteristics: u16,
        pub pe32_plus: bool,
        pub imports: Vec<Vec<u8>>,
        pub delay_imports: Vec<Vec<u8>>,
    }

    impl SyntheticImage {
        /// An x86-64 console program.
        pub fn executable() -> Self {
            Self {
                machine: MACHINE_AMD64,
                characteristics: CHARACTERISTIC_EXECUTABLE_IMAGE | 0x0020,
                pe32_plus: true,
                imports: Vec::new(),
                delay_imports: Vec::new(),
            }
        }

        /// An x86-64 DLL.
        pub fn dll() -> Self {
            Self {
                characteristics: CHARACTERISTIC_EXECUTABLE_IMAGE | CHARACTERISTIC_DLL | 0x0020,
                ..Self::executable()
            }
        }

        pub fn importing(mut self, names: &[&str]) -> Self {
            self.imports = names.iter().map(|name| name.as_bytes().to_vec()).collect();
            self
        }

        pub fn delay_importing(mut self, names: &[&str]) -> Self {
            self.delay_imports = names.iter().map(|name| name.as_bytes().to_vec()).collect();
            self
        }

        /// Where the tables sit, for tests that corrupt a specific field:
        /// `(import_descriptors, delay_descriptors)` file offsets.
        pub fn table_offsets(&self) -> (usize, usize) {
            let imports = HEADERS_SIZE as usize;
            (imports, imports + (self.imports.len() + 1) * IMPORT_DESCRIPTOR_BYTES)
        }

        /// Offset of the optional header's data-directory table.
        pub fn directory_table_offset(&self) -> usize {
            SIGNATURE + 4 + COFF_HEADER_BYTES + if self.pe32_plus { 112 } else { 96 }
        }

        pub fn build(&self) -> Vec<u8> {
            let optional_size: usize = if self.pe32_plus { 240 } else { 224 };
            let (import_at, delay_at) = self.table_offsets();
            let names_at = delay_at + (self.delay_imports.len() + 1) * DELAY_IMPORT_DESCRIPTOR_BYTES;
            let mut name_offsets = Vec::new();
            let mut cursor = names_at;
            for name in self.imports.iter().chain(self.delay_imports.iter()) {
                name_offsets.push(cursor);
                cursor += name.len() + 1;
            }
            let raw_size = (cursor - HEADERS_SIZE as usize).div_ceil(0x200) * 0x200;
            let mut image = vec![0_u8; HEADERS_SIZE as usize + raw_size];
            let rva = |file_offset: usize| SECTION_RVA + (file_offset - HEADERS_SIZE as usize) as u32;
            let put16 = |image: &mut Vec<u8>, at: usize, value: u16| {
                image[at..at + 2].copy_from_slice(&value.to_le_bytes());
            };
            let put32 = |image: &mut Vec<u8>, at: usize, value: u32| {
                image[at..at + 4].copy_from_slice(&value.to_le_bytes());
            };

            image[..2].copy_from_slice(DOS_MAGIC);
            put32(&mut image, E_LFANEW_OFFSET, SIGNATURE as u32);
            image[SIGNATURE..SIGNATURE + 4].copy_from_slice(PE_SIGNATURE);
            let coff = SIGNATURE + 4;
            put16(&mut image, coff, self.machine);
            put16(&mut image, coff + 2, 1);
            put16(&mut image, coff + 16, optional_size as u16);
            put16(&mut image, coff + 18, self.characteristics);
            let optional = coff + COFF_HEADER_BYTES;
            put16(
                &mut image,
                optional,
                if self.pe32_plus { OPTIONAL_MAGIC_PE32_PLUS } else { OPTIONAL_MAGIC_PE32 },
            );
            put32(&mut image, optional + 60, HEADERS_SIZE);
            let (count_at, table_at) = if self.pe32_plus { (108, 112) } else { (92, 96) };
            put32(&mut image, optional + count_at, 16);
            if !self.imports.is_empty() {
                put32(&mut image, optional + table_at + 8, rva(import_at));
                put32(
                    &mut image,
                    optional + table_at + 12,
                    ((self.imports.len() + 1) * IMPORT_DESCRIPTOR_BYTES) as u32,
                );
            }
            if !self.delay_imports.is_empty() {
                put32(&mut image, optional + table_at + 13 * 8, rva(delay_at));
                put32(
                    &mut image,
                    optional + table_at + 13 * 8 + 4,
                    ((self.delay_imports.len() + 1) * DELAY_IMPORT_DESCRIPTOR_BYTES) as u32,
                );
            }

            let section = optional + optional_size;
            image[section..section + 8].copy_from_slice(b".idata\0\0");
            put32(&mut image, section + 8, raw_size as u32);
            put32(&mut image, section + 12, SECTION_RVA);
            put32(&mut image, section + 16, raw_size as u32);
            put32(&mut image, section + 20, HEADERS_SIZE);

            for (index, _) in self.imports.iter().enumerate() {
                let descriptor = import_at + index * IMPORT_DESCRIPTOR_BYTES;
                put32(&mut image, descriptor + 12, rva(name_offsets[index]));
                put32(&mut image, descriptor + 16, SECTION_RVA);
            }
            for (index, _) in self.delay_imports.iter().enumerate() {
                let descriptor = delay_at + index * DELAY_IMPORT_DESCRIPTOR_BYTES;
                put32(&mut image, descriptor, DELAY_ATTRIBUTE_RVA);
                put32(&mut image, descriptor + 4, rva(name_offsets[self.imports.len() + index]));
            }
            for (name, offset) in self
                .imports
                .iter()
                .chain(self.delay_imports.iter())
                .zip(name_offsets)
            {
                image[offset..offset + name.len()].copy_from_slice(name);
            }
            image
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// A minimal, structurally valid PE header block for tests.
    pub fn synthetic_image(machine: u16, characteristics: u16, magic: u16) -> Vec<u8> {
        let signature = 0x80_usize;
        let mut image = vec![0_u8; signature + 4 + COFF_HEADER_BYTES + 240];
        image[..2].copy_from_slice(DOS_MAGIC);
        image[E_LFANEW_OFFSET..E_LFANEW_OFFSET + 4].copy_from_slice(&(signature as u32).to_le_bytes());
        image[signature..signature + 4].copy_from_slice(PE_SIGNATURE);
        let coff = signature + 4;
        image[coff..coff + 2].copy_from_slice(&machine.to_le_bytes());
        image[coff + 16..coff + 18].copy_from_slice(&240_u16.to_le_bytes());
        image[coff + 18..coff + 20].copy_from_slice(&characteristics.to_le_bytes());
        let optional = coff + COFF_HEADER_BYTES;
        image[optional..optional + 2].copy_from_slice(&magic.to_le_bytes());
        image
    }

    #[test]
    fn a_well_formed_executable_is_recognized() {
        let image = synthetic_image(MACHINE_AMD64, 0x0022, OPTIONAL_MAGIC_PE32_PLUS);
        let headers = parse_headers(&image).unwrap();
        assert_eq!(headers.machine, MACHINE_AMD64);
        assert!(headers.pe32_plus && headers.is_executable_image() && !headers.is_dll());
        assert!(is_native_executable(&image));
        for machine in [MACHINE_I386, MACHINE_ARM64] {
            let image = synthetic_image(machine, 0x0102, OPTIONAL_MAGIC_PE32);
            assert!(is_native_executable(&image), "{machine:#x}");
        }
    }

    #[test]
    fn a_dll_is_an_image_but_not_a_program() {
        let image = synthetic_image(MACHINE_AMD64, 0x2022, OPTIONAL_MAGIC_PE32_PLUS);
        assert!(parse_headers(&image).unwrap().is_dll());
        assert!(!is_native_executable(&image));
    }

    #[test]
    fn non_programs_are_rejected() {
        // Scripts and shims: a shebang, a batch file, an empty file.
        for bytes in [&b"#!/bin/sh\n"[..], b"@echo off\r\n", b"", b"M", b"MZ"] {
            assert!(!is_native_executable(bytes));
        }
        // Wrong CPU, not executable, bad magic, bad signature.
        assert!(!is_native_executable(&synthetic_image(0x01c0, 0x0002, OPTIONAL_MAGIC_PE32)));
        assert!(!is_native_executable(&synthetic_image(MACHINE_AMD64, 0x0000, OPTIONAL_MAGIC_PE32_PLUS)));
        assert!(!is_native_executable(&synthetic_image(MACHINE_AMD64, 0x0002, 0x0107)));
        let mut image = synthetic_image(MACHINE_AMD64, 0x0002, OPTIONAL_MAGIC_PE32_PLUS);
        image[0x80] = b'X';
        assert!(!is_native_executable(&image));
    }

    #[test]
    fn hostile_offsets_are_bounded_not_trusted() {
        let mut image = synthetic_image(MACHINE_AMD64, 0x0002, OPTIONAL_MAGIC_PE32_PLUS);
        // A signature offset past the end, inside the DOS header, and at the
        // overflow edge must all be refused without panicking.
        for offset in [u32::MAX, 0xffff_fff0, 0, 0x3c, 0x3f, image.len() as u32 - 2] {
            image[E_LFANEW_OFFSET..E_LFANEW_OFFSET + 4].copy_from_slice(&offset.to_le_bytes());
            assert!(!is_native_executable(&image), "offset {offset:#x}");
        }
        // Truncation at every length never panics and never succeeds early.
        let whole = synthetic_image(MACHINE_AMD64, 0x0002, OPTIONAL_MAGIC_PE32_PLUS);
        for length in 0..0x80 + 4 + COFF_HEADER_BYTES + 2 {
            assert!(!is_native_executable(&whole[..length]), "length {length}");
        }
        assert!(is_native_executable(&whole[..0x80 + 4 + COFF_HEADER_BYTES + 2]));
    }

    mod imports {
        use super::super::testing::SyntheticImage;
        use super::super::*;

        const SYSTEM: &[&str] = &["KERNEL32.dll", "ucrtbase.dll", "c10.dll", "torch_cpu.dll"];

        fn imports_of(image: &[u8]) -> Result<Imports, MalformedImage> {
            parse_imports(image)
        }

        #[test]
        fn load_time_imports_are_listed_in_order_for_both_formats() {
            for pe32_plus in [true, false] {
                let mut image = SyntheticImage::executable().importing(SYSTEM);
                image.pe32_plus = pe32_plus;
                let imports = imports_of(&image.build()).unwrap();
                assert_eq!(imports.load_time, SYSTEM, "pe32_plus={pe32_plus}");
                assert!(imports.delay_load.is_empty());
            }
        }

        #[test]
        fn delay_load_imports_are_kept_apart_from_load_time_ones() {
            let image = SyntheticImage::dll()
                .importing(&["kernel32.dll"])
                .delay_importing(&["winhttp.dll", "ws2_32.dll"])
                .build();
            let parsed = parse_image(&mut Bytes(&image)).unwrap();
            assert!(parsed.headers.is_dll());
            assert_eq!(parsed.imports.load_time, ["kernel32.dll"]);
            assert_eq!(parsed.imports.delay_load, ["winhttp.dll", "ws2_32.dll"]);
            assert_eq!(
                parsed.imports.all().collect::<Vec<_>>(),
                ["kernel32.dll", "winhttp.dll", "ws2_32.dll"]
            );
        }

        #[test]
        fn an_image_with_no_tables_imports_nothing() {
            let image = SyntheticImage::executable().build();
            assert_eq!(imports_of(&image).unwrap(), Imports::default());
        }

        #[test]
        fn a_directory_size_that_leaves_out_the_terminator_still_parses() {
            let model = SyntheticImage::executable().importing(&["a.dll", "b.dll"]);
            let mut image = model.build();
            let size_at = model.directory_table_offset() + 8 + 4;
            // Declare exactly two descriptors; the zero terminator follows.
            image[size_at..size_at + 4].copy_from_slice(&(2 * 20_u32).to_le_bytes());
            assert_eq!(imports_of(&image).unwrap().load_time, ["a.dll", "b.dll"]);
        }

        #[test]
        fn a_table_that_never_terminates_is_refused() {
            let model = SyntheticImage::executable().importing(&["a.dll", "b.dll"]);
            let mut image = model.build();
            let (table, _) = model.table_offsets();
            let first_name = u32::from_le_bytes(image[table + 12..table + 16].try_into().unwrap());
            // The declared size covers three descriptors; make the "terminator"
            // and the one read past it ordinary, nonzero descriptors.
            for descriptor in [table + 40, table + 60] {
                image[descriptor..descriptor + 4].copy_from_slice(&1_u32.to_le_bytes());
                image[descriptor + 12..descriptor + 16].copy_from_slice(&first_name.to_le_bytes());
            }
            let error = imports_of(&image).unwrap_err();
            assert!(error.to_string().contains("not terminated"), "{error}");
        }

        #[test]
        fn hostile_names_are_refused() {
            for name in [
                &b""[..],
                b"bad\x01name.dll",
                b"caf\xc3\xa9.dll",
                b"tab\there.dll",
            ] {
                let mut model = SyntheticImage::executable();
                model.imports = vec![name.to_vec()];
                assert!(imports_of(&model.build()).is_err(), "{name:?}");
            }
            // Overlong: no terminator within MAX_DLL_NAME_BYTES.
            let mut model = SyntheticImage::executable();
            model.imports = vec![vec![b'a'; 300]];
            assert!(imports_of(&model.build()).is_err());
            // Longest accepted.
            let mut model = SyntheticImage::executable();
            model.imports = vec![vec![b'a'; 260]];
            assert_eq!(imports_of(&model.build()).unwrap().load_time[0].len(), 260);
        }

        #[test]
        fn a_delay_descriptor_with_virtual_addresses_is_refused() {
            let model = SyntheticImage::executable().delay_importing(&["a.dll"]);
            let mut image = model.build();
            let (_, delay) = model.table_offsets();
            image[delay..delay + 4].copy_from_slice(&0_u32.to_le_bytes());
            assert!(imports_of(&image).is_err());
        }

        #[test]
        fn headers_that_lie_are_bounded_and_never_trusted() {
            let model = SyntheticImage::executable().importing(&["a.dll"]).delay_importing(&["b.dll"]);
            let good = model.build();
            let coff = 0x80 + 4;
            let optional = coff + COFF_HEADER_BYTES;
            let directory = model.directory_table_offset();
            let section = optional + 240;
            let cases: Vec<(&str, Box<dyn Fn(&mut Vec<u8>)>)> = vec![
                ("no sections", Box::new(move |image| image[coff + 2..coff + 4].copy_from_slice(&0_u16.to_le_bytes()))),
                ("97 sections", Box::new(move |image| image[coff + 2..coff + 4].copy_from_slice(&97_u16.to_le_bytes()))),
                ("65535 sections", Box::new(move |image| image[coff + 2..coff + 4].copy_from_slice(&u16::MAX.to_le_bytes()))),
                ("optional header of 1 byte", Box::new(move |image| image[coff + 16..coff + 18].copy_from_slice(&1_u16.to_le_bytes()))),
                ("optional header of 64 KiB", Box::new(move |image| image[coff + 16..coff + 18].copy_from_slice(&u16::MAX.to_le_bytes()))),
                ("import RVA in no section", Box::new(move |image| image[directory + 8..directory + 12].copy_from_slice(&0x00ff_0000_u32.to_le_bytes()))),
                ("import RVA at u32::MAX", Box::new(move |image| image[directory + 8..directory + 12].copy_from_slice(&u32::MAX.to_le_bytes()))),
                ("section start near u32::MAX", Box::new(move |image| image[section + 12..section + 16].copy_from_slice(&(u32::MAX - 8).to_le_bytes()))),
                ("raw data past the file", Box::new(move |image| image[section + 20..section + 24].copy_from_slice(&0x7fff_0000_u32.to_le_bytes()))),
                ("raw size zero", Box::new(move |image| image[section + 16..section + 20].copy_from_slice(&0_u32.to_le_bytes()))),
                ("PE signature moved out", Box::new(|image| image[0x3c..0x40].copy_from_slice(&u32::MAX.to_le_bytes()))),
                ("not an image", Box::new(|image| image[0] = b'Z')),
            ];
            for (label, mutate) in cases {
                let mut image = good.clone();
                mutate(&mut image);
                assert!(imports_of(&image).is_err(), "{label} was accepted");
            }
            // Directory counts that lie about their size are bounded, not honored.
            let mut image = good.clone();
            image[directory + 8 + 4..directory + 8 + 8].copy_from_slice(&u32::MAX.to_le_bytes());
            assert_eq!(imports_of(&image).unwrap().load_time, ["a.dll"]);
            // A data-directory count that is too small hides the tables.
            let mut image = good.clone();
            let count_at = optional + 108;
            image[count_at..count_at + 4].copy_from_slice(&1_u32.to_le_bytes());
            assert_eq!(imports_of(&image).unwrap(), Imports::default());
            let mut image = good;
            image[count_at..count_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
            assert_eq!(imports_of(&image).unwrap().load_time, ["a.dll"]);
        }

        #[test]
        fn truncation_never_panics_and_never_misreads() {
            let model = SyntheticImage::executable()
                .importing(&["kernel32.dll", "torch_cpu.dll"])
                .delay_importing(&["winhttp.dll"]);
            let whole = model.build();
            let expected = imports_of(&whole).unwrap();
            for length in 0..whole.len() {
                if let Ok(imports) = imports_of(&whole[..length]) {
                    // A prefix that parses at all must parse to the truth.
                    assert_eq!(imports, expected, "length {length}");
                }
            }
            for length in 0..0x200 {
                assert!(imports_of(&whole[..length]).is_err(), "length {length}");
            }
        }

        #[test]
        fn a_file_is_parsed_through_positional_reads() {
            let image = SyntheticImage::executable()
                .importing(&["kernel32.dll"])
                .delay_importing(&["winhttp.dll"])
                .build();
            let path = std::env::temp_dir()
                .join(format!("deltafin-pe-imports-{}.exe", std::process::id()));
            std::fs::write(&path, &image).unwrap();
            let mut file = std::fs::File::open(&path).unwrap();
            let parsed = parse_image(&mut file).unwrap();
            std::fs::remove_file(&path).unwrap();
            assert_eq!(parsed.imports.load_time, ["kernel32.dll"]);
            assert_eq!(parsed.imports.delay_load, ["winhttp.dll"]);
            assert!(parsed.headers.is_executable_image());
        }
    }
}
