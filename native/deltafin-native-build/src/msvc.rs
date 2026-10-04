//! The MSVC side of the native build graph.
//!
//! The Unix graph spells every compiler decision in GCC/Clang flags. On
//! Windows the admitted toolchain is Microsoft's: `cl.exe` compiles, `lib.exe`
//! archives, `link.exe` links the isolated native tests, and rustc links the
//! product. This module owns the flag vocabulary and the tool discovery for
//! it, as pure functions of their inputs so every decision is unit-tested on
//! any host rather than only on the one machine that can run `cl.exe`.
//!
//! Arithmetic is a first-class concern. The exact kernels rest on explicit
//! intrinsics (including explicit fused multiply-add), never on what the
//! compiler decides to contract or reorder, so the flags here pin
//! `/fp:precise`, never enable `/fp:fast` or `/fp:contract`, and stop the
//! baseline at `/arch:AVX`: with no FMA in the baseline ISA, ordinary C code
//! cannot be fused behind the kernels' back, and the AVX2+FMA paths are
//! isolated to functions that call the intrinsics and are selected at run time
//! by CPUID. That is the same split the `-mavx -mfma` baseline plus
//! `target("avx2,fma")` attributes make on GCC and Clang.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

/// Which dialect of compiler command line a toolchain speaks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CompilerFlavor {
    /// GCC or Clang driver: `-O3 -c file -o out`.
    Gnu,
    /// MSVC `cl.exe`: `/O2 /c file /Fo:out`.
    Msvc,
}

/// `cl.exe`, `lib.exe` and `link.exe` from one Visual Studio installation,
/// with the environment (`PATH`, `INCLUDE`, `LIB`, ...) they need to run.
#[derive(Clone, Debug)]
pub(crate) struct MsvcTools {
    pub(crate) cl: PathBuf,
    pub(crate) lib: PathBuf,
    pub(crate) link: PathBuf,
    pub(crate) environment: Vec<(OsString, OsString)>,
}

/// Locate the MSVC tools for `target` (for example `x86_64-pc-windows-msvc`).
/// Discovery is `find-msvc-tools`, the dependency-free locator that Cargo's
/// own ecosystem uses: it honors an active developer prompt and otherwise asks
/// Visual Studio's setup configuration for the newest installation. Nothing
/// here consults `vswhere`, a batch file, or a shell.
pub(crate) fn discover(target: &str) -> Result<MsvcTools, String> {
    let find = |name: &str| {
        find_msvc_tools::find_tool(target, name).ok_or_else(|| {
            format!(
                "could not locate {name} for {target}: install Visual Studio Build Tools with the \
                 \"MSVC v14x x64/x86 build tools\" and \"Windows SDK\" components, or run from a \
                 Developer Command Prompt"
            )
        })
    };
    let cl = find("cl.exe")?;
    let lib = find("lib.exe")?;
    let link = find("link.exe")?;
    // The three come from one installation; their environments agree, so the
    // compiler's is the one applied to all of them.
    let environment = cl
        .env()
        .into_iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    Ok(MsvcTools {
        cl: cl.path().to_path_buf(),
        lib: lib.path().to_path_buf(),
        link: link.path().to_path_buf(),
        environment,
    })
}

/// The flavor an explicitly named compiler speaks, from its file name:
/// `cl` and `clang-cl` take MSVC options, everything else GNU ones.
pub(crate) fn flavor_of(compiler: &Path) -> CompilerFlavor {
    let stem = compiler
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if stem == "cl" || stem == "clang-cl" {
        CompilerFlavor::Msvc
    } else {
        CompilerFlavor::Gnu
    }
}

/// Whether the banner a compiler prints identifies Microsoft's compiler (or
/// its clang-cl driver) rather than something merely named `cl.exe`.
pub(crate) fn is_msvc_banner(banner: &str) -> bool {
    let lower = banner.to_ascii_lowercase();
    lower.contains("microsoft (r) c/c++ optimizing compiler") || lower.contains("clang")
}

fn arg(text: &str) -> OsString {
    OsString::from(text)
}

fn prefixed(prefix: &str, path: &Path) -> OsString {
    let mut value = OsString::from(prefix);
    value.push(path.as_os_str());
    value
}

/// Options every `cl.exe` invocation shares: no banner, the dynamic CRT that
/// LibTorch's DLLs use (mixing CRTs across a DLL boundary corrupts the heap),
/// strict IEEE-style floating point, and no inherited options.
fn common(args: &mut Vec<OsString>) {
    args.extend(
        [
            "/nologo",
            "/MD",
            // Stated even though it is the default, so a `CL` variable or a
            // future default cannot move the arithmetic.
            "/fp:precise",
            "/DNDEBUG",
            "/DNOMINMAX",
            "/DWIN32_LEAN_AND_MEAN",
            "/D_CRT_SECURE_NO_WARNINGS",
        ]
        .map(arg),
    );
}

/// Compile one C++ provider translation unit.
pub(crate) fn cpp_compile_args(
    source: &Path,
    object: &Path,
    torch_root: &Path,
    generated_include: Option<&Path>,
    cuda_include: Option<&Path>,
    definitions: &[&str],
) -> Vec<OsString> {
    let mut args = Vec::new();
    common(&mut args);
    args.extend(
        [
            "/O2",
            "/std:c++20",
            "/EHsc",
            // The headers are C++ as the standard defines it, and LibTorch's
            // are written against that reading.
            "/permissive-",
            "/Zc:__cplusplus",
            "/Zc:preprocessor",
            // Large templates (ATen) overflow the default section count.
            "/bigobj",
            "/W4",
            "/WX",
            // LibTorch's headers are the toolchain's, not ours: their
            // warnings are not this build's to fail on (`-isystem` on GNU).
            "/external:W0",
            "/c",
        ]
        .map(arg),
    );
    for definition in definitions {
        args.push(OsString::from(format!("/D{definition}")));
    }
    if let Some(include) = generated_include {
        args.push(prefixed("/I", include));
    }
    for include in [
        cuda_include.map(Path::to_path_buf),
        Some(torch_root.join("include")),
        Some(torch_root.join("include/torch/csrc/api/include")),
    ]
    .into_iter()
    .flatten()
    {
        args.push(prefixed("/external:I", &include));
    }
    args.push(source.as_os_str().to_owned());
    args.push(prefixed("/Fo:", object));
    args
}

/// Compile one C kernel translation unit. The baseline is `/arch:AVX`, the
/// analogue of the GNU `-mavx` baseline; AVX2 stays inside the kernels'
/// intrinsic-calling functions.
pub(crate) fn c_kernel_compile_args(source: &Path, object: &Path, x86_64: bool) -> Vec<OsString> {
    let mut args = Vec::new();
    common(&mut args);
    args.extend(
        [
            "/O2",
            "/std:c11",
            // MSVC gates C11 `_Atomic` behind this switch; the kernels'
            // synchronization is built on it.
            "/experimental:c11atomics",
            "/W4",
            "/WX",
            // `static inline` helpers a configuration does not call.
            "/wd4505",
            "/external:W0",
            "/c",
        ]
        .map(arg),
    );
    if x86_64 {
        args.push(arg("/arch:AVX"));
    }
    args.push(source.as_os_str().to_owned());
    args.push(prefixed("/Fo:", object));
    args
}

/// Compile one C test program or helper (no ISA baseline of its own).
pub(crate) fn c_test_compile_args(source: &Path, object: &Path) -> Vec<OsString> {
    let mut args = Vec::new();
    common(&mut args);
    args.extend(
        [
            "/O2",
            "/std:c11",
            "/experimental:c11atomics",
            "/W4",
            "/WX",
            "/external:W0",
            "/c",
        ]
        .map(arg),
    );
    args.push(source.as_os_str().to_owned());
    args.push(prefixed("/Fo:", object));
    args
}

/// Compile and link the Python-denial guard in one step.
pub(crate) fn guard_compile_args(source: &Path, object: &Path, executable: &Path) -> Vec<OsString> {
    let mut args: Vec<OsString> = [
        "/nologo",
        "/MD",
        "/O2",
        "/W4",
        "/WX",
        // The guard writes its marker with fopen.
        "/D_CRT_SECURE_NO_WARNINGS",
    ]
    .map(arg)
    .to_vec();
    args.push(source.as_os_str().to_owned());
    args.push(prefixed("/Fo:", object));
    args.push(prefixed("/Fe:", executable));
    args
}

/// Archive objects into a static library with `lib.exe`.
pub(crate) fn archive_args(archive: &Path, objects: &[PathBuf]) -> Vec<OsString> {
    let mut args = vec![arg("/NOLOGO"), prefixed("/OUT:", archive)];
    args.extend(objects.iter().map(|object| object.as_os_str().to_owned()));
    args
}

/// The name rustc and `link.exe` expect for a static library called `name`.
pub(crate) fn static_library_name(name: &str) -> String {
    format!("{name}.lib")
}

/// Link an isolated native test with `link.exe`: the test objects, the
/// provider archive, then LibTorch's import libraries, with an explicit
/// stack matching the product's.
pub(crate) fn link_args(
    objects: &[PathBuf],
    archive: Option<&Path>,
    executable: &Path,
    library_directories: &[&Path],
    import_libraries: &[&str],
    stack_bytes: u64,
) -> Vec<OsString> {
    let mut args = vec![
        arg("/NOLOGO"),
        arg("/SUBSYSTEM:CONSOLE"),
        arg("/INCREMENTAL:NO"),
        arg("/NXCOMPAT"),
        arg("/DYNAMICBASE"),
        OsString::from(format!("/STACK:{stack_bytes}")),
        prefixed("/OUT:", executable),
    ];
    for directory in library_directories {
        args.push(prefixed("/LIBPATH:", directory));
    }
    args.extend(objects.iter().map(|object| object.as_os_str().to_owned()));
    args.extend(archive.map(|archive| archive.as_os_str().to_owned()));
    args.extend(import_libraries.iter().map(|library| OsString::from(*library)));
    args
}

/// Options that make a Microsoft tool read configuration from the process
/// environment instead of the audited command line.
pub(crate) const INJECTION_VARIABLES: &[&str] = &["CL", "_CL_", "LINK", "_LINK_"];

/// A tool environment entry's name, compared the way Windows does.
pub(crate) fn same_variable(left: &OsStr, right: &OsStr) -> bool {
    left.to_string_lossy()
        .eq_ignore_ascii_case(&right.to_string_lossy())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[OsString]) -> Vec<String> {
        args.iter().map(|arg| arg.to_string_lossy().into_owned()).collect()
    }

    fn has(args: &[String], flag: &str) -> bool {
        args.iter().any(|arg| arg == flag)
    }

    /// Spellings that would let the compiler change a kernel's arithmetic.
    const ARITHMETIC_HAZARDS: &[&str] = &[
        "/fp:fast",
        "/fp:contract",
        "/Ofast",
        "/Qvec",
        "/arch:AVX2",
        "/arch:AVX512",
        "/favor:AMD64",
    ];

    fn every_kind() -> Vec<(&'static str, Vec<String>)> {
        let source = Path::new(r"C:\src\kernel.c");
        let object = Path::new(r"C:\out\kernel.obj");
        vec![
            (
                "cpp",
                strings(&cpp_compile_args(
                    Path::new(r"C:\src\provider.cpp"),
                    object,
                    Path::new(r"C:\torch"),
                    Some(Path::new(r"C:\out\generated")),
                    Some(Path::new(r"C:\cuda\include")),
                    &["DELTAFIN_HAVE_MXFP4_CPU_V1=1", "USE_RPC"],
                )),
            ),
            ("kernel", strings(&c_kernel_compile_args(source, object, true))),
            ("test", strings(&c_test_compile_args(source, object))),
            (
                "guard",
                strings(&guard_compile_args(source, object, Path::new(r"C:\out\deny.exe"))),
            ),
        ]
    }

    #[test]
    fn no_compile_ever_asks_for_looser_arithmetic_or_a_wider_baseline() {
        for (kind, args) in every_kind() {
            for hazard in ARITHMETIC_HAZARDS {
                assert!(!has(&args, hazard), "{kind} compile uses {hazard}: {args:?}");
            }
        }
    }

    #[test]
    fn every_compile_that_does_arithmetic_pins_precise_floating_point() {
        for (kind, args) in every_kind() {
            if kind == "guard" {
                continue;
            }
            assert!(has(&args, "/fp:precise"), "{kind}: {args:?}");
        }
    }

    #[test]
    fn every_compile_uses_the_dynamic_crt_libtorch_uses() {
        for (kind, args) in every_kind() {
            assert!(has(&args, "/MD"), "{kind}: {args:?}");
            assert!(!has(&args, "/MT") && !has(&args, "/MTd") && !has(&args, "/MDd"), "{kind}");
        }
    }

    #[test]
    fn warnings_are_errors_everywhere_the_gnu_graph_makes_them_so() {
        for (kind, args) in every_kind() {
            assert!(has(&args, "/WX") && has(&args, "/W4"), "{kind}: {args:?}");
        }
    }

    #[test]
    fn the_cpp_compile_names_its_inputs_and_treats_libtorch_as_external() {
        let args = strings(&cpp_compile_args(
            Path::new(r"C:\src\provider.cpp"),
            Path::new(r"C:\out\provider.obj"),
            Path::new(r"C:\torch"),
            Some(Path::new(r"C:\out\generated")),
            Some(Path::new(r"C:\cuda\include")),
            &["DELTAFIN_HAVE_MXFP4_CPU_V1=1", "USE_RPC"],
        ));
        let torch = Path::new(r"C:\torch");
        let external = |directory: &Path| format!("/external:I{}", directory.display());
        for expected in [
            "/std:c++20".to_owned(),
            "/EHsc".to_owned(),
            "/c".to_owned(),
            r"C:\src\provider.cpp".to_owned(),
            r"/Fo:C:\out\provider.obj".to_owned(),
            "/DDELTAFIN_HAVE_MXFP4_CPU_V1=1".to_owned(),
            "/DUSE_RPC".to_owned(),
            r"/IC:\out\generated".to_owned(),
            external(&torch.join("include")),
            external(&torch.join("include/torch/csrc/api/include")),
            external(Path::new(r"C:\cuda\include")),
            "/external:W0".to_owned(),
        ] {
            assert!(args.contains(&expected), "missing {expected}: {args:?}");
        }
        // The provider's own headers are not external: warnings there fail.
        assert!(!args.iter().any(|arg| arg == r"/external:IC:\out\generated"));
    }

    #[test]
    fn the_kernel_baseline_is_avx_and_only_on_x86_64() {
        let source = Path::new("k.c");
        let object = Path::new("k.obj");
        let x86 = strings(&c_kernel_compile_args(source, object, true));
        assert!(has(&x86, "/arch:AVX"));
        assert!(has(&x86, "/experimental:c11atomics") && has(&x86, "/std:c11"));
        let other = strings(&c_kernel_compile_args(source, object, false));
        assert!(!other.iter().any(|arg| arg.starts_with("/arch:")));
    }

    #[test]
    fn archives_are_lib_exe_invocations_with_the_msvc_library_name() {
        let args = strings(&archive_args(
            Path::new(r"C:\out\deltafin_provider_abi.lib"),
            &[PathBuf::from(r"C:\out\a.obj"), PathBuf::from(r"C:\out\b.obj")],
        ));
        assert_eq!(
            args,
            [
                "/NOLOGO",
                r"/OUT:C:\out\deltafin_provider_abi.lib",
                r"C:\out\a.obj",
                r"C:\out\b.obj"
            ]
        );
        assert_eq!(static_library_name("deltafin_provider_abi"), "deltafin_provider_abi.lib");
    }

    #[test]
    fn the_isolated_test_link_orders_objects_then_archive_then_import_libraries() {
        let args = strings(&link_args(
            &[PathBuf::from("main.obj"), PathBuf::from("extra.obj")],
            Some(Path::new("provider.lib")),
            Path::new("test.exe"),
            &[Path::new(r"C:\torch\lib")],
            &["torch.lib", "torch_cpu.lib", "c10.lib"],
            8 << 20,
        ));
        let position = |needle: &str| args.iter().position(|arg| arg == needle).unwrap();
        assert!(position("main.obj") < position("extra.obj"));
        assert!(position("extra.obj") < position("provider.lib"));
        assert!(position("provider.lib") < position("torch.lib"));
        assert!(position("torch.lib") < position("c10.lib"));
        assert!(has(&args, "/STACK:8388608"));
        assert!(has(&args, r"/LIBPATH:C:\torch\lib"));
        assert!(has(&args, "/OUT:test.exe"));
        assert!(has(&args, "/DYNAMICBASE") && has(&args, "/NXCOMPAT"));
    }

    #[test]
    fn compiler_flavor_follows_the_file_name() {
        // Joined, not spelled with one platform's separator: the file name is
        // what is under test, on every host.
        let in_bin = |name: &str| PathBuf::from("tools").join("bin").join(name);
        for (name, flavor) in [
            ("cl.exe", CompilerFlavor::Msvc),
            ("clang-cl.exe", CompilerFlavor::Msvc),
            ("CL.EXE", CompilerFlavor::Msvc),
            ("Clang-CL.exe", CompilerFlavor::Msvc),
            ("clang.exe", CompilerFlavor::Gnu),
            ("gcc.exe", CompilerFlavor::Gnu),
            ("cc", CompilerFlavor::Gnu),
            ("cl-wrapper.exe", CompilerFlavor::Gnu),
        ] {
            assert_eq!(flavor_of(&in_bin(name)), flavor, "{name}");
        }
    }

    #[test]
    fn only_microsofts_or_clangs_banner_passes_for_a_compiler() {
        assert!(is_msvc_banner(
            "Microsoft (R) C/C++ Optimizing Compiler Version 19.50.35725 for x64\nCopyright (C) Microsoft Corporation."
        ));
        assert!(is_msvc_banner("clang version 22.1.8\nTarget: x86_64-pc-windows-msvc"));
        assert!(!is_msvc_banner("usage: cl [ option... ] filename... [ /link linkoption... ]"));
        assert!(!is_msvc_banner(""));
    }

    // Visual Studio exists only on Windows; elsewhere the answer must be the
    // actionable one rather than a panic about a missing path.
    #[cfg(not(windows))]
    #[test]
    fn discovery_without_visual_studio_says_what_to_install() {
        let error = discover("x86_64-pc-windows-msvc").unwrap_err();
        assert!(error.contains("cl.exe"), "{error}");
        assert!(error.contains("Build Tools") && error.contains("Windows SDK"), "{error}");
        // A non-MSVC target is never the MSVC toolchain's to answer.
        assert!(discover("x86_64-unknown-linux-gnu").is_err());
    }

    #[test]
    fn variables_that_inject_options_are_known_and_compared_without_case() {
        for name in ["CL", "_CL_", "LINK", "_LINK_"] {
            assert!(INJECTION_VARIABLES.contains(&name));
        }
        assert!(same_variable(OsStr::new("Path"), OsStr::new("PATH")));
        assert!(!same_variable(OsStr::new("LIB"), OsStr::new("LIBPATH")));
    }
}
