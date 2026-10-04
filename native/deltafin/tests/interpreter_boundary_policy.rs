//! Static release policy for every process edge owned by the compiled product.
//!
//! This test parses Rust syntax instead of grepping source text.  Every
//! production `Command::new` and terminal process method has to remain in the
//! reviewed allowlist below; comments, strings, and `#[cfg(test)]` fixtures do
//! not create false positives.  Public shell examples and native provider
//! sources are audited separately without executing any of them.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use deltafin_native_build::PRODUCTION_PROVIDER_SOURCES;
use syn::visit::{self, Visit};
use syn::{
    Attribute, Expr, ExprCall, ExprMethodCall, ForeignItemFn, ImplItemFn, ItemConst, ItemFn,
    ItemImpl, ItemMod, ItemStatic, ItemType, ItemUse, Local, Macro, UseTree,
};

const PUBLIC_DOCUMENTS: &[&str] = &[
    "README.md",
    "docs/OPTIMIZATIONS.md",
    "docs/COMPILED-RUNTIME.md",
    "docs/REQUIREMENTS.md",
    "docs/THIRD_PARTY_NOTICES.md",
    "native/deltafin-curl-sys-direct/README.md",
    "native/deltafin/TOKENIZER_AUDIT.md",
    "native/provider_gate/README.md",
    "tools/README.md",
];

const INTERNAL_PACKAGES: &[&str] = &[
    "native/deltafin",
    "native/deltafin-bootstrap",
    "native/deltafin-curl-sys-direct",
    "native/deltafin-native-build",
    "native/deltafin-sys",
    "native/deltafin-xtask",
];

const FORBIDDEN_PROCESS_OR_LOADER_APIS: &[&str] = &[
    "system",
    "popen",
    "execl",
    "execlp",
    "execle",
    "execv",
    "execvp",
    "execvpe",
    "posix_spawn",
    "posix_spawnp",
    "fork",
    "vfork",
    "dlopen",
    "dlmopen",
    "LoadLibraryA",
    "LoadLibraryW",
    "LoadLibraryExA",
    "LoadLibraryExW",
    "GetProcAddress",
];

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("native/deltafin must remain two levels below the repository")
        .to_path_buf()
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| {
        panic!(
            "read interpreter-boundary policy input {}: {error}",
            path.display()
        )
    })
}

/// A repository-relative path spelled with `/` on every host, so it compares
/// equal to the inventories and classifications this file keeps as strings.
fn slash_relative(relative: &Path) -> String {
    relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn walk_rust(root: &Path, files: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = fs::read_dir(root)
        .unwrap_or_else(|error| panic!("scan Rust policy root {}: {error}", root.display()))
        .map(|entry| entry.expect("read Rust policy entry"))
        .collect();
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let kind = entry
            .file_type()
            .unwrap_or_else(|error| panic!("inspect {}: {error}", path.display()));
        if kind.is_dir() {
            walk_rust(&path, files);
        } else if kind.is_file() && path.extension().and_then(|value| value.to_str()) == Some("rs")
        {
            files.push(path);
        }
    }
}

fn is_test_only(attributes: &[Attribute]) -> bool {
    attributes.iter().any(|attribute| {
        if attribute.path().is_ident("test") {
            return true;
        }
        let syn::Meta::List(list) = &attribute.meta else {
            return false;
        };
        list.path.is_ident("cfg")
            && cfg_predicate_requires_test(
                &list
                    .tokens
                    .to_string()
                    .split_whitespace()
                    .collect::<String>(),
            )
    })
}

/// Whether a `cfg` predicate can only hold in a test build: `test` itself, or
/// an `all(...)` conjunction with such a term (`all(test, unix)`). `any`, `not`
/// and everything else may hold outside tests, so they are production code.
fn cfg_predicate_requires_test(predicate: &str) -> bool {
    if predicate == "test" {
        return true;
    }
    let Some(inner) = predicate
        .strip_prefix("all(")
        .and_then(|rest| rest.strip_suffix(')'))
    else {
        return false;
    };
    let mut depth = 0_usize;
    let mut start = 0;
    let mut terms = Vec::new();
    for (index, character) in inner.char_indices() {
        match character {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                terms.push(&inner[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    terms.push(&inner[start..]);
    terms.into_iter().any(cfg_predicate_requires_test)
}

#[test]
fn test_only_cfgs_are_recognized_through_conjunctions_only() {
    for predicate in ["test", "all(test,unix)", "all(unix,test)", "all(unix,all(test,windows))"] {
        assert!(cfg_predicate_requires_test(predicate), "{predicate}");
    }
    for predicate in ["unix", "any(test,unix)", "not(test)", "all(unix,windows)", "feature=\"runtime\"", "all(any(test,unix),windows)"] {
        assert!(!cfg_predicate_requires_test(predicate), "{predicate}");
    }
}

fn type_label(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(path) => path
            .path
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect::<Vec<_>>()
            .join("::"),
        syn::Type::Reference(reference) => type_label(&reference.elem),
        _ => "<impl>".to_owned(),
    }
}

fn path_label(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

fn path_last_owned(path: &syn::Path) -> Option<String> {
    path.segments
        .last()
        .map(|segment| segment.ident.to_string())
}

fn path_is_command_constructor(path: &syn::Path) -> bool {
    let segments: Vec<_> = path.segments.iter().collect();
    segments.len() >= 2
        && segments[segments.len() - 2].ident == "Command"
        && segments[segments.len() - 1].ident == "new"
}

fn type_is_command(ty: &syn::Type) -> bool {
    matches!(
        ty,
        syn::Type::Path(path)
            if path.path.segments.last().is_some_and(|segment| segment.ident == "Command")
    )
}

fn forbidden_link_name(attributes: &[Attribute]) -> Option<String> {
    attributes.iter().find_map(|attribute| {
        if !attribute.path().is_ident("link_name") {
            return None;
        }
        let syn::Meta::NameValue(name_value) = &attribute.meta else {
            return None;
        };
        let Expr::Lit(literal) = &name_value.value else {
            return None;
        };
        let syn::Lit::Str(value) = &literal.lit else {
            return None;
        };
        FORBIDDEN_PROCESS_OR_LOADER_APIS
            .contains(&value.value().as_str())
            .then(|| value.value())
    })
}

fn expression_label(expression: &Expr) -> String {
    match expression {
        Expr::Path(path) => path_label(&path.path),
        Expr::Reference(reference) => format!("&{}", expression_label(&reference.expr)),
        Expr::Field(field) => {
            let member = match &field.member {
                syn::Member::Named(name) => name.to_string(),
                syn::Member::Unnamed(index) => index.index.to_string(),
            };
            format!("{}.{}", expression_label(&field.base), member)
        }
        Expr::Paren(paren) => expression_label(&paren.expr),
        Expr::Lit(literal) => match &literal.lit {
            syn::Lit::Str(value) => format!("{:?}", value.value()),
            _ => "<literal>".to_owned(),
        },
        _ => "<expression>".to_owned(),
    }
}

fn expression_path(expression: &Expr) -> Option<&syn::Path> {
    match expression {
        Expr::Path(path) => Some(&path.path),
        Expr::Group(group) => expression_path(&group.expr),
        Expr::Paren(paren) => expression_path(&paren.expr),
        Expr::Reference(reference) => expression_path(&reference.expr),
        _ => None,
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Launch {
    file: String,
    context: String,
    argument: String,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ProcessMethod {
    file: String,
    context: String,
    method: String,
}

#[derive(Default)]
struct RustProcessAudit {
    file: String,
    context: Option<String>,
    implementation: Option<String>,
    process_scope: bool,
    launches: BTreeMap<Launch, usize>,
    terminal_methods: BTreeMap<ProcessMethod, usize>,
    facts: BTreeMap<(String, String), BTreeSet<String>>,
    forbidden: Vec<String>,
}

impl RustProcessAudit {
    fn context(&self) -> String {
        self.context
            .clone()
            .unwrap_or_else(|| "<module>".to_owned())
    }

    fn record_fact(&mut self, fact: String) {
        self.facts
            .entry((self.file.clone(), self.context()))
            .or_default()
            .insert(fact);
    }

    fn enter_function(&mut self, name: String, visit: impl FnOnce(&mut Self)) {
        let previous = self.context.replace(name);
        visit(self);
        self.context = previous;
    }

    fn reject_process_alias_expression(&mut self, kind: &str, expression: &Expr) {
        let Some(path) = expression_path(expression) else {
            return;
        };
        let name = path_last_owned(path).unwrap_or_default();
        if FORBIDDEN_PROCESS_OR_LOADER_APIS.contains(&name.as_str())
            || path_is_command_constructor(path)
        {
            self.forbidden.push(format!(
                "{} in {} creates {kind} alias for forbidden process/loader edge {}",
                self.file,
                self.context(),
                path_label(path)
            ));
        }
    }

    fn reject_renamed_use(&mut self, tree: &UseTree) {
        match tree {
            UseTree::Rename(rename) => {
                let original = rename.ident.to_string();
                if original == "Command"
                    || FORBIDDEN_PROCESS_OR_LOADER_APIS.contains(&original.as_str())
                {
                    self.forbidden.push(format!(
                        "{} renames process/loader import {} as {}",
                        self.file, rename.ident, rename.rename
                    ));
                }
            }
            UseTree::Path(path) => self.reject_renamed_use(&path.tree),
            UseTree::Group(group) => {
                for item in &group.items {
                    self.reject_renamed_use(item);
                }
            }
            UseTree::Name(_) | UseTree::Glob(_) => {}
        }
    }
}

impl<'ast> Visit<'ast> for RustProcessAudit {
    fn visit_item_mod(&mut self, node: &'ast ItemMod) {
        if !is_test_only(&node.attrs) {
            visit::visit_item_mod(self, node);
        }
    }

    fn visit_item_impl(&mut self, node: &'ast ItemImpl) {
        let previous = self.implementation.replace(type_label(&node.self_ty));
        visit::visit_item_impl(self, node);
        self.implementation = previous;
    }

    fn visit_item_use(&mut self, node: &'ast ItemUse) {
        self.reject_renamed_use(&node.tree);
        visit::visit_item_use(self, node);
    }

    fn visit_item_type(&mut self, node: &'ast ItemType) {
        if type_is_command(&node.ty) {
            self.forbidden.push(format!(
                "{} aliases std::process::Command as {}",
                self.file, node.ident
            ));
        }
        visit::visit_item_type(self, node);
    }

    fn visit_local(&mut self, node: &'ast Local) {
        if let Some(initializer) = &node.init {
            self.reject_process_alias_expression("local", &initializer.expr);
        }
        visit::visit_local(self, node);
    }

    fn visit_item_const(&mut self, node: &'ast ItemConst) {
        self.reject_process_alias_expression("const", &node.expr);
        visit::visit_item_const(self, node);
    }

    fn visit_item_static(&mut self, node: &'ast ItemStatic) {
        self.reject_process_alias_expression("static", &node.expr);
        visit::visit_item_static(self, node);
    }

    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        if is_test_only(&node.attrs) {
            return;
        }
        self.enter_function(node.sig.ident.to_string(), |audit| {
            visit::visit_item_fn(audit, node)
        });
    }

    fn visit_impl_item_fn(&mut self, node: &'ast ImplItemFn) {
        if is_test_only(&node.attrs) {
            return;
        }
        let context = format!(
            "{}::{}",
            self.implementation.as_deref().unwrap_or("<impl>"),
            node.sig.ident
        );
        self.enter_function(context, |audit| visit::visit_impl_item_fn(audit, node));
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(function) = node.func.as_ref() {
            let segments: Vec<_> = function.path.segments.iter().collect();
            if path_is_command_constructor(&function.path) {
                let argument = node
                    .args
                    .first()
                    .map(expression_label)
                    .unwrap_or_else(|| "<missing>".to_owned());
                *self
                    .launches
                    .entry(Launch {
                        file: self.file.clone(),
                        context: self.context(),
                        argument,
                    })
                    .or_default() += 1;
            }
            if let Some(last) = segments.last() {
                let name = last.ident.to_string();
                self.record_fact(name.clone());
                if FORBIDDEN_PROCESS_OR_LOADER_APIS.contains(&name.as_str()) {
                    self.forbidden.push(format!(
                        "{} in {} calls forbidden process API {name}",
                        self.file,
                        self.context()
                    ));
                }
            }
        }
        visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        let method = format!(".{}", node.method);
        self.record_fact(method.clone());
        if self.process_scope
            && matches!(
                node.method.to_string().as_str(),
                "spawn" | "output" | "status" | "exec"
            )
        {
            *self
                .terminal_methods
                .entry(ProcessMethod {
                    file: self.file.clone(),
                    context: self.context(),
                    method,
                })
                .or_default() += 1;
        }
        visit::visit_expr_method_call(self, node);
    }

    fn visit_foreign_item_fn(&mut self, node: &'ast ForeignItemFn) {
        let name = node.sig.ident.to_string();
        if FORBIDDEN_PROCESS_OR_LOADER_APIS.contains(&name.as_str()) {
            self.forbidden.push(format!(
                "{} declares forbidden process/loader API {name}",
                self.file
            ));
        }
        if let Some(link_name) = forbidden_link_name(&node.attrs) {
            self.forbidden.push(format!(
                "{} hides forbidden process/loader API {link_name} behind foreign declaration {name}",
                self.file
            ));
        }
        visit::visit_foreign_item_fn(self, node);
    }

    fn visit_macro(&mut self, node: &'ast Macro) {
        let name = node
            .path
            .segments
            .last()
            .map(|segment| segment.ident.to_string())
            .unwrap_or_default();
        if ["cmd", "cmd_lib", "run_script", "shell"].contains(&name.as_str()) {
            self.forbidden.push(format!(
                "{} in {} invokes unclassified process macro {name}!",
                self.file,
                self.context()
            ));
        }
        visit::visit_macro(self, node);
    }
}

fn audit_rust_processes(root: &Path) -> RustProcessAudit {
    let mut files = Vec::new();
    for relative in [
        "native/deltafin/src",
        "native/deltafin-bootstrap/src",
        "native/deltafin-native-build/src",
        "native/deltafin-sys/src",
        "native/deltafin-xtask/src",
    ] {
        walk_rust(&root.join(relative), &mut files);
    }
    files.extend([
        root.join("native/deltafin/build.rs"),
        root.join("native/deltafin-curl-sys-direct/build.rs"),
        root.join("native/deltafin-curl-sys-direct/build_support.rs"),
        root.join("native/deltafin-curl-sys-direct/lib.rs"),
    ]);
    files.sort();
    files.dedup();

    let mut audit = RustProcessAudit::default();
    for path in files {
        let source = read(&path);
        let syntax = syn::parse_file(&source).unwrap_or_else(|error| {
            panic!("parse supported Rust source {}: {error}", path.display())
        });
        audit.file = slash_relative(
            path.strip_prefix(root)
                .expect("audited Rust source remains inside repository"),
        );
        audit.process_scope = source.contains("std::process::{")
            || source.contains("std::process::Command")
            || source.contains("process::Command");
        audit.context = None;
        audit.implementation = None;
        audit.visit_file(&syntax);
    }
    audit
}

fn audit_rust_policy_fixture(source: &str) -> RustProcessAudit {
    let syntax = syn::parse_file(source).expect("parse Rust process-policy fixture");
    let mut audit = RustProcessAudit {
        file: "<policy-fixture>".to_owned(),
        process_scope: true,
        ..RustProcessAudit::default()
    };
    audit.visit_file(&syntax);
    audit
}

#[test]
fn rust_process_audit_rejects_loader_and_process_aliases() {
    let bypasses = [
        (
            "renamed Command import",
            "use std::process::Command as HiddenCommand;",
        ),
        (
            "Command type alias",
            "type HiddenCommand = std::process::Command;",
        ),
        (
            "local Command constructor alias",
            "fn edge() { let hidden = std::process::Command::new; let _ = hidden; }",
        ),
        ("renamed process API", "use libc::system as hidden;"),
        ("renamed loader API", "use libc::dlopen as hidden_loader;"),
        (
            "local loader alias",
            "fn edge() { let hidden = libc::dlopen; let _ = hidden; }",
        ),
        (
            "Windows loader call",
            "fn edge() { unsafe { LoadLibraryW(core::ptr::null()); } }",
        ),
        (
            "foreign link-name alias",
            "unsafe extern \"C\" { #[link_name = \"dlopen\"] fn hidden_loader(); }",
        ),
    ];
    for (label, source) in bypasses {
        let audit = audit_rust_policy_fixture(source);
        assert!(
            !audit.forbidden.is_empty(),
            "Rust process audit accepted {label}: {source}"
        );
    }

    let ordinary = audit_rust_policy_fixture("use std::fmt::Result as FormatResult;");
    assert!(
        ordinary.forbidden.is_empty(),
        "ordinary non-process aliases must remain valid: {:?}",
        ordinary.forbidden
    );
}

fn expected_launches() -> BTreeMap<Launch, usize> {
    let mut expected = BTreeMap::new();
    let mut add = |file: &str, context: &str, argument: &str, count: usize| {
        expected.insert(
            Launch {
                file: file.to_owned(),
                context: context.to_owned(),
                argument: argument.to_owned(),
            },
            count,
        );
    };
    add(
        "native/deltafin/src/benchmark.rs",
        "execute_runner_with_limits",
        "runner",
        1,
    );
    add(
        "native/deltafin/src/upgrade.rs",
        "ProcessRunner::run",
        "&executable",
        1,
    );

    let build = "native/deltafin-native-build/src/lib.rs";
    // The CPU-only link goes through the C driver on Unix and link.exe on
    // Windows; the provider link likewise through the C++ driver or link.exe.
    add(build, "build_and_run_cpu_only_test", "linker_program", 1);
    add(build, "link_provider_test", "&provider.toolchain.cxx", 1);
    add(build, "link_provider_test_msvc", "linker", 1);
    add(build, "run_native_test_cases", "executable", 1);
    add(build, "audit_macho_dependencies", "&tool", 1);
    add(build, "macho_install_name", "tool", 1);
    add(build, "audit_elf_dependencies", "&readelf", 1);
    add(build, "linux_loader_cache_directories", "tool", 1);
    add(build, "PythonGuard::build", "cc", 1);
    add(build, "PythonGuard::build", "&executable", 1);
    add(
        build,
        "build_embedded_metal_libraries",
        "&toolchain.metal",
        1,
    );
    add(
        build,
        "build_embedded_metal_libraries",
        "&toolchain.metallib",
        1,
    );
    add(build, "discover_metal_toolchain", "&xcode_select", 1);
    add(build, "discover_metal_toolchain", "&xcodebuild", 1);
    add(build, "metal_toolchain_under", "&metal", 1);
    add(build, "metal_toolchain_under", "&metallib", 1);
    add(build, "compile_cpp", "&toolchain.cxx", 1);
    add(build, "compile_gemv", "&toolchain.cc", 1);
    add(build, "compile_c_test_main", "&toolchain.cc", 1);
    add(build, "archive_objects", "&toolchain.ar", 1);
    // The device-kernel compile loop is shared by NVCC and HIPCC; each planner
    // separately launches its own compiler once to read a version banner.
    add(build, "build_cuda_kernel", "&plan.compiler", 1);
    add(build, "plan_cuda_kernel_build", "&compiler", 1);
    add(build, "plan_hip_kernel_build", "&compiler", 1);
    add(build, "validate_compiler", "path", 1);
    expected
}

fn expected_terminal_methods() -> BTreeMap<ProcessMethod, usize> {
    let mut expected = BTreeMap::new();
    let mut add = |file: &str, context: &str, method: &str| {
        expected.insert(
            ProcessMethod {
                file: file.to_owned(),
                context: context.to_owned(),
                method: method.to_owned(),
            },
            1,
        );
    };
    add(
        "native/deltafin/src/benchmark.rs",
        "execute_runner_with_limits",
        ".spawn",
    );
    add(
        "native/deltafin/src/upgrade.rs",
        "ProcessRunner::run",
        ".output",
    );
    let build = "native/deltafin-native-build/src/lib.rs";
    add(build, "run_bounded_child", ".spawn");
    add(build, "PythonGuard::build", ".output");
    add(build, "run_guarded_raw_output", ".output");
    add(build, "run_checked", ".output");
    expected
}

fn require_facts(audit: &RustProcessAudit, file: &str, context: &str, required: &[&str]) {
    let facts = audit
        .facts
        .get(&(file.to_owned(), context.to_owned()))
        .unwrap_or_else(|| panic!("missing AST facts for {file}::{context}"));
    for fact in required {
        assert!(
            facts.contains(*fact),
            "{file}::{context} lost required process guard/call {fact:?}; facts={facts:?}"
        );
    }
}

#[test]
fn owned_process_edges_are_ast_classified_and_guarded() {
    let audit = audit_rust_processes(&repository_root());
    assert!(audit.forbidden.is_empty(), "{}", audit.forbidden.join("\n"));
    assert_eq!(
        audit.launches,
        expected_launches(),
        "a supported Rust process launch was added, removed, or moved without classification"
    );
    assert_eq!(
        audit.terminal_methods,
        expected_terminal_methods(),
        "a terminal process operation was added, removed, or moved without classification"
    );

    require_facts(
        &audit,
        "native/deltafin/src/upgrade.rs",
        "ProcessRunner::run",
        &["resolve_native_program", ".env_clear", ".output"],
    );
    require_facts(
        &audit,
        "native/deltafin/src/upgrade.rs",
        "resolve_native_program_in_path",
        &["inspect_native_executable"],
    );
    require_facts(
        &audit,
        "native/deltafin/src/benchmark.rs",
        "run_campaign",
        &[
            "validate_options",
            "audit_benchmark_executable",
            "pin_native_executable",
            "verify_pinned_runner",
        ],
    );
    require_facts(
        &audit,
        "native/deltafin/src/benchmark.rs",
        "execute_runner_with_limits",
        &[".process_group", ".spawn"],
    );

    let build = "native/deltafin-native-build/src/lib.rs";
    require_facts(
        &audit,
        build,
        "resolve_tool_path",
        &["validate_native_executable"],
    );
    require_facts(
        &audit,
        build,
        "PythonGuard::build",
        &[
            "sanitize_native_environment",
            "run_checked",
            "validate_native_executable",
            ".output",
        ],
    );
    require_facts(
        &audit,
        build,
        "run_guarded_raw_output",
        &[".prepare", ".output", ".assert_clean"],
    );
    require_facts(&audit, build, "build_cuda_kernel", &["run_guarded_checked"]);
    require_facts(
        &audit,
        build,
        "plan_cuda_kernel_build",
        &["validate_cuda_driver", "run_guarded_output"],
    );
    require_facts(
        &audit,
        build,
        "plan_hip_kernel_build",
        &["validate_native_executable", "run_guarded_output"],
    );
    require_facts(
        &audit,
        build,
        "discover_metal_toolchain",
        &["validate_apple_native_tool", "run_guarded_raw_output"],
    );
    // The compile functions report a failure as a value (so one run can name
    // every translation unit that failed) through the same guarded runner the
    // panicking form wraps.
    for context in ["compile_cpp", "compile_gemv", "compile_c_test_main"] {
        require_facts(&audit, build, context, &["run_guarded_try"]);
    }
    require_facts(&audit, build, "archive_objects", &["run_guarded_checked"]);
    for context in [
        "link_provider_test",
        "link_provider_test_msvc",
        "build_and_run_cpu_only_test",
    ] {
        require_facts(&audit, build, context, &["run_guarded_try"]);
    }
}

fn strip_native_comments_and_literals(source: &str) -> String {
    #[derive(Clone, Copy)]
    enum State {
        Code,
        LineComment,
        BlockComment(usize),
        String(u8),
    }
    let bytes = source.as_bytes();
    let mut output = String::with_capacity(source.len());
    let mut state = State::Code;
    let mut index = 0;
    while index < bytes.len() {
        match state {
            State::Code if bytes[index..].starts_with(b"//") => {
                state = State::LineComment;
                output.push(' ');
                index += 2;
            }
            State::Code if bytes[index..].starts_with(b"/*") => {
                state = State::BlockComment(1);
                output.push(' ');
                index += 2;
            }
            State::Code if matches!(bytes[index], b'\'' | b'"') => {
                state = State::String(bytes[index]);
                output.push(' ');
                index += 1;
            }
            State::Code => {
                output.push(char::from(bytes[index]));
                index += 1;
            }
            State::LineComment if bytes[index] == b'\n' => {
                state = State::Code;
                output.push('\n');
                index += 1;
            }
            State::LineComment => index += 1,
            State::BlockComment(depth) if bytes[index..].starts_with(b"/*") => {
                state = State::BlockComment(depth + 1);
                index += 2;
            }
            State::BlockComment(depth) if bytes[index..].starts_with(b"*/") => {
                state = if depth == 1 {
                    State::Code
                } else {
                    State::BlockComment(depth - 1)
                };
                index += 2;
            }
            State::BlockComment(_) => index += 1,
            State::String(_) if bytes[index] == b'\\' => index = (index + 2).min(bytes.len()),
            State::String(delimiter) if bytes[index] == delimiter => {
                state = State::Code;
                index += 1;
            }
            State::String(_) => index += 1,
        }
    }
    output
}

fn native_identifiers(source: &str) -> Vec<String> {
    let code = strip_native_comments_and_literals(source);
    let mut identifiers = Vec::new();
    let mut current = String::new();
    for character in code.chars() {
        if character == '_' || character.is_ascii_alphanumeric() {
            current.push(character);
        } else if !current.is_empty() {
            identifiers.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        identifiers.push(current);
    }
    identifiers
}

#[test]
fn in_process_provider_sources_have_no_process_or_interpreter_api() {
    let root = repository_root();
    let inventory: BTreeSet<&str> = PRODUCTION_PROVIDER_SOURCES.iter().copied().collect();
    assert_eq!(
        inventory.len(),
        PRODUCTION_PROVIDER_SOURCES.len(),
        "the shared production-provider source inventory contains duplicates"
    );

    for entry in fs::read_dir(root.join("native/provider_gate")).expect("scan provider sources") {
        let entry = entry.expect("read provider source");
        if !entry
            .file_type()
            .expect("inspect provider source")
            .is_file()
        {
            continue;
        }
        let path = entry.path();
        let extension = path.extension().and_then(|value| value.to_str());
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let production = extension == Some("h")
            || (matches!(extension, Some("c" | "cpp" | "mm" | "cu" | "metal"))
                && !name.contains("_test.")
                && name.as_ref() != "provider_gate.cpp");
        if production {
            let relative = slash_relative(
                path.strip_prefix(&root)
                    .expect("provider source remains inside repository"),
            );
            assert!(
                inventory.contains(relative.as_str()),
                "production provider source is absent from the shared inventory: {relative}"
            );
        }
    }

    let mut sources: Vec<PathBuf> = PRODUCTION_PROVIDER_SOURCES
        .iter()
        .map(|relative| root.join(relative))
        .collect();
    sources.sort();
    let forbidden = FORBIDDEN_PROCESS_OR_LOADER_APIS.iter().copied().chain([
        "CreateProcess",
        "ShellExecute",
        "WinExec",
        "NSTask",
        "Py_Initialize",
        "PyRun_SimpleString",
        "PyImport_ImportModule",
    ]);
    for path in sources {
        assert!(
            path.is_file(),
            "production provider source is missing: {}",
            path.display()
        );
        let relative = slash_relative(
            path.strip_prefix(&root)
                .expect("provider source remains in repository"),
        );
        let identifiers = native_identifiers(&read(&path));
        for name in forbidden.clone() {
            assert!(
                !identifiers.iter().any(|identifier| identifier == name),
                "compiled provider source {relative} contains forbidden process/interpreter API {name}"
            );
        }
    }
}

fn shell_fence_commands(document: &str) -> Vec<String> {
    let mut shell = false;
    let mut commands = Vec::new();
    let mut pending = String::new();
    for line in document.lines() {
        let trimmed = line.trim();
        if let Some(info) = trimmed.strip_prefix("```") {
            if shell && !pending.trim().is_empty() {
                commands.push(std::mem::take(&mut pending));
            }
            shell = if shell {
                false
            } else {
                matches!(info.trim(), "bash" | "sh" | "shell" | "zsh" | "console")
            };
            continue;
        }
        if !shell {
            continue;
        }
        if trimmed.is_empty() || trimmed.starts_with('#') {
            if !pending.trim().is_empty() {
                commands.push(std::mem::take(&mut pending));
            }
            continue;
        }
        let continued = trimmed.ends_with('\\');
        let piece = trimmed.strip_suffix('\\').unwrap_or(trimmed).trim_end();
        if !pending.is_empty() {
            pending.push(' ');
        }
        pending.push_str(piece);
        if !continued {
            commands.push(std::mem::take(&mut pending));
        }
    }
    commands
}

fn shell_segments(command: &str) -> Vec<Vec<String>> {
    let mut segments = vec![Vec::new()];
    let mut word = String::new();
    let mut quote = None;
    let mut escaped = false;
    let mut characters = command.chars().peekable();
    while let Some(character) = characters.next() {
        if escaped {
            word.push(character);
            escaped = false;
            continue;
        }
        if let Some(delimiter) = quote {
            if character == delimiter {
                quote = None;
            } else if character == '\\' && delimiter == '"' {
                escaped = true;
            } else {
                word.push(character);
            }
            continue;
        }
        match character {
            '\'' | '"' => quote = Some(character),
            '\\' => escaped = true,
            ';' | '|' | '&' => {
                if !word.is_empty() {
                    segments
                        .last_mut()
                        .expect("shell segment")
                        .push(std::mem::take(&mut word));
                }
                if characters.peek() == Some(&character) {
                    characters.next();
                }
                if !segments.last().expect("shell segment").is_empty() {
                    segments.push(Vec::new());
                }
            }
            value if value.is_whitespace() => {
                if !word.is_empty() {
                    segments
                        .last_mut()
                        .expect("shell segment")
                        .push(std::mem::take(&mut word));
                }
            }
            _ => word.push(character),
        }
    }
    if !word.is_empty() {
        segments.last_mut().expect("shell segment").push(word);
    }
    segments
        .into_iter()
        .filter(|segment| !segment.is_empty())
        .collect()
}

fn shell_executable(words: &[String]) -> Option<&str> {
    let mut index = 0;
    while let Some(word) = words.get(index) {
        if word.starts_with('$') && matches!(word.as_str(), "$" | ">") {
            index += 1;
            continue;
        }
        let assignment = word.split_once('=').is_some_and(|(name, _)| {
            !name.is_empty()
                && name.bytes().enumerate().all(|(offset, byte)| {
                    byte == b'_'
                        || byte.is_ascii_alphabetic()
                        || (offset > 0 && byte.is_ascii_digit())
                })
        });
        if assignment {
            index += 1;
            continue;
        }
        if matches!(
            word.as_str(),
            "env" | "command" | "exec" | "time" | "nice" | "sudo"
        ) {
            index += 1;
            while words.get(index).is_some_and(|value| value.starts_with('-')) {
                index += 1;
            }
            continue;
        }
        return Some(word);
    }
    None
}

fn markdown_command_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|word| {
            let word = word.trim_matches(|character: char| {
                !character.is_ascii_alphanumeric()
                    && !matches!(character, '_' | '-' | '.' | '/' | '\\')
            });
            let word = word.trim_end_matches('.');
            (!word.is_empty()).then(|| word.to_owned())
        })
        .collect()
}

fn is_interpreter_name(word: &str) -> bool {
    let basename = word
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(word)
        .to_ascii_lowercase();
    let python_version = basename.strip_prefix("python").is_some_and(|suffix| {
        !suffix.is_empty()
            && suffix
                .chars()
                .all(|character| character.is_ascii_digit() || character == '.')
    });
    python_version
        || matches!(
            basename.as_str(),
            "python"
                | "pypy"
                | "pypy3"
                | "perl"
                | "ruby"
                | "node"
                | "bash"
                | "zsh"
                | "sh"
                | "powershell"
                | "pwsh"
        )
}

fn public_interpreter_invocations(document: &str) -> Vec<(usize, String)> {
    let mut found = Vec::new();
    for (offset, line) in document.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            continue;
        }

        // Inline code is an executable-looking surface even when prose around
        // it is not inside a shell fence.
        for (index, span) in line.split('`').enumerate() {
            if index % 2 == 1
                && markdown_command_words(span)
                    .iter()
                    .any(|word| is_interpreter_name(word))
            {
                found.push((offset + 1, line.to_owned()));
                break;
            }
        }

        let words = markdown_command_words(line);
        for (index, word) in words.iter().enumerate() {
            if !is_interpreter_name(word) {
                continue;
            }
            let previous = index
                .checked_sub(1)
                .and_then(|position| words.get(position))
                .map(|value| value.to_ascii_lowercase());
            let next = words.get(index + 1).map(|value| value.to_ascii_lowercase());
            let command_context = previous.as_deref().is_some_and(|value| {
                matches!(
                    value,
                    "run" | "execute" | "invoke" | "launch" | "start" | "using" | "with"
                )
            });
            let command_argument = next.as_deref().is_some_and(|value| {
                value.starts_with('-')
                    || value.starts_with('/')
                    || [
                        ".py", ".pyw", ".pl", ".rb", ".sh", ".bash", ".zsh", ".js", ".mjs", ".cjs",
                        ".ps1",
                    ]
                    .iter()
                    .any(|suffix| value.ends_with(suffix))
            });
            if command_context || command_argument {
                let finding = (offset + 1, line.to_owned());
                if found.last() != Some(&finding) {
                    found.push(finding);
                }
                break;
            }
        }
    }
    found
}

#[test]
fn public_interpreter_detector_covers_inline_and_prose_commands() {
    for example in [
        "Run `python tools/legacy.py` to continue.",
        "Run python3 -m legacy.module to continue.",
        "$ python tools/legacy.py",
        "Execute /usr/bin/env bash ./legacy-task.sh.",
    ] {
        assert!(
            !public_interpreter_invocations(example).is_empty(),
            "public-doc interpreter detector accepted {example:?}"
        );
    }
    for prose in [
        "Python native builders were retired.",
        "The binary rejects libpython and libtorch_python.",
        "A shell runner is not part of the product.",
    ] {
        assert!(
            public_interpreter_invocations(prose).is_empty(),
            "public-doc interpreter detector rejected ordinary prose {prose:?}"
        );
    }
}

#[test]
fn public_commands_use_only_supported_compiled_entrypoints() {
    let root = repository_root();
    let allowed = [
        "./target/release/deltafin",
        "cargo",
        "cd",
        "curl",
        "git",
        "xcodebuild",
    ];
    let forbidden_substitutions = [
        "$(python", "$(pypy", "$(perl", "$(bash", "$(sh ", "`python", "`pypy", "`perl", "`bash",
        "`sh ",
    ];
    for relative in PUBLIC_DOCUMENTS {
        let document = read(&root.join(relative));
        let interpreter_invocations = public_interpreter_invocations(&document);
        assert!(
            interpreter_invocations.is_empty(),
            "{relative} contains public interpreter invocation(s): {interpreter_invocations:?}"
        );
        for command in shell_fence_commands(&document) {
            let lower = command.to_ascii_lowercase();
            for forbidden in forbidden_substitutions {
                assert!(
                    !lower.contains(forbidden),
                    "{relative} contains interpreter command substitution: {command:?}"
                );
            }
            for segment in shell_segments(&command) {
                let executable = shell_executable(&segment).unwrap_or_else(|| {
                    panic!("{relative} has an empty shell command: {command:?}")
                });
                assert!(
                    allowed.contains(&executable),
                    "{relative} publishes unclassified command {executable:?}: {command:?}"
                );
                assert!(
                    !(executable == "cargo"
                        && segment
                            .iter()
                            .any(|word| matches!(word.as_str(), "package" | "publish"))),
                    "{relative} presents an internal Cargo package as a supported distribution: {command:?}"
                );
            }
        }
    }
}

fn toml_without_comments(source: &str) -> String {
    source
        .lines()
        .map(|line| line.split('#').next().unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n")
}

fn toml_table<'a>(source: &'a str, name: &str) -> Option<&'a str> {
    let header = format!("[{name}]");
    let start = source.find(&header)? + header.len();
    let tail = &source[start..];
    let end = tail
        .match_indices('\n')
        .find_map(|(offset, _)| {
            tail[offset + 1..]
                .trim_start()
                .starts_with('[')
                .then_some(offset)
        })
        .unwrap_or(tail.len());
    Some(&tail[..end])
}

fn toml_scalar<'a>(table: &'a str, key: &str) -> Option<&'a str> {
    table.lines().find_map(|line| {
        let (name, value) = line.split_once('=')?;
        (name.trim() == key).then(|| value.trim())
    })
}

#[test]
fn internal_workspace_crates_cannot_be_published_as_partial_packages() {
    let root = repository_root();
    for relative in INTERNAL_PACKAGES {
        let manifest = toml_without_comments(&read(&root.join(relative).join("Cargo.toml")));
        let package = toml_table(&manifest, "package")
            .unwrap_or_else(|| panic!("{relative}/Cargo.toml has no package table"));
        assert_eq!(
            toml_scalar(package, "publish"),
            Some("false"),
            "{relative} must remain publish-disabled because the product source graph spans the repository"
        );
        for line in manifest.lines() {
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            if !matches!(key.trim(), "build" | "path") {
                continue;
            }
            let value = value.trim().trim_matches('"');
            assert!(
                ![".py", ".sh", ".bash", ".zsh", ".pl", ".rb", ".ps1"]
                    .iter()
                    .any(|suffix| value.ends_with(suffix)),
                "{relative}/Cargo.toml selects interpreted target {value:?}"
            );
        }
    }

    let lock = read(&root.join("Cargo.lock"));
    for forbidden in [
        "cmake",
        "duct",
        "subprocess",
        "xshell",
        "shell-words",
        "run_script",
        "cmd_lib",
        "autotools",
        "vcpkg",
    ] {
        assert!(
            !lock
                .lines()
                .any(|line| { line.trim() == format!("name = \"{forbidden}\"") }),
            "process/build-script helper crate {forbidden:?} entered the locked product graph"
        );
    }
}
