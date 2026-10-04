//! Host and accelerator selection without chip-generation assumptions.
//!
//! The eventual linked target engine supplies the runtime provider inventory;
//! this module never assumes that being built on macOS implies a working MPS
//! device, or that being built on Linux implies a working CUDA installation.

use std::fmt::{self, Display, Formatter};

use crate::error::{DeltafinError, Result};

/// Darwin's stable CPU-family identifier for the complete first-generation
/// Apple M1 family (base, Pro, Max and Ultra).
///
/// Device auto-selection uses the family rather than a product/model string:
/// the same measured provider behavior follows the CPU/GPU generation, while
/// newer Apple families remain independently eligible for MPS.
pub(crate) const APPLE_M1_CPU_FAMILY: u64 = 0x1B58_8BB3;

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn decode_native_cpu_family(bytes: &[u8]) -> Option<u64> {
    match bytes {
        [a, b, c, d] => Some(u32::from_ne_bytes([*a, *b, *c, *d]).into()),
        [a, b, c, d, e, f, g, h] => Some(u64::from_ne_bytes([*a, *b, *c, *d, *e, *f, *g, *h])),
        _ => None,
    }
}

/// Read Darwin's native CPU-family identity without launching `sysctl` or any
/// other helper process. Unknown or differently-sized results fail open to the
/// ordinary provider selection policy.
#[cfg(target_os = "macos")]
pub(crate) fn apple_cpu_family() -> Option<u64> {
    use std::ffi::{c_char, c_int, c_void};

    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            old_value: *mut c_void,
            old_len: *mut usize,
            new_value: *mut c_void,
            new_len: usize,
        ) -> c_int;
    }

    let mut bytes = [0_u8; 8];
    let mut length = bytes.len();
    // SAFETY: `hw.cpufamily` is a static C string, `bytes` is writable for the
    // advertised in/out length, and this is a read-only sysctl query.
    let result = unsafe {
        sysctlbyname(
            c"hw.cpufamily".as_ptr(),
            bytes.as_mut_ptr().cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    if result != 0 {
        return None;
    }
    decode_native_cpu_family(bytes.get(..length)?)
}

#[cfg(not(target_os = "macos"))]
pub(crate) const fn apple_cpu_family() -> Option<u64> {
    None
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum HostOs {
    MacOs,
    Linux,
    Windows,
    Unsupported,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum HostArch {
    Aarch64,
    X86_64,
    Unsupported,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Host {
    pub os: HostOs,
    pub arch: HostArch,
}

impl Host {
    pub fn compiled() -> Self {
        Self {
            os: if cfg!(target_os = "macos") {
                HostOs::MacOs
            } else if cfg!(target_os = "linux") {
                HostOs::Linux
            } else if cfg!(windows) {
                HostOs::Windows
            } else {
                HostOs::Unsupported
            },
            arch: if cfg!(target_arch = "aarch64") {
                HostArch::Aarch64
            } else if cfg!(target_arch = "x86_64") {
                HostArch::X86_64
            } else {
                HostArch::Unsupported
            },
        }
    }

    pub fn validate(self) -> Result<Self> {
        let supported = matches!(
            (self.os, self.arch),
            (HostOs::MacOs, HostArch::Aarch64)
                | (HostOs::Linux, HostArch::Aarch64 | HostArch::X86_64)
                | (HostOs::Windows, HostArch::X86_64)
        );
        if supported {
            Ok(self)
        } else {
            Err(DeltafinError::new(format!(
                "unsupported native Deltafin host: {self}"
            )))
        }
    }
}

impl Display for Host {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        let os = match self.os {
            HostOs::MacOs => "macos",
            HostOs::Linux => "linux",
            HostOs::Windows => "windows",
            HostOs::Unsupported => std::env::consts::OS,
        };
        let arch = match self.arch {
            HostArch::Aarch64 => "aarch64",
            HostArch::X86_64 => "x86_64",
            HostArch::Unsupported => std::env::consts::ARCH,
        };
        write!(formatter, "{os}/{arch}")
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DeviceRequest {
    Auto,
    Cpu,
    Mps,
    Cuda(u16),
}

impl DeviceRequest {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        let Some(value) = value.filter(|value| !value.trim().is_empty()) else {
            return Ok(Self::Auto);
        };
        let value = value.trim().to_ascii_lowercase();
        match value.as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "mps" => Ok(Self::Mps),
            "cuda" => Ok(Self::Cuda(0)),
            _ => {
                let Some(index) = value.strip_prefix("cuda:") else {
                    return Err(DeltafinError::new(
                        "device must be auto, cpu, mps, cuda, or cuda:N",
                    ));
                };
                let index = index.parse::<u16>().map_err(|_| {
                    DeltafinError::new("CUDA device index must be a non-negative integer")
                })?;
                Ok(Self::Cuda(index))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Device {
    Cpu,
    Mps,
    Cuda(u16),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DeviceSelectionPolicy {
    ProviderDefault,
    MeasuredM1OriginalBf16Cpu,
}

impl DeviceSelectionPolicy {
    pub const fn name(self) -> &'static str {
        match self {
            Self::ProviderDefault => "provider-default",
            Self::MeasuredM1OriginalBf16Cpu => "measured-m1-original-bf16-cpu",
        }
    }

    pub const fn startup_note(self) -> Option<&'static str> {
        match self {
            Self::ProviderDefault => None,
            Self::MeasuredM1OriginalBf16Cpu => Some(
                "automatic device policy selected CPU for original-BF16 on Apple M1 after a same-binary physical one-token check measured 22.83 s on CPU versus 69.42 s on MPS; explicit --device mps remains available",
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct DeviceSelection {
    pub device: Device,
    pub policy: DeviceSelectionPolicy,
}

impl Display for Device {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => formatter.write_str("cpu"),
            Self::Mps => formatter.write_str("mps"),
            Self::Cuda(0) => formatter.write_str("cuda"),
            Self::Cuda(index) => write!(formatter, "cuda:{index}"),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct ProviderInventory {
    pub mps: bool,
    pub cuda_devices: u16,
}

impl ProviderInventory {
    pub fn select(self, request: DeviceRequest) -> Result<Device> {
        match request {
            DeviceRequest::Auto if self.mps => Ok(Device::Mps),
            DeviceRequest::Auto if self.cuda_devices > 0 => Ok(Device::Cuda(0)),
            DeviceRequest::Auto | DeviceRequest::Cpu => Ok(Device::Cpu),
            DeviceRequest::Mps if self.mps => Ok(Device::Mps),
            DeviceRequest::Mps => Err(DeltafinError::new(
                "MPS was requested, but the linked target engine reports it unavailable",
            )),
            DeviceRequest::Cuda(index) if index < self.cuda_devices => Ok(Device::Cuda(index)),
            DeviceRequest::Cuda(index) => Err(DeltafinError::new(format!(
                "CUDA device {index} was requested, but only {} device(s) are available",
                self.cuda_devices
            ))),
        }
    }

    /// Resolve a target provider while retaining one narrowly measured M1
    /// exception to the ordinary accelerator-first policy.
    ///
    /// `original_bf16` is supplied by the authenticated model program. The
    /// exception cannot affect explicit requests, quantized resident weights,
    /// newer/unknown Apple families, Linux, or CUDA-capable inventories.
    pub fn select_target(
        self,
        request: DeviceRequest,
        original_bf16: bool,
        apple_family: Option<u64>,
    ) -> Result<DeviceSelection> {
        if request == DeviceRequest::Auto
            && original_bf16
            && apple_family == Some(APPLE_M1_CPU_FAMILY)
            && self.mps
            && self.cuda_devices == 0
        {
            return Ok(DeviceSelection {
                device: self.select(DeviceRequest::Cpu)?,
                policy: DeviceSelectionPolicy::MeasuredM1OriginalBf16Cpu,
            });
        }
        Ok(DeviceSelection {
            device: self.select(request)?,
            policy: DeviceSelectionPolicy::ProviderDefault,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apple_cpu_family_identity_and_native_integer_decode_are_exact() {
        assert_eq!(APPLE_M1_CPU_FAMILY, 458_787_763);
        assert_eq!(
            decode_native_cpu_family(&(APPLE_M1_CPU_FAMILY as u32).to_ne_bytes()),
            Some(APPLE_M1_CPU_FAMILY),
        );
        assert_eq!(
            decode_native_cpu_family(&APPLE_M1_CPU_FAMILY.to_ne_bytes()),
            Some(APPLE_M1_CPU_FAMILY),
        );
        assert_eq!(decode_native_cpu_family(&[0; 3]), None);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_cpu_family_probe_reads_a_nonzero_native_identity() {
        let family = apple_cpu_family().expect("Darwin should expose hw.cpufamily");
        assert_ne!(family, 0);
    }

    #[test]
    fn platform_matrix_is_generation_agnostic_and_fail_closed() {
        for host in [
            Host {
                os: HostOs::MacOs,
                arch: HostArch::Aarch64,
            },
            Host {
                os: HostOs::Linux,
                arch: HostArch::Aarch64,
            },
            Host {
                os: HostOs::Linux,
                arch: HostArch::X86_64,
            },
            Host {
                os: HostOs::Windows,
                arch: HostArch::X86_64,
            },
        ] {
            assert!(host.validate().is_ok());
        }
        // Intel macOS and Windows on ARM have no provider build.
        for host in [
            Host {
                os: HostOs::MacOs,
                arch: HostArch::X86_64,
            },
            Host {
                os: HostOs::Windows,
                arch: HostArch::Aarch64,
            },
            Host {
                os: HostOs::Unsupported,
                arch: HostArch::X86_64,
            },
        ] {
            assert!(host.validate().is_err(), "{host}");
        }
        assert_eq!(
            Host {
                os: HostOs::Windows,
                arch: HostArch::X86_64,
            }
            .to_string(),
            "windows/x86_64"
        );
    }

    #[test]
    fn device_parser_rejects_prefix_typos() {
        assert!(DeviceRequest::parse(Some("cudax")).is_err());
        assert!(DeviceRequest::parse(Some("cuda:-1")).is_err());
        assert_eq!(
            DeviceRequest::parse(Some("CUDA:3")).unwrap(),
            DeviceRequest::Cuda(3)
        );
    }

    #[test]
    fn automatic_selection_matches_existing_precedence() {
        let all = ProviderInventory {
            mps: true,
            cuda_devices: 2,
        };
        assert_eq!(all.select(DeviceRequest::Auto).unwrap(), Device::Mps);

        let cuda_only = ProviderInventory {
            mps: false,
            cuda_devices: 2,
        };
        assert_eq!(
            cuda_only.select(DeviceRequest::Auto).unwrap(),
            Device::Cuda(0)
        );
        assert_eq!(
            ProviderInventory::default()
                .select(DeviceRequest::Auto)
                .unwrap(),
            Device::Cpu
        );
    }

    #[test]
    fn measured_m1_bf16_fallback_is_narrow_and_explicit_requests_win() {
        let apple = ProviderInventory {
            mps: true,
            cuda_devices: 0,
        };
        let fallback = apple
            .select_target(DeviceRequest::Auto, true, Some(APPLE_M1_CPU_FAMILY))
            .unwrap();
        assert_eq!(
            fallback,
            DeviceSelection {
                device: Device::Cpu,
                policy: DeviceSelectionPolicy::MeasuredM1OriginalBf16Cpu,
            },
        );
        assert_eq!(fallback.policy.name(), "measured-m1-original-bf16-cpu");
        let note = fallback
            .policy
            .startup_note()
            .expect("measured fallback should explain itself at startup");
        assert!(note.contains("22.83 s on CPU versus 69.42 s on MPS"));
        assert!(note.contains("explicit --device mps"));
        for (request, original_bf16, family, expected) in [
            (
                DeviceRequest::Mps,
                true,
                Some(APPLE_M1_CPU_FAMILY),
                Device::Mps,
            ),
            (
                DeviceRequest::Auto,
                false,
                Some(APPLE_M1_CPU_FAMILY),
                Device::Mps,
            ),
            (
                DeviceRequest::Auto,
                true,
                Some(APPLE_M1_CPU_FAMILY + 1),
                Device::Mps,
            ),
            (DeviceRequest::Auto, true, None, Device::Mps),
        ] {
            assert_eq!(
                apple.select_target(request, original_bf16, family).unwrap(),
                DeviceSelection {
                    device: expected,
                    policy: DeviceSelectionPolicy::ProviderDefault,
                },
            );
        }
    }

    #[test]
    fn measured_m1_policy_cannot_change_cuda_or_cpu_only_inventories() {
        let cuda = ProviderInventory {
            mps: false,
            cuda_devices: 2,
        };
        assert_eq!(
            cuda.select_target(DeviceRequest::Auto, true, Some(APPLE_M1_CPU_FAMILY))
                .unwrap()
                .device,
            Device::Cuda(0),
        );
        let mixed = ProviderInventory {
            mps: true,
            cuda_devices: 1,
        };
        assert_eq!(
            mixed
                .select_target(DeviceRequest::Auto, true, Some(APPLE_M1_CPU_FAMILY))
                .unwrap(),
            DeviceSelection {
                device: Device::Mps,
                policy: DeviceSelectionPolicy::ProviderDefault,
            },
        );
        assert_eq!(
            ProviderInventory::default()
                .select_target(DeviceRequest::Auto, true, Some(APPLE_M1_CPU_FAMILY))
                .unwrap()
                .device,
            Device::Cpu,
        );
    }

    #[test]
    fn cross_platform_auto_target_keeps_accelerator_precedence_except_measured_m1() {
        let cases = [
            (
                ProviderInventory {
                    mps: true,
                    cuda_devices: 2,
                },
                true,
                Some(APPLE_M1_CPU_FAMILY),
                Device::Mps,
                DeviceSelectionPolicy::ProviderDefault,
            ),
            (
                ProviderInventory {
                    mps: true,
                    cuda_devices: 0,
                },
                true,
                Some(APPLE_M1_CPU_FAMILY),
                Device::Cpu,
                DeviceSelectionPolicy::MeasuredM1OriginalBf16Cpu,
            ),
            (
                ProviderInventory {
                    mps: true,
                    cuda_devices: 0,
                },
                true,
                Some(APPLE_M1_CPU_FAMILY + 1),
                Device::Mps,
                DeviceSelectionPolicy::ProviderDefault,
            ),
            (
                ProviderInventory {
                    mps: true,
                    cuda_devices: 0,
                },
                false,
                Some(APPLE_M1_CPU_FAMILY),
                Device::Mps,
                DeviceSelectionPolicy::ProviderDefault,
            ),
            (
                ProviderInventory {
                    mps: false,
                    cuda_devices: 2,
                },
                true,
                None,
                Device::Cuda(0),
                DeviceSelectionPolicy::ProviderDefault,
            ),
            (
                ProviderInventory::default(),
                true,
                None,
                Device::Cpu,
                DeviceSelectionPolicy::ProviderDefault,
            ),
        ];
        for (inventory, original_bf16, family, device, policy) in cases {
            assert_eq!(
                inventory
                    .select_target(DeviceRequest::Auto, original_bf16, family)
                    .unwrap(),
                DeviceSelection { device, policy },
            );
        }
    }

    #[test]
    fn explicit_unavailable_provider_fails_closed() {
        let none = ProviderInventory::default();
        assert!(none.select(DeviceRequest::Mps).is_err());
        assert!(none.select(DeviceRequest::Cuda(0)).is_err());
    }
}
