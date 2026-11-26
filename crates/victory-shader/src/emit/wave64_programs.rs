//! Console wave64 compute programs for a host of 32-lane waves, and their lowering: the
//! programs `wave64_tests` validates and runs on the card, and tests of the lowering alone,
//! which need neither and run on every platform (`cargo test -p victory-shader` on Linux).

use std::collections::HashSet;

use super::{HostShader, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// A wave64 compute program of 64 threads (one console wave; the thread index in v0) with 4
/// root constants in s0..s3 and a raw UAV's descriptor in s[8:11], then `body`.
#[cfg_attr(not(windows), allow(dead_code))] // (for `wave64_tests`)
pub(super) fn native(body: &[u32]) -> NativeProgram {
    native_of(body, true, 64)
}

/// The same for a console of 32-lane waves when `wave64` is not set, with `threads` threads.
pub(super) fn native_of(body: &[u32], wave64: bool, threads: u32) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `native` for a host of 32-lane waves, with the branches closing `wide` taken wave-wide.
pub(super) fn lowered(native: &NativeProgram, p: &ir::Program, wide: &HashSet<usize>) -> Result<HostShader, ShaderError> {
    lowered_on(native, p, wide, 32)
}

/// [`lowered`] for a host whose waves have at most `host_wave` lanes.
pub(super) fn lowered_on(
    native: &NativeProgram,
    p: &ir::Program,
    wide: &HashSet<usize>,
    host_wave: u32,
) -> Result<HostShader, ShaderError> {
    lowered_counted(native, p, wide, host_wave, false)
}

/// [`lowered_on`], with the dispatch's thread count passed in (`counted`).
pub(super) fn lowered_counted(
    native: &NativeProgram,
    p: &ir::Program,
    wide: &HashSet<usize>,
    host_wave: u32,
    counted: bool,
) -> Result<HostShader, ShaderError> {
    lowered_with_wave(native, p, wide, host_wave, counted).map(|(shader, _)| shader)
}

/// [`lowered_counted`], and the wave size the translation requires of the host.
pub(super) fn lowered_with_wave(
    native: &NativeProgram,
    p: &ir::Program,
    wide: &HashSet<usize>,
    host_wave: u32,
    counted: bool,
) -> Result<(HostShader, Option<u32>), ShaderError> {
    let st = structure::structure_wide(p, wide).expect("structure");
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
    let lowered = lower::lower_counted(&lower::Parts {
        native,
        p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    }, counted);
    lower::TEST_HOST_WAVE.set(was);
    let lowered = lowered?;
    let wave = lowered.wave_size;
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    };
    Ok((shader, wave))
}

/// The prefix of wave totals the console's structure builder computes: each console
/// wave of a 256-thread group writes its total (10 + its index) to LDS, then, under
/// `if (id >> 6 != 0)`, loops over the totals of the waves before it with lanes leaving at their
/// own wave's index, reading the running sum's first lane (`v_readfirstlane`) every iteration.
/// (The total is read from the wave's last lane first, as the builder's scan does: a read
/// across the two host waves, which makes them exchange.)
/// The condition and the sum are the same in every lane of a console wave, so each host wave
/// takes its own first lane; before, the read was refused ("a first-lane read where not every
/// lane runs"), which kept the builder off on 32-lane GPUs. Every thread stores the sum of the
/// waves before its own: 0, 10, 10 + 11, 10 + 11 + 12.
pub(super) fn wave_prefix_program(lane_dependent: bool) -> NativeProgram {
    let lds_address = if lane_dependent {
        0x3410_0082 // v_lshlrev_b32 v8, 2, v0
    } else {
        0x7E10_0210 // v_mov_b32 v8, s16
    };
    let words = [
        0x2C0E_0086, // v_lshrrev_b32 v7, 6, v0             (the console wave's index)
        0x3410_0E82, // v_lshlrev_b32 v8, 2, v7
        0x4A12_0E8A, // v_add_nc_u32 v9, 10, v7
        0xD760_0013,
        0x0201_7F09, // v_readlane_b32 s19, v9, 63   (the wave's last lane: the other host wave)
        0x7E12_0213, // v_mov_b32 v9, s19
        0xD834_0000,
        0x0000_0908, // ds_write_b32 v8, v9
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBF8A_0000, // s_barrier
        0x7E10_0280, // v_mov_b32 v8, 0
        0xBE80_047E, // s_mov_b64 s[0:1], exec
        0x7DAA_0E80, // v_cmpx_ne_u32 0, v7
        0xBF88_0011, // s_cbranch_execz END
        0xBE86_047E, // s_mov_b64 s[6:7], exec
        0xBE92_0380, // s_mov_b32 s18, 0
        0xBE91_0380, // s_mov_b32 s17, 0
        0xBF82_0002, // s_branch BODY
        0x87FE_0614, // LOOPTOP: s_and_b64 exec, s[20:21], s[6:7]
        0x7E22_0508, // v_readfirstlane_b32 s17, v8
        0x8F10_8212, // BODY: s_lshl_b32 s16, s18, 2
        0x8112_8112, // s_add_i32 s18, s18, 1
        lds_address, // LDS address for the next subtotal
        0xD8D8_0000,
        0x0800_0008, // ds_read_b32 v8, v8
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x4A10_1011, // v_add_nc_u32 v8, s17, v8
        0xBE94_047E, // s_mov_b64 s[20:21], exec
        0x7DA4_0E12, // v_cmpx_eq_u32 s18, v7
        0x8A86_7E06, // s_andn2_b64 s[6:7], s[6:7], exec
        0xBF85_FFF3, // s_cbranch_scc1 LOOPTOP
        0xBEFE_0400, // END: s_mov_b64 exec, s[0:1]
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xE070_1000,
        0x8002_0801, // buffer_store_dword v8, v1, s[8:11], 0 offen
    ];
    let mut native = native_of(&words, true, 256);
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 2 << 15; // 1 KiB covers the lane-varying address as well.
    }
    native
}

/// A first-lane read where not every lane runs, on a host of 32-lane waves: of a value alike in
/// the console wave, each host wave's own first lane; of any other, refused. Before #101 a
/// switch (`VICTORY_FIRSTLANE_OWN=1`) took the own first lane of any value, so a read like the
/// second program's gave the upper host wave its own subtotal.
mod first_lane {
    use super::*;

    /// The IR block of the program's (only) `v_readfirstlane_b32`.
    fn first_lane_block(p: &ir::Program) -> usize {
        let blocks: Vec<usize> = (0..p.blocks.len())
            .filter(|&b| p.blocks[b].insts.iter().any(|i| i.name() == "v_readfirstlane_b32"))
            .collect();
        assert_eq!(blocks.len(), 1, "one block with the first-lane read");
        blocks[0]
    }

    #[test]
    fn a_wave_alike_prefix_takes_each_host_waves_own_first_lane() {
        let native = wave_prefix_program(false);
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        first_lane_block(&p);
        lower::take_needs();
        let (_, wave) = lowered_with_wave(&native, &p, &HashSet::new(), 32, false).expect("lower");
        assert_ne!(wave, Some(64), "the host's waves of 32 run it");
        assert!(lower::take_needs().is_empty(), "no block where the halves had to meet");
    }

    #[test]
    fn a_lane_dependent_prefix_cannot_take_its_own_first_lane() {
        let native = wave_prefix_program(true);
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        lower::take_needs();
        let Err(error) = lowered(&native, &p, &HashSet::new()) else {
            panic!("the two host waves may read different subtotals");
        };
        assert!(error.to_string().contains("a first-lane read where not every lane runs"), "{error}");
        // the block is named, for the forms that take the branches around it wave-wide
        assert!(lower::take_needs().contains(&first_lane_block(&p)));
    }

    #[test]
    fn a_lane_dependent_prefix_lowers_for_waves_of_64() {
        // (the console's wave is the host's: its first lane is the console's)
        let native = wave_prefix_program(true);
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let (_, wave) = lowered_with_wave(&native, &p, &HashSet::new(), 64, false).expect("lower");
        assert_eq!(wave, Some(64));
    }
}
