//! Unit tests ported from the reference's `test_external_input.js` and the
//! state-level assertions of `test_midi.js`.

use std::sync::Arc;

use super::*;
use crate::clock::ManualClock;

fn port<'a>(id: &'a str, name: &'a str) -> Option<MidiPortRef<'a>> {
    Some(MidiPortRef::new(id, name))
}

fn manual() -> (Arc<ManualClock>, MidiState) {
    let clock = Arc::new(ManualClock::new(1_000.0));
    let state = MidiState::with_clock(clock.clone(), true);
    (clock, state)
}

#[test]
fn channel_initializes_at_rest() {
    let channel = MidiChannelState::new();
    assert_eq!((channel.key, channel.velocity, channel.gate), (0, 0, 0));
    assert_eq!(channel.time, 0.0);
    assert_eq!(channel.pitch_bend, 8192);
    assert!(channel.keys.iter().all(|&k| k == 0));
}

#[test]
fn channel_note_on_sets_all_properties() {
    let mut channel = MidiChannelState::new();
    channel.note_on_at(60, 100, 42.0);
    assert_eq!((channel.key, channel.velocity, channel.gate), (60, 100, 1));
    assert_eq!(channel.time, 42.0);
    assert_eq!(channel.keys[60], 100);
}

#[test]
fn channel_note_off_preserves_key_and_velocity() {
    let mut channel = MidiChannelState::new();
    channel.note_on_at(60, 100, 1.0);
    channel.note_off(None);
    assert_eq!((channel.key, channel.velocity, channel.gate), (60, 100, 0));
    assert!(channel.held_notes.is_empty());
}

#[test]
fn channel_note_off_clears_one_key() {
    let mut channel = MidiChannelState::new();
    channel.note_on_at(60, 100, 1.0);
    channel.note_on_at(64, 80, 2.0);
    channel.note_off(Some(60));
    assert_eq!(channel.keys[60], 0);
    assert_eq!(channel.keys[64], 80);
}

#[test]
fn channel_reset_clears_everything() {
    let mut channel = MidiChannelState::new();
    channel.note_on_at(60, 100, 1.0);
    channel.control_change(7, 99);
    channel.reset();
    assert_eq!((channel.key, channel.velocity, channel.gate), (0, 0, 0));
    assert_eq!(channel.time, 0.0);
    assert_eq!(channel.cc[7], 0);
}

#[test]
fn get_channel_falls_back_to_channel_one() {
    let mut midi = MidiState::new();
    midi.channels[0].note_on_at(60, 100, 1.0);
    midi.channels[4].note_on_at(64, 80, 1.0);
    assert_eq!(midi.get_channel(5).key, 64);
    assert_eq!(midi.get_channel(99).key, 60);
    assert_eq!(midi.get_channel(0).key, 60);
    assert_eq!(midi.get_channel(-3).key, 60);
}

#[test]
fn handle_message_note_on_and_off() {
    let (_, mut midi) = manual();
    midi.handle_message(&[0x90, 60, 100], None);
    assert_eq!(
        (
            midi.get_channel(1).key,
            midi.get_channel(1).velocity,
            midi.get_channel(1).gate
        ),
        (60, 100, 1)
    );
    midi.handle_message(&[0x94, 72, 64], None);
    assert_eq!((midi.get_channel(5).key, midi.get_channel(5).gate), (72, 1));
    midi.handle_message(&[0x80, 60, 0], None);
    assert_eq!((midi.get_channel(1).gate, midi.get_channel(1).key), (0, 60));
    assert_eq!(midi.get_channel(1).keys[60], 0);
    midi.handle_message(&[0x90, 61, 100], None);
    midi.handle_message(&[0x90, 61, 0], None);
    assert_eq!(midi.get_channel(1).gate, 0);
    assert_eq!(midi.get_channel(1).keys[61], 0);
}

#[test]
fn handle_message_counts_clock_and_reset_clears() {
    let mut midi = MidiState::new();
    assert_eq!(midi.clock_count, 0);
    midi.handle_message(&[0xf8], None);
    midi.handle_message(&[0xf8], None);
    assert_eq!(midi.clock_count, 2);
    midi.handle_message(&[0x90, 60, 100], None);
    midi.handle_message(&[0x95, 72, 80], None);
    midi.reset();
    assert_eq!(midi.clock_count, 0);
    for n in 1..=16 {
        assert_eq!((midi.get_channel(n).gate, midi.get_channel(n).key), (0, 0));
    }
}

#[test]
fn note_grid_packs_velocity_and_gate() {
    let mut midi = MidiState::new();
    midi.handle_message(&[0x90, 60, 100], None);
    midi.handle_message(&[0x95, 72, 80], None);
    midi.update_note_grid();
    assert_eq!(midi.note_grid[60 * 4], (100.0f64 / 127.0) as f32);
    assert_eq!(midi.note_grid[60 * 4 + 1], 1.0);
    let row6 = (5 * 128 + 72) * 4;
    assert_eq!(midi.note_grid[row6], (80.0f64 / 127.0) as f32);
    midi.handle_message(&[0x80, 60, 0], None);
    midi.update_note_grid();
    assert_eq!(midi.note_grid[60 * 4], 0.0);
    assert_eq!(midi.note_grid[60 * 4 + 1], 0.0);
}

#[test]
fn ports_are_isolated_behind_the_aggregate() {
    let (_, mut midi) = manual();
    midi.handle_message(&[0x90, 60, 40], port("left-id", "Launch Control XL"));
    midi.handle_message(&[0x90, 72, 100], port("right-id", "Launch Control XL"));
    assert_eq!(midi.get_channel(1).key, 72);
    let left = midi.get_port_state(&PortSelector::new(
        Some("Launch Control XL"),
        Some("left-id"),
    ));
    assert_eq!(left.unwrap().get_channel(1).key, 60);
    let right = midi.get_port_state(&PortSelector::new(
        Some("Launch Control XL"),
        Some("right-id"),
    ));
    assert_eq!(right.unwrap().get_channel(1).key, 72);
}

#[test]
fn readable_names_resolve_once_and_ambiguity_fails() {
    let mut midi = MidiState::new();
    midi.register_port("left-id", "Launch Control XL");
    assert!(
        midi.get_port_state(&PortSelector::new(Some("Launch Control XL"), None))
            .is_some()
    );
    midi.register_port("right-id", "Launch Control XL");
    assert!(
        midi.get_port_state(&PortSelector::new(Some("Launch Control XL"), None))
            .is_none()
    );
}

#[test]
fn id_is_authoritative_over_a_stale_name() {
    let mut midi = MidiState::new();
    midi.register_port("port-id", "Renamed Controller");
    assert!(
        midi.get_port_state(&PortSelector::new(
            Some("Old Controller Name"),
            Some("port-id")
        ))
        .is_some()
    );
}

#[test]
fn disconnect_makes_the_port_unavailable_and_clears_notes() {
    let (_, mut midi) = manual();
    midi.handle_message(&[0x90, 60, 127], port("port-id", "Launch Control XL"));
    assert_eq!(
        midi.port_state_by_id("port-id")
            .unwrap()
            .get_channel(1)
            .gate,
        1
    );
    midi.disconnect_port("port-id");
    assert_eq!(
        midi.port_entry("port-id")
            .unwrap()
            .state
            .get_channel(1)
            .gate,
        0
    );
    assert!(midi.port_state_by_id("port-id").is_none());
    assert_eq!(
        midi.get_ports(),
        vec![MidiPortInfo {
            id: "port-id".into(),
            name: "Launch Control XL".into(),
            connected: false
        }]
    );
}

#[test]
fn cc14_never_mixes_bytes_across_ports() {
    let (_, mut midi) = manual();
    midi.handle_message(&[0xb0, 1, 64], port("left", "Left"));
    assert_eq!(midi.get_channel(1).cc14[1], 64 << 7);
    midi.handle_message(&[0xb0, 33, 127], port("right", "Right"));
    assert_eq!(midi.get_channel(1).cc14[1], 127);
    midi.handle_message(&[0xb1, 33, 127], port("left", "Left"));
    midi.handle_message(&[0xb0, 33, 1], port("left", "Left"));
    assert_eq!(midi.get_channel(1).cc14[1], (64 << 7) | 1);
    midi.handle_message(&[0xb0, 1, 127], port("left", "Left"));
    midi.handle_message(&[0xb0, 33, 127], port("left", "Left"));
    assert_eq!(midi.get_channel(1).cc14[1], 16383);
    midi.disconnect_port("left");
    assert_eq!(midi.get_channel(1).cc14[1], 0);
    midi.handle_message(&[0xb0, 33, 127], port("left", "Left"));
    assert_eq!(
        midi.get_channel(1).cc14[1],
        127,
        "reconnect must not resurrect old MSB"
    );
    midi.reset();
    assert_eq!(midi.get_channel(1).cc14[1], 0);
}

#[test]
fn pitch_bend_and_pressures_decode_wire_values() {
    let (_, mut midi) = manual();
    let p = port("expressive", "Expressive");
    midi.handle_message(&[0xe1, 127, 127], p);
    midi.handle_message(&[0xd1, 91], p);
    midi.handle_message(&[0x91, 60, 100], p);
    midi.handle_message(&[0xa1, 60, 63], p);
    let channel = midi.get_channel(2);
    assert_eq!(
        (
            channel.pitch_bend,
            channel.pressure,
            channel.poly_pressure[60]
        ),
        (16383, 91, 63)
    );
    for data in [
        &[0xe1, 128, 127][..],
        &[0xe1, 0],
        &[0xd1],
        &[0xd1, 128],
        &[0xa1, 60, 255],
    ] {
        midi.handle_message(data, p);
    }
    let channel = midi.get_channel(2);
    assert_eq!(
        (
            channel.pitch_bend,
            channel.pressure,
            channel.poly_pressure[60]
        ),
        (16383, 91, 63)
    );
    midi.handle_message(&[0x81, 60, 0], p);
    assert_eq!(midi.get_channel(2).poly_pressure[60], 0);
}

#[test]
fn nrpn_increment_decrement_and_reset_all_controllers() {
    let mut midi = MidiState::new();
    let cc = |midi: &mut MidiState, controller: u8, value: u8| {
        midi.handle_message(&[0xb0, controller, value], None);
    };
    for (c, v) in [(99, 0), (98, 3), (6, 127), (38, 127), (96, 92)] {
        cc(&mut midi, c, v);
    }
    assert_eq!(midi.get_channel(1).nrpn.get(&3), Some(&16383));
    cc(&mut midi, 97, 0);
    assert_eq!(midi.get_channel(1).nrpn.get(&3), Some(&16382));
    cc(&mut midi, 6, 0);
    cc(&mut midi, 97, 127);
    assert_eq!(midi.get_channel(1).nrpn.get(&3), Some(&0));
    cc(&mut midi, 6, 20);
    cc(&mut midi, 38, 3);
    for (c, v) in [(74, 99), (7, 81), (10, 61), (1, 91), (64, 127)] {
        cc(&mut midi, c, v);
    }
    midi.handle_message(&[0xe0, 127, 127], None);
    midi.handle_message(&[0xd0, 80], None);
    midi.handle_message(&[0x90, 60, 100], None);
    midi.handle_message(&[0xa0, 60, 80], None);
    cc(&mut midi, 121, 0);
    let channel = midi.get_channel(1);
    assert_eq!(channel.pitch_bend, 8192);
    assert_eq!(channel.pressure, 0);
    assert_eq!(channel.poly_pressure[60], 0);
    assert_eq!(
        (channel.cc[74], channel.cc[7], channel.cc[10]),
        (99, 81, 61)
    );
    assert_eq!(
        (
            channel.cc[1],
            channel.cc[11],
            channel.cc[64],
            channel.cc[99]
        ),
        (0, 127, 0, 127)
    );
    assert_eq!(channel.nrpn.get(&3), Some(&2563));
    assert_eq!(channel.keys[60], 100);
    cc(&mut midi, 6, 90);
    assert_eq!(
        midi.get_channel(1).nrpn.get(&3),
        Some(&2563),
        "CC121 nulls selectors"
    );
}

#[test]
fn mpe_zone_configuration_and_voice_selection() {
    let (clock, mut midi) = manual();
    let p = port("zones", "Zones");
    let configure = |midi: &mut MidiState, master: u8, count: u8| {
        for (c, v) in [(101, 0), (100, 6), (6, count)] {
            midi.handle_message(&[0xb0 | (master - 1), c, v], p);
        }
    };
    let read = |midi: &MidiState, zone: u8, members: Option<u8>| {
        let scoped = midi.port_state_by_id("zones").unwrap();
        scoped.get_zone_voice(zone, members).map(|v| v.note.key)
    };
    configure(&mut midi, 1, 10);
    clock.advance(1.0);
    midi.handle_message(&[0x9a, 70, 100], p);
    assert_eq!(read(&midi, 0, None), Some(70));
    configure(&mut midi, 16, 8);
    assert_eq!(
        read(&midi, 0, None),
        None,
        "overlapping upper zone clears reassigned held notes"
    );
    midi.handle_message(&[0x97, 66, 100], p);
    assert_eq!(read(&midi, 0, None), None, "lower shrinks to channels 2..7");
    assert_eq!(read(&midi, 1, None), Some(66));
    assert_eq!(read(&midi, 0, Some(15)), Some(66));
    configure(&mut midi, 16, 0);
    midi.handle_message(&[0x9e, 71, 100], p);
    assert_eq!(read(&midi, 1, None), None);
    assert_eq!(read(&midi, 1, Some(1)), Some(71));
    configure(&mut midi, 1, 15);
    midi.handle_message(&[0x9f, 72, 100], p);
    assert_eq!(
        read(&midi, 0, None),
        Some(72),
        "single lower zone can consume channel 16"
    );
}

#[test]
fn zone_configuration_resets_only_the_originating_port() {
    let (_, mut midi) = manual();
    let a = port("config-a", "A");
    let b = port("config-b", "B");
    midi.handle_message(&[0xe1, 127, 127], a);
    midi.handle_message(&[0xd1, 100], a);
    midi.handle_message(&[0x91, 60, 100], a);
    midi.handle_message(&[0xe2, 0, 32], b);
    midi.handle_message(&[0x92, 62, 100], b);
    for (c, v) in [(101, 0), (100, 6), (6, 3)] {
        midi.handle_message(&[0xb0, c, v], a);
    }
    assert_eq!(midi.get_channel(2).pitch_bend, 8192);
    assert_eq!(midi.get_channel(2).pressure, 0);
    assert!(midi.get_channel(2).held_notes.is_empty());
    assert_eq!(midi.get_channel(3).pitch_bend, 4096);
    assert_eq!(
        midi.port_state_by_id("config-b")
            .unwrap()
            .get_channel(3)
            .held_notes
            .len(),
        1
    );
    assert_eq!(
        midi.port_state_by_id("config-a").unwrap().get_channel(2).cc[74],
        64
    );
}

#[test]
fn port_resets_preserve_unscoped_values_and_other_ports() {
    let (_, mut midi) = manual();
    let a = port("origin-a", "A");
    let b = port("origin-b", "B");
    midi.handle_message(&[0xd1, 100], None);
    midi.handle_message(&[0xb1, 1, 100], None);
    midi.handle_message(&[0xb1, 121, 0], a);
    assert_eq!(midi.get_channel(2).pressure, 100);
    assert_eq!(midi.get_channel(2).cc[1], 100);
    midi.handle_message(&[0x91, 60, 80], a);
    midi.handle_message(&[0x91, 60, 100], b);
    midi.handle_message(&[0xa1, 60, 99], b);
    midi.handle_message(&[0x81, 60, 0], a);
    assert_eq!(midi.get_channel(2).poly_pressure[60], 99);
    assert_eq!(
        midi.get_channel(2).held_notes.get(&60).unwrap().velocity,
        100
    );
}

#[test]
fn mpe_activation_keeps_the_rpn6_transaction_usable() {
    let (_, mut midi) = manual();
    let p = port("manager-reset", "Manager");
    midi.handle_message(&[0xd0, 95], p);
    midi.handle_message(&[0xb0, 74, 99], p);
    midi.handle_message(&[0x90, 60, 100], p);
    for (c, v) in [(101, 0), (100, 6), (6, 2)] {
        midi.handle_message(&[0xb0, c, v], p);
    }
    let scoped = midi.port_state_by_id("manager-reset").unwrap();
    assert_eq!(scoped.get_channel(1).pressure, 0);
    assert_eq!(scoped.get_channel(1).cc[74], 64);
    assert!(scoped.get_channel(1).held_notes.is_empty());
    assert_eq!(scoped.get_channel(1).cc[100], 6);
    midi.handle_message(&[0xb0, 6, 4], p);
    assert_eq!(
        midi.port_state_by_id("manager-reset")
            .unwrap()
            .mpe_zones
            .lower,
        Some(4)
    );
}

#[test]
fn keyless_note_off_clears_held_voices() {
    let mut midi = MidiState::without_port_registry();
    midi.channels[1].note_on_at(60, 100, 1.0);
    midi.channels[1].note_on_at(64, 100, 2.0);
    midi.channels[1].note_off(None);
    assert!(midi.get_zone_voice(0, None).is_none());
    assert_eq!(midi.get_channel(2).key, 64);
}

#[test]
fn disconnect_and_reset_clear_expression_and_origins() {
    let (_, mut midi) = manual();
    let p = port("reset-expression", "Reset");
    for data in [
        [0xb1, 99, 0],
        [0xb1, 98, 1],
        [0xb1, 6, 100],
        [0xe1, 127, 127],
        [0xd1, 90, 0],
        [0x91, 60, 100],
    ] {
        midi.handle_message(&data, p);
    }
    midi.disconnect_port("reset-expression");
    let aggregate = midi.get_channel(2);
    assert_eq!(aggregate.pitch_bend, 8192);
    assert_eq!(aggregate.pressure, 0);
    assert!(aggregate.nrpn.is_empty());
    assert!(aggregate.held_notes.is_empty());
    midi.handle_message(&[0xb1, 38, 127], p);
    assert!(
        midi.port_state_by_id("reset-expression")
            .unwrap()
            .get_channel(2)
            .nrpn
            .is_empty()
    );
    midi.handle_message(&[0x91, 60, 100], p);
    midi.reset();
    assert!(midi.get_zone_voice(0, None).is_none());
    assert_eq!(
        midi.port_state_by_id("reset-expression")
            .unwrap()
            .get_channel(2)
            .pitch_bend,
        8192
    );
}

#[test]
fn port_inventory_keeps_duplicates_ambiguous() {
    let mut midi = MidiState::new();
    midi.register_port("twin-a", "Twin");
    midi.set_port_inventory(&[
        MidiPortInfo {
            id: "twin-a".into(),
            name: "Twin".into(),
            connected: true,
        },
        MidiPortInfo {
            id: "twin-b".into(),
            name: "Twin".into(),
            connected: true,
        },
    ]);
    assert!(
        midi.get_port_state(&PortSelector::new(Some("Twin"), None))
            .is_none()
    );
    assert!(midi.port_state_by_id("twin-a").is_some());
    midi.set_port_inventory(&[
        MidiPortInfo {
            id: "twin-a".into(),
            name: "Twin".into(),
            connected: true,
        },
        MidiPortInfo {
            id: "twin-b".into(),
            name: "Twin".into(),
            connected: false,
        },
    ]);
    assert!(
        midi.get_port_state(&PortSelector::new(Some("Twin"), None))
            .is_some()
    );
}

#[test]
fn note_on_uses_the_injected_clock_and_source_times() {
    let (clock, mut midi) = manual();
    clock.set(5_000.0);
    midi.handle_message(&[0x90, 60, 100], port("a", "A"));
    assert_eq!(midi.get_channel(1).time, 5_000.0);
    let source = midi
        .port_state_by_id("a")
        .unwrap()
        .get_channel(1)
        .held_notes[&60]
        .clone();
    let aggregate = midi.get_channel(1).held_notes[&60].clone();
    assert_eq!(source.order, aggregate.order);
    assert_eq!(source.time, aggregate.time);
    assert_eq!(aggregate.origin, Some(MidiOrigin::port("a")));
}

#[test]
fn empty_port_ids_are_dropped() {
    let mut midi = MidiState::new();
    midi.handle_message(&[0x90, 60, 10], port("", "nameless"));
    assert_eq!(midi.get_channel(1).gate, 0);
    assert!(midi.get_ports().is_empty());
}
