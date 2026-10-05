//! Wall-clock time sources.
//!
//! The reference reads `Date.now()` in two places: `MidiChannelState.noteOn`
//! stamps note-on times with it, and `evaluateAutomation` samples it once per
//! evaluation for the trigger and velocity falloff of `midi()` automation. The
//! port takes an injectable [`Clock`] in both places so tests and parity gates
//! are deterministic.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// A source of `Date.now()` values: milliseconds since the Unix epoch.
pub trait Clock: Send + Sync + fmt::Debug {
    /// The current time in milliseconds, as `Date.now()` returns it.
    fn now_ms(&self) -> f64;
}

/// `Date.now()`: whole milliseconds since the Unix epoch from the system clock.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> f64 {
        system_now_ms()
    }
}

/// `Date.now()` from the system clock: whole milliseconds since the Unix epoch
/// (negative before it), as an `f64` like the JavaScript number.
pub fn system_now_ms() -> f64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(elapsed) => elapsed.as_millis() as f64,
        Err(before) => -(before.duration().as_millis() as f64),
    }
}

/// A clock that only moves when told to; for tests and parity gates.
#[derive(Debug, Default)]
pub struct ManualClock {
    bits: AtomicU64,
}

impl ManualClock {
    /// A clock that reads `time_ms` until changed.
    pub fn new(time_ms: f64) -> Self {
        ManualClock {
            bits: AtomicU64::new(time_ms.to_bits()),
        }
    }

    /// Sets the time returned by [`Clock::now_ms`].
    pub fn set(&self, time_ms: f64) {
        self.bits.store(time_ms.to_bits(), Ordering::SeqCst);
    }

    /// Moves the clock forward by `delta_ms`.
    pub fn advance(&self, delta_ms: f64) {
        let now = self.now_ms();
        self.set(now + delta_ms);
    }
}

impl Clock for ManualClock {
    fn now_ms(&self) -> f64 {
        f64::from_bits(self.bits.load(Ordering::SeqCst))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_clock_reads_what_was_set() {
        let clock = ManualClock::new(1_000.0);
        assert_eq!(clock.now_ms(), 1_000.0);
        clock.advance(16.0);
        assert_eq!(clock.now_ms(), 1_016.0);
        clock.set(5.5);
        assert_eq!(clock.now_ms(), 5.5);
    }

    #[test]
    fn system_clock_is_whole_milliseconds() {
        let now = SystemClock.now_ms();
        assert_eq!(now, now.trunc());
        assert!(now > 1.5e12);
    }
}
