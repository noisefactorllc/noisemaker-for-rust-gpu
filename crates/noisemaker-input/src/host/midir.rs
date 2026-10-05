//! A [`MidiBackend`] over midir.
//!
//! Inputs are listed and connected with midir's port ids. Messages are
//! delivered as Web MIDI delivers them without the SysEx permission: System
//! Exclusive is filtered out, timing clock and active sensing pass through.

use midir::{Ignore, MidiInput, MidiInputConnection};

use crate::midi_input::{MidiAccessError, MidiBackend, MidiInputInfo, MidiSink};

/// MIDI inputs through midir.
pub struct MidirBackend {
    client_name: String,
    lister: Option<MidiInput>,
}

impl std::fmt::Debug for MidirBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MidirBackend")
            .field("client_name", &self.client_name)
            .finish_non_exhaustive()
    }
}

impl MidirBackend {
    /// A backend whose MIDI clients carry `client_name`.
    pub fn new(client_name: impl Into<String>) -> Self {
        MidirBackend {
            client_name: client_name.into(),
            lister: None,
        }
    }

    fn client(&self) -> Result<MidiInput, String> {
        MidiInput::new(&self.client_name).map_err(|error| error.to_string())
    }
}

impl Default for MidirBackend {
    fn default() -> Self {
        Self::new("noisemaker-input")
    }
}

impl MidiBackend for MidirBackend {
    type Connection = MidiInputConnection<()>;

    fn request_access(&mut self) -> Result<(), MidiAccessError> {
        let lister = self.client().map_err(MidiAccessError::Failed)?;
        self.lister = Some(lister);
        Ok(())
    }

    fn inputs(&mut self) -> Vec<MidiInputInfo> {
        if self.lister.is_none() {
            self.lister = self.client().ok();
        }
        let Some(lister) = self.lister.as_ref() else {
            return Vec::new();
        };
        lister
            .ports()
            .iter()
            .map(|port| MidiInputInfo {
                id: port.id(),
                name: lister.port_name(port).unwrap_or_default(),
            })
            .collect()
    }

    fn connect(
        &mut self,
        input: &MidiInputInfo,
        sink: MidiSink,
    ) -> Result<Self::Connection, String> {
        let mut client = self.client()?;
        client.ignore(Ignore::Sysex);
        let port = client
            .find_port_by_id(&input.id)
            .ok_or_else(|| format!("MIDI input {} is not available", input.id))?;
        client
            .connect(
                &port,
                &input.name,
                move |_timestamp, data, _| sink.deliver(data),
                (),
            )
            .map_err(|error| error.to_string())
    }
}
