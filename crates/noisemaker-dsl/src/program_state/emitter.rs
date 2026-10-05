//! Port of `demo/shaders/lib/emitter.js`: the lightweight event emitter
//! ProgramState extends.
//!
//! The reference keeps a `Map` of event name -> `Set` of handler functions and
//! dispatches by iterating the event's `Set`. The port keeps those semantics
//! exactly:
//!
//! * a listener is identified by its `Rc` (function identity): registering the
//!   same `Rc` twice for an event is a no-op, `off` removes that `Rc` only;
//! * dispatch iterates the event's set live, as a `Set` iterator does: a
//!   listener added during an emit runs in that emit (after the existing ones),
//!   a listener removed before its turn does not run, and one removed and added
//!   again runs again at the end;
//! * `removeAllListeners` drops the event's set from the map; an emit already
//!   iterating that set finishes over it;
//! * `once` registers a wrapper that removes itself, then calls the callback
//!   (so `off(event, callback)` does not cancel a `once` registration);
//! * a listener that throws (returns `Err`) is reported with
//!   `console.error('[Emitter] Error in <event> handler:', err)` and the
//!   remaining listeners still run.
//!
//! Listeners receive the emitting context (`&mut C`, ProgramState for its
//! events) so that they can call back into it, as the reference's closures do.

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use indexmap::IndexMap;

use super::console;
use crate::JsError;
use crate::value::Value;

/// An event handler: `(context, data) => void`, where returning `Err` is the
/// handler throwing.
pub type Listener<C> = Rc<dyn Fn(&mut C, &Value) -> Result<(), JsError>>;

/// Wrap a closure as a [`Listener`] (keep the returned `Rc` to `off` it later).
pub fn listener<C, F>(f: F) -> Listener<C>
where
    F: Fn(&mut C, &Value) -> Result<(), JsError> + 'static,
{
    Rc::new(f)
}

/// The handler `Set` of one event: insertion-ordered, with tombstones while an
/// emit iterates it so that the iteration observes additions and deletions the
/// way a JavaScript `Set` iterator does.
struct ListenerSet<C> {
    entries: Vec<Option<Listener<C>>>,
    iterating: usize,
}

impl<C> ListenerSet<C> {
    fn new() -> Self {
        ListenerSet {
            entries: Vec::new(),
            iterating: 0,
        }
    }

    fn contains(&self, l: &Listener<C>) -> bool {
        self.entries.iter().flatten().any(|e| Rc::ptr_eq(e, l))
    }

    /// `set.add(l)`.
    fn add(&mut self, l: Listener<C>) {
        if !self.contains(&l) {
            self.entries.push(Some(l));
        }
    }

    /// `set.delete(l)`.
    fn delete(&mut self, l: &Listener<C>) {
        if let Some(slot) = self
            .entries
            .iter_mut()
            .find(|e| e.as_ref().is_some_and(|e| Rc::ptr_eq(e, l)))
        {
            *slot = None;
        }
        self.compact();
    }

    fn compact(&mut self) {
        if self.iterating == 0 {
            self.entries.retain(Option::is_some);
        }
    }

    fn len(&self) -> usize {
        self.entries.iter().flatten().count()
    }
}

type ListenerMap<C> = IndexMap<String, Rc<RefCell<ListenerSet<C>>>>;

/// The event emitter (`class Emitter` of the reference).
///
/// `C` is the context listeners receive; ProgramState uses itself. A standalone
/// emitter can use `()`.
pub struct Emitter<C> {
    listeners: Rc<RefCell<ListenerMap<C>>>,
}

impl<C> Default for Emitter<C> {
    fn default() -> Self {
        Emitter {
            listeners: Rc::new(RefCell::new(IndexMap::new())),
        }
    }
}

impl<C> std::fmt::Debug for Emitter<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let map = self.listeners.borrow();
        f.debug_map()
            .entries(map.iter().map(|(k, s)| (k, s.borrow().len())))
            .finish()
    }
}

/// Decrements a set's iteration count when an emit finishes (or unwinds).
struct IterationGuard<'a, C> {
    set: &'a Rc<RefCell<ListenerSet<C>>>,
}

impl<C> Drop for IterationGuard<'_, C> {
    fn drop(&mut self) {
        let mut s = self.set.borrow_mut();
        s.iterating -= 1;
        s.compact();
    }
}

fn off_in<C>(map: &RefCell<ListenerMap<C>>, event: &str, callback: &Listener<C>) {
    let set = map.borrow().get(event).cloned();
    if let Some(set) = set {
        set.borrow_mut().delete(callback);
    }
}

impl<C> Emitter<C> {
    /// `new Emitter()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Another handle to the same listener table (used to dispatch while the
    /// owner of this emitter is borrowed as the listeners' context).
    pub(crate) fn share(&self) -> Self {
        Emitter {
            listeners: self.listeners.clone(),
        }
    }

    /// `on(event, callback)`: subscribe (a no-op when this `Rc` is already
    /// subscribed to `event`).
    pub fn on(&self, event: &str, callback: Listener<C>) {
        let set = self
            .listeners
            .borrow_mut()
            .entry(event.to_owned())
            .or_insert_with(|| Rc::new(RefCell::new(ListenerSet::new())))
            .clone();
        set.borrow_mut().add(callback);
    }

    /// `off(event, callback)`: unsubscribe this `Rc`.
    pub fn off(&self, event: &str, callback: &Listener<C>) {
        off_in(&self.listeners, event, callback);
    }

    /// `removeAllListeners(event)`: drop one event's listeners, or every
    /// event's when `event` is `None`.
    pub fn remove_all_listeners(&self, event: Option<&str>) {
        let mut map = self.listeners.borrow_mut();
        match event {
            // `if (event)`: an empty event name is falsy and clears everything.
            Some(e) if !e.is_empty() => {
                map.shift_remove(e);
            }
            _ => map.clear(),
        }
    }

    /// The number of listeners subscribed to `event`.
    pub fn listener_count(&self, event: &str) -> usize {
        self.listeners
            .borrow()
            .get(event)
            .map_or(0, |s| s.borrow().len())
    }

    /// `emit(event, data)`: call every listener of `event` with `data`, in
    /// subscription order. A listener's `Err` is reported to the console and
    /// does not stop the others.
    pub fn emit(&self, ctx: &mut C, event: &str, data: &Value) {
        let Some(set) = self.listeners.borrow().get(event).cloned() else {
            return;
        };
        set.borrow_mut().iterating += 1;
        let _guard = IterationGuard { set: &set };
        let mut i = 0;
        loop {
            let next = {
                let s = set.borrow();
                let mut found = None;
                while i < s.entries.len() {
                    let entry = s.entries[i].clone();
                    i += 1;
                    if entry.is_some() {
                        found = entry;
                        break;
                    }
                }
                found
            };
            let Some(handler) = next else {
                break;
            };
            if let Err(err) = handler(ctx, data) {
                console::error(&[
                    format!("[Emitter] Error in {event} handler:").into(),
                    err.into(),
                ]);
            }
        }
    }
}

impl<C: 'static> Emitter<C> {
    /// `once(event, callback)`: subscribe a wrapper that unsubscribes itself
    /// before calling `callback`. Returns the wrapper (which `off` accepts).
    pub fn once(&self, event: &str, callback: Listener<C>) -> Listener<C> {
        type Slot<C> = Rc<RefCell<Option<Weak<dyn Fn(&mut C, &Value) -> Result<(), JsError>>>>>;
        let map: Weak<RefCell<ListenerMap<C>>> = Rc::downgrade(&self.listeners);
        let me: Slot<C> = Rc::new(RefCell::new(None));
        let name = event.to_owned();
        let wrapper: Listener<C> = {
            let me = me.clone();
            Rc::new(move |ctx: &mut C, data: &Value| {
                let this = me.borrow().as_ref().and_then(Weak::upgrade);
                if let (Some(this), Some(map)) = (this, map.upgrade()) {
                    off_in(&map, &name, &this);
                }
                callback(ctx, data)
            })
        };
        *me.borrow_mut() = Some(Rc::downgrade(&wrapper));
        self.on(event, wrapper.clone());
        wrapper
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Log = Rc<RefCell<Vec<String>>>;

    fn recorder(log: &Log, tag: &'static str) -> Listener<()> {
        let log = log.clone();
        listener(move |_: &mut (), data: &Value| {
            log.borrow_mut()
                .push(format!("{tag}:{}", data.to_json().unwrap_or_default()));
            Ok(())
        })
    }

    #[test]
    fn set_semantics_during_emit() {
        let log: Log = Rc::default();
        let e: Rc<Emitter<()>> = Rc::new(Emitter::new());
        let a = recorder(&log, "a");
        let b = recorder(&log, "b");
        let late = recorder(&log, "late");
        // `first` removes `b` (not yet visited) and adds `late` (visited in this emit).
        let first: Listener<()> = {
            let (e, b, late, log) = (Rc::downgrade(&e), b.clone(), late.clone(), log.clone());
            listener(move |_, _| {
                log.borrow_mut().push("first".into());
                let e = e.upgrade().unwrap();
                e.off("x", &b);
                e.on("x", late.clone());
                Ok(())
            })
        };
        e.on("x", first.clone());
        e.on("x", a.clone());
        e.on("x", a.clone()); // duplicate: no-op
        e.on("x", b.clone());
        e.emit(&mut (), "x", &Value::from(1.0));
        assert_eq!(*log.borrow(), ["first", "a:1", "late:1"]);
        assert_eq!(e.listener_count("x"), 3);
    }

    #[test]
    fn once_and_errors() {
        let log: Log = Rc::default();
        let e: Emitter<()> = Emitter::new();
        let cb = recorder(&log, "once");
        e.once("x", cb.clone());
        // off(callback) does not cancel the once wrapper.
        e.off("x", &cb);
        e.on("x", listener(|_: &mut (), _| Err(JsError::error("boom"))));
        e.on("x", recorder(&log, "after"));
        e.emit(&mut (), "x", &Value::Null);
        e.emit(&mut (), "x", &Value::Null);
        assert_eq!(*log.borrow(), ["once:null", "after:null", "after:null"]);
    }

    #[test]
    fn remove_all_during_emit_finishes_old_set() {
        let log: Log = Rc::default();
        let e: Rc<Emitter<()>> = Rc::new(Emitter::new());
        let clear: Listener<()> = {
            let (e, log) = (Rc::downgrade(&e), log.clone());
            listener(move |_, _| {
                let e = e.upgrade().unwrap();
                e.remove_all_listeners(Some("x"));
                e.on("x", recorder(&log, "new"));
                Ok(())
            })
        };
        e.on("x", clear);
        e.on("x", recorder(&log, "old"));
        e.emit(&mut (), "x", &Value::Null);
        assert_eq!(*log.borrow(), ["old:null"]);
        e.emit(&mut (), "x", &Value::Null);
        assert_eq!(*log.borrow(), ["old:null", "new:null"]);
    }
}
