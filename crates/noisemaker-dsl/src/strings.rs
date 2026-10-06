//! Effect-facing strings and manifest queries of `CanvasRenderer`
//! (`renderer/canvas.js`): `setLocale`, `getLocale`, `localize`,
//! `getEffectDescription` and `getEffectsFromManifest`.
//!
//! The string catalogs are the reference's `shaders/effects/strings.<locale>.json`
//! (embedded as `share/strings/strings.<locale>.json`): ids such as
//! `filter/adjust` (display name), `filter/adjust#desc` (description),
//! `filter/adjust.rotation` (parameter label) and `@ns/filter` (namespace
//! label). Localization is opt-in: until a locale is set every lookup returns
//! its English fallback. A locale without a catalog behaves as the
//! reference's failed fetch: an empty catalog, so lookups fall back to the
//! English catalog.

use indexmap::IndexMap;

use crate::registry::Registry;
use crate::unparser::jsv::{get_opt, object_member};
use crate::value::{Object, Value};

/// The locales with an embedded string catalog.
pub fn locales() -> Vec<&'static str> {
    noisemaker_effects::SHARE_FILES
        .iter()
        .filter_map(|(path, _)| {
            path.strip_prefix("share/strings/strings.")?
                .strip_suffix(".json")
        })
        .collect()
}

/// The string catalog of `locale` (`{}` when there is none).
pub fn strings_catalog(locale: &str) -> Object {
    noisemaker_effects::share_file(&format!("share/strings/strings.{locale}.json"))
        .and_then(|bytes| std::str::from_utf8(bytes).ok())
        .and_then(|text| Value::from_json(text).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default()
}

/// The localization state of a renderer (`_locale`, `_strings`).
#[derive(Debug, Clone, Default)]
pub struct EffectStrings {
    locale: Option<String>,
    catalogs: IndexMap<String, Object>,
}

impl EffectStrings {
    /// `setLocale(locale)`: the active locale (`None`, or an empty string,
    /// returns to English), loading its catalog and the English base.
    /// Returns the active locale.
    pub fn set_locale(&mut self, locale: Option<&str>) -> Option<&str> {
        self.locale = locale.filter(|l| !l.is_empty()).map(str::to_owned);
        let locale = self.locale.clone()?;
        for l in ["en", locale.as_str()] {
            if !self.catalogs.contains_key(l) {
                self.catalogs.insert(l.to_owned(), strings_catalog(l));
            }
        }
        self.locale.as_deref()
    }

    /// `getLocale()`.
    pub fn locale(&self) -> Option<&str> {
        self.locale.as_deref()
    }

    /// `localize(id, fallback)`: the active locale's string for `id`, else the
    /// English catalog's, else `fallback`. Empty strings count as missing.
    /// Catalog lookups are property reads (the names of `Object.prototype`
    /// members find those members, as in the reference).
    pub fn localize(&self, id: &str, fallback: Value) -> Value {
        if let Some(locale) = &self.locale {
            let lookup = |l: &str| {
                self.catalogs
                    .get(l)
                    .map_or(Value::Undefined, |c| object_member(c, id))
            };
            let hit = lookup(locale);
            if hit.is_truthy() {
                return hit;
            }
            let base = lookup("en");
            if base.is_truthy() {
                return base;
            }
        }
        fallback
    }
}

impl Registry {
    /// `getEffectsFromManifest(namespace, {includeHidden})`: the effect names
    /// of `namespace` in the manifest, sorted, hidden ones only when asked.
    pub fn effects_from_manifest(&self, namespace: &str, include_hidden: bool) -> Vec<String> {
        let prefix = format!("{namespace}/");
        let mut names: Vec<String> = self
            .manifest
            .iter()
            .filter(|(key, entry)| {
                key.starts_with(&prefix) && (include_hidden || !entry.get("hidden").is_truthy())
            })
            .map(|(key, _)| key[prefix.len()..].to_owned())
            .collect();
        // Array.prototype.sort(): UTF-16 code unit order.
        names.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
        names
    }

    /// `getEffectDescription(effectId)`: the effect's description in the
    /// active locale, else the manifest's, else `null`.
    pub fn effect_description(&self, effect_id: &str, strings: &EffectStrings) -> Value {
        let entry = object_member(&self.manifest, effect_id);
        let description = match get_opt(&entry, "description") {
            Value::Undefined | Value::Null => Value::Null,
            d => d,
        };
        strings.localize(&format!("{effect_id}#desc"), description)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn localizes_with_english_fallback() {
        assert!(locales().contains(&"en"));
        let mut s = EffectStrings::default();
        assert_eq!(
            s.localize("filter/adjust", Value::from("x")),
            Value::from("x")
        );
        assert_eq!(s.set_locale(Some("de")), Some("de"));
        assert!(s.localize("@ns/points", Value::Null).as_str().is_some());
        assert_eq!(s.set_locale(Some("")), None);
        s.set_locale(Some("xx"));
        let english = strings_catalog("en");
        assert_eq!(
            s.localize("@ns/points", Value::Null),
            english.get("@ns/points").cloned().unwrap()
        );
    }

    #[test]
    fn manifest_queries() {
        let reg = Registry::with_catalog();
        let names = reg.effects_from_manifest("synth", false);
        assert!(names.contains(&"noise".to_owned()));
        assert!(names.windows(2).all(|w| w[0] <= w[1]));
        let d = reg.effect_description("synth/noise", &EffectStrings::default());
        assert!(d.as_str().is_some());
        assert!(
            reg.effect_description("nope/none", &EffectStrings::default())
                .is_null()
        );
    }
}
