//! The reference demo page as a host: a port of what `demo/shaders/index.html`
//! (`rebuildPipelineFromDsl`) and `demo/shaders/lib/demo-ui.js`
//! (`UIController`) do between a DSL program and the renderer, without the
//! DOM.
//!
//! Loading a program ([`DemoHost::rebuild_pipeline_from_dsl`]) compiles it,
//! loads it into [`ProgramState`] (`fromDsl`), builds the effect controls —
//! whose initialization writes every visible parameter back through
//! `setValue` (validation coerces values; member controls store enum paths) —
//! and applies the step values to the pipeline (`applyStepParameterValues`).
//! Building the controls also starts the host inputs:
//!
//! - `synth/media` steps (`externalTexture` other than `textTex`) load the
//!   demo's default image ([`DemoHostOptions::default_media`]: by default
//!   `img/testcard.png`, embedded in the catalog, [`default_media_image`])
//!   into `<externalTexture>_step_N` with `flipY: false` and set the step's
//!   `imageSize`;
//! - `filter/text` steps (`externalTexture: 'textTex'`) draw their text
//!   canvas at the renderer width ([`noisemaker_host::text`]), upload it as
//!   `textTex_step_N` and set the step's `textSize`;
//! - steps with an `externalMesh` load their first `builtinMeshes` entry.
//!
//! The page does these asynchronously (image decode, a 50 ms timer, OBJ
//! fetches); here they are queued and run by [`DemoHost::settle`], which also
//! fires the pipeline's debounced overlay regenerations and waits for the
//! overlay traces — the state the page reaches once every input has loaded.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use indexmap::IndexMap;
use noisemaker_dsl::program_state::{EffectInfo, ProgramState, listener};
use noisemaker_dsl::registry::EffectEntry;
use noisemaker_dsl::unparser::jsv::member;
use noisemaker_dsl::{JsError, Object, Registry, Value};
use noisemaker_host::text::{HostStyle, TextColor, TextFonts, TextParams, demo_canvas_size};

use crate::error::RenderError;
use crate::host::{CanvasRenderer, CompileOptions, TextureUpdateOptions};
use crate::jsv::{to_js_string, to_number};
use crate::png_io::{Rgba8Image, decode_png_rgba8};

/// Where the reference demo keeps its default media image, relative to a
/// reference checkout (`_loadDefaultMediaImage`: `img/testcard.png` of
/// `demo/shaders/`). The catalog embeds a byte copy of it
/// ([`noisemaker_effects::test_card_png`], decoded by
/// [`default_media_image`]).
pub const DEFAULT_MEDIA_PATH: &str = "demo/shaders/img/testcard.png";

/// The reference demo's default media image, decoded: the Philips PM5544 test
/// card the catalog embeds ([`noisemaker_effects::test_card_png`], which
/// carries its attribution), 768x576. Decoded once per thread.
pub fn default_media_image() -> Rc<Rgba8Image> {
    thread_local! {
        static IMAGE: Rc<Rgba8Image> = Rc::new(
            decode_png_rgba8(
                noisemaker_effects::test_card_png(),
                noisemaker_effects::TEST_CARD_PATH,
            )
            .expect("the embedded test card decodes"),
        );
    }
    IMAGE.with(Rc::clone)
}

/// Options of [`DemoHost::new`].
#[derive(Clone)]
pub struct DemoHostOptions {
    /// The image each media step loads by default (default: the demo's test
    /// card, [`default_media_image`]); `None` is the page's failed load ("no
    /// media loaded": no texture is uploaded).
    pub default_media: Option<Rc<Rgba8Image>>,
    /// Whether `requestMIDIAccess()` succeeds when a program needs MIDI (the
    /// page then gives the renderer a MIDI state). Headless Chromium denies it.
    pub midi_access: bool,
    /// Whether `getUserMedia()` succeeds when a program needs audio.
    pub audio_access: bool,
    /// Render a frame at this normalized time on every parameter change (the
    /// page's `renderSingleFrameIfPaused` while paused); `None` renders
    /// nothing.
    pub render_on_change: Option<f64>,
    /// The fonts text canvases draw with.
    pub text_fonts: Rc<TextFonts>,
    /// The page style text canvases inherit.
    pub text_style: HostStyle,
}

impl Default for DemoHostOptions {
    fn default() -> Self {
        DemoHostOptions {
            default_media: Some(default_media_image()),
            midi_access: false,
            audio_access: false,
            render_on_change: None,
            text_fonts: Rc::new(TextFonts::default()),
            text_style: HostStyle::demo(),
        }
    }
}

/// One media step's input (`_mediaInputs`).
#[derive(Clone, Debug)]
pub struct MediaInput {
    /// `'<externalTexture>_step_N'`.
    pub texture_id: String,
    /// The image shown (`media.source`).
    pub source: Option<Rc<Rgba8Image>>,
    /// Which media section this is (a rebuilt control set makes new ones;
    /// a load started for an old one sets the old one's source).
    generation: u64,
}

/// One text step's canvas state (`_textInputs`).
#[derive(Clone, Debug)]
pub struct TextInput {
    pub texture_id: String,
    pub effect_key: String,
    pub params: TextParams,
}

/// One mesh step's input (`_meshInputs`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshInput {
    pub mesh_id: String,
    pub loaded: bool,
    pub vertex_count: usize,
    pub status: String,
}

/// A host input the page completes asynchronously.
#[derive(Clone, Debug, PartialEq)]
enum HostTask {
    /// `img.onload` of `_loadDefaultMediaImage`.
    DefaultMedia { step_index: usize, generation: u64 },
    /// The 50 ms `setTimeout` of `_initTextCanvas`.
    RenderText { step_index: usize },
    /// `_loadBuiltinMesh` (an OBJ fetch).
    BuiltinMesh {
        step_index: usize,
        shape: String,
        path: String,
    },
}

/// Page state the ProgramState listeners update (they only see the state).
#[derive(Default)]
struct PageState {
    /// The DSL editor's text (`getDsl()`).
    dsl: String,
    /// A `recompileNeeded` event asked for `_recompilePipeline()`.
    recompile_requested: bool,
}

/// `groupGlobalsByCategory(globals)`: visible globals by `ui.category`
/// (`general` first, then first-occurrence order).
pub fn group_globals_by_category(globals: &Object) -> Vec<(String, Vec<(String, Value)>)> {
    let mut categories: IndexMap<String, Vec<(String, Value)>> = IndexMap::new();
    let mut order: Vec<String> = Vec::new();
    for (key, spec) in globals.iter() {
        let ui = member(spec, "ui");
        if strict_false(&member(&ui, "control")) || member(&ui, "hidden") == Value::Bool(true) {
            continue;
        }
        let category = member(&ui, "category");
        let category = if category.is_truthy() {
            to_js_string(&category)
        } else {
            "general".to_owned()
        };
        if !categories.contains_key(&category) {
            if category != "general" {
                order.push(category.clone());
            }
            categories.insert(category.clone(), Vec::new());
        }
        categories[&category].push((key.clone(), spec.clone()));
    }
    if categories.contains_key("general") {
        order.insert(0, "general".to_owned());
    }
    order
        .into_iter()
        .map(|c| {
            let items = categories.shift_remove(&c).unwrap_or_default();
            (c, items)
        })
        .collect()
}

fn strict_false(v: &Value) -> bool {
    matches!(v, Value::Bool(false))
}

/// `isAutomated` of `_createControlGroup` for one value: an object with a
/// `_varRef` or an oscillator / MIDI / audio type.
fn is_automation_object(v: &Value) -> bool {
    if !v.is_truthy() || !matches!(v, Value::Object(_) | Value::Array(_)) {
        return false;
    }
    if member(v, "_varRef").is_truthy() {
        return true;
    }
    let types = ["Oscillator", "Midi", "Audio"];
    let ty = member(v, "type");
    let ast_ty = member(&member(v, "_ast"), "type");
    types
        .iter()
        .any(|t| ty.as_str() == Some(t) || ast_ty.as_str() == Some(t))
}

/// `_automationBindingsChanged` test of one argument value.
fn is_automation_arg(v: &Value) -> bool {
    if !v.is_truthy() || !matches!(v, Value::Object(_) | Value::Array(_)) {
        return false;
    }
    let types = ["Oscillator", "Midi", "Audio"];
    let ty = member(v, "type");
    let ast_ty = member(&member(v, "_ast"), "type");
    types
        .iter()
        .any(|t| ty.as_str() == Some(t) || ast_ty.as_str() == Some(t))
}

/// `textState` of `_initTextCanvas`, from a step's values.
pub fn text_params_from_values(values: &Object) -> TextParams {
    let get = |k: &str| values.get_or_undefined(k);
    let text = get("text");
    let color = match get("color") {
        Value::Array(items) => TextColor::Array(items.iter().take(3).map(to_number).collect()),
        other => TextColor::Hex(to_js_string(other)),
    };
    TextParams {
        // String(textState.textContent || '')
        text: if text.is_truthy() {
            to_js_string(text)
        } else {
            String::new()
        },
        font: to_js_string(get("font")),
        size: to_number(get("size")),
        pos_x: to_number(get("posX")),
        pos_y: to_number(get("posY")),
        rotation: to_number(get("rotation")),
        color,
        justify: to_js_string(get("justify")),
    }
}

/// `_syncTextInputsFromParams()` for one text state: every member the step's
/// values hold (`textState.x = params.x ?? textState.x`) replaces the current
/// one.
fn merge_text_params(current: &TextParams, values: &Object) -> TextParams {
    let fresh = text_params_from_values(values);
    let has = |key: &str| !values.get_or_undefined(key).is_nullish();
    TextParams {
        text: if has("text") {
            fresh.text
        } else {
            current.text.clone()
        },
        font: if has("font") {
            fresh.font
        } else {
            current.font.clone()
        },
        size: if has("size") {
            fresh.size
        } else {
            current.size
        },
        pos_x: if has("posX") {
            fresh.pos_x
        } else {
            current.pos_x
        },
        pos_y: if has("posY") {
            fresh.pos_y
        } else {
            current.pos_y
        },
        rotation: if has("rotation") {
            fresh.rotation
        } else {
            current.rotation
        },
        color: if has("color") {
            fresh.color
        } else {
            current.color.clone()
        },
        justify: if has("justify") {
            fresh.justify
        } else {
            current.justify.clone()
        },
    }
}

/// The reference demo page's host behavior over a [`CanvasRenderer`].
pub struct DemoHost {
    state: ProgramState<CanvasRenderer>,
    options: DemoHostOptions,
    page: Rc<RefCell<PageState>>,
    /// `_parsedDslStructure`.
    parsed_structure: Vec<EffectInfo>,
    media_generation: u64,
    media_inputs: IndexMap<usize, MediaInput>,
    text_inputs: IndexMap<usize, TextInput>,
    mesh_inputs: IndexMap<usize, MeshInput>,
    tasks: VecDeque<HostTask>,
    shader_overrides: Object,
}

impl DemoHost {
    /// The page with `renderer` (`new UIController(renderer, ...)`).
    pub fn new(renderer: CanvasRenderer, options: DemoHostOptions) -> DemoHost {
        let registry = renderer.registry().clone();
        let state = ProgramState::with_renderer(registry, renderer);
        let page = Rc::new(RefCell::new(PageState::default()));
        if let Some(time) = options.render_on_change {
            // onControlChange: renderSingleFrameIfPaused
            state.on(
                "change",
                listener(move |s: &mut ProgramState<CanvasRenderer>, _| {
                    if let Some(r) = s.renderer_mut() {
                        r.render(time).map_err(|e| JsError::error(e.to_string()))?;
                    }
                    Ok(())
                }),
            );
        }
        // recompileNeeded: _updateDslFromEffectParams(), then
        // _recompilePipeline() (asynchronous; run by settle()).
        let shared = page.clone();
        state.on(
            "recompileNeeded",
            listener(move |s: &mut ProgramState<CanvasRenderer>, _| {
                let new_dsl = s.to_dsl();
                let mut page = shared.borrow_mut();
                if !new_dsl.is_empty() && new_dsl != page.dsl {
                    page.dsl = new_dsl.clone();
                    if let Some(r) = s.renderer_mut() {
                        r.set_current_dsl(new_dsl);
                    }
                }
                page.recompile_requested = true;
                Ok(())
            }),
        );
        DemoHost {
            state,
            options,
            page,
            parsed_structure: Vec::new(),
            media_generation: 0,
            media_inputs: IndexMap::new(),
            text_inputs: IndexMap::new(),
            mesh_inputs: IndexMap::new(),
            tasks: VecDeque::new(),
            shader_overrides: Object::new(),
        }
    }

    /// The renderer.
    pub fn renderer(&self) -> &CanvasRenderer {
        self.state.renderer().expect("the page has its renderer")
    }

    /// The renderer (mutable).
    pub fn renderer_mut(&mut self) -> &mut CanvasRenderer {
        self.state
            .renderer_mut()
            .expect("the page has its renderer")
    }

    /// `renderer.registerPortableEffect(definition)` on the page's renderer
    /// ([`CanvasRenderer::register_portable_effect`]), with the page's
    /// ProgramState resolving effects in the renderer's updated registry, as
    /// the page's shared registries do in the reference.
    pub fn register_portable_effect(
        &mut self,
        definition: &Value,
    ) -> Result<Rc<EffectEntry>, RenderError> {
        let effect = self.renderer_mut().register_portable_effect(definition)?;
        let registry = self.renderer().registry().clone();
        self.state.set_registry(registry);
        Ok(effect)
    }

    /// The page's ProgramState.
    pub fn program_state(&self) -> &ProgramState<CanvasRenderer> {
        &self.state
    }

    /// The page's ProgramState (mutable).
    pub fn program_state_mut(&mut self) -> &mut ProgramState<CanvasRenderer> {
        &mut self.state
    }

    /// The options.
    pub fn options(&self) -> &DemoHostOptions {
        &self.options
    }

    /// `ui.getDsl()`: the editor text.
    pub fn dsl(&self) -> String {
        self.page.borrow().dsl.clone()
    }

    /// `ui.setDsl(dsl)`.
    pub fn set_dsl(&mut self, dsl: &str) {
        self.page.borrow_mut().dsl = dsl.to_owned();
    }

    /// The media inputs by step index.
    pub fn media_inputs(&self) -> &IndexMap<usize, MediaInput> {
        &self.media_inputs
    }

    /// The text inputs by step index.
    pub fn text_inputs(&self) -> &IndexMap<usize, TextInput> {
        &self.text_inputs
    }

    /// The mesh inputs by step index.
    pub fn mesh_inputs(&self) -> &IndexMap<usize, MeshInput> {
        &self.mesh_inputs
    }

    /// `effectParameterValues` (`programState.getAllStepValues()`).
    pub fn effect_parameter_values(&self) -> Object {
        self.state.get_all_step_values()
    }

    fn registry(&self) -> Rc<Registry> {
        self.state.registry().clone()
    }

    fn js(e: JsError) -> RenderError {
        match e {
            JsError::Error { name, message } => RenderError::Js(format!("{name}: {message}")),
            JsError::Thrown(v) => RenderError::Thrown(v),
        }
    }

    /// `needsMidi(dsl)` / `needsAudio(dsl)`: a `midi(`/`audio(` call, or a call
    /// of an effect tagged `midi`/`audio` in the manifest.
    fn needs_input(&self, dsl: &str, call: &str, tag: &str) -> bool {
        let calls = |name: &str| {
            crate::jsre::JsRegex::new(&format!(r"\b{}\s*\(", regex_escape(name)), "").test(dsl)
        };
        if calls(call) {
            return true;
        }
        for (id, info) in self.registry().manifest.iter() {
            let func = id.rsplit('/').next().unwrap_or(id);
            let tagged = info
                .get("tags")
                .as_array()
                .is_some_and(|t| t.iter().any(|v| v.as_str() == Some(tag)));
            if tagged && calls(func) {
                return true;
            }
        }
        false
    }

    /// `rebuildPipelineFromDsl({rebuildControls})` with `dsl` in the editor:
    /// compile, (re)build the controls, apply the step values.
    pub fn rebuild_pipeline_from_dsl(
        &mut self,
        dsl: &str,
        rebuild_controls: bool,
    ) -> Result<(), RenderError> {
        self.set_dsl(dsl);
        if dsl.is_empty() {
            return Err(RenderError::Js("Error: DSL is empty".into()));
        }
        // Clear shader overrides when explicitly rebuilding from DSL.
        self.shader_overrides = Object::new();
        self.renderer_mut().set_current_dsl(dsl);
        // Auto-enable external inputs when the DSL references them.
        if self.options.midi_access
            && self.renderer().midi_state().is_none()
            && self.needs_input(dsl, "midi", "midi")
        {
            self.renderer_mut().set_midi_state(None);
        }
        if self.options.audio_access
            && self.renderer().audio_state().is_none()
            && self.needs_input(dsl, "audio", "audio")
        {
            self.renderer_mut().set_audio_state(None);
        }
        self.renderer_mut()
            .compile(dsl, &CompileOptions::default())?;
        if rebuild_controls && !self.check_structure_and_apply_state(dsl)? {
            self.load_dsl_and_create_controls(dsl)?;
        }
        // Always apply stored parameter values after compile.
        let values = self.state.get_all_step_values();
        self.renderer_mut().apply_step_parameter_values(&values)
    }

    /// `checkStructureAndApplyState(dsl)`: when the program keeps the current
    /// structure and automation bindings, take its values without rebuilding
    /// the controls (`false` asks the caller to rebuild them).
    pub fn check_structure_and_apply_state(&mut self, dsl: &str) -> Result<bool, RenderError> {
        if self.state.would_change_structure(dsl) {
            return Ok(false);
        }
        if self.automation_bindings_changed(dsl) {
            return Ok(false);
        }
        self.state.from_dsl(dsl).map_err(Self::js)?;
        Ok(true)
    }

    /// `_automationBindingsChanged(dsl)`.
    fn automation_bindings_changed(&self, dsl: &str) -> bool {
        let new_effects =
            noisemaker_dsl::program_state::extract_effects_from_dsl(dsl, &self.registry());
        for (i, new_effect) in new_effects.iter().enumerate() {
            let Some(old_effect) = self.parsed_structure.get(i) else {
                continue;
            };
            let mut params: Vec<&String> = new_effect.args.keys().collect();
            for k in old_effect.args.keys() {
                if !params.contains(&k) {
                    params.push(k);
                }
            }
            for param in params {
                let new_auto = is_automation_arg(new_effect.args.get_or_undefined(param));
                let old_auto = is_automation_arg(old_effect.args.get_or_undefined(param));
                if new_auto != old_auto {
                    return true;
                }
            }
        }
        false
    }

    /// `loadDslAndCreateControls(dsl)`: `programState.fromDsl(dsl)`, then the
    /// controls from the state.
    pub fn load_dsl_and_create_controls(&mut self, dsl: &str) -> Result<(), RenderError> {
        self.state.from_dsl(dsl).map_err(Self::js)?;
        self.create_effect_controls_from_state()
    }

    /// `getEffect(effectKey)`, then `namespace.name`, then `name`.
    fn effect_def(&self, info: &EffectInfo) -> Option<Rc<EffectEntry>> {
        let registry = self.registry();
        if let Some(e) = registry.get_effect(&info.effect_key) {
            return Some(e.clone());
        }
        if info.namespace.is_truthy()
            && let Some(e) =
                registry.get_effect(&format!("{}.{}", to_js_string(&info.namespace), info.name))
        {
            return Some(e.clone());
        }
        registry.get_effect(&info.name).cloned()
    }

    /// `createEffectControlsFromState()`: for every effect step, initialize
    /// each visible parameter's control (writing its value through
    /// `setValue`), then its mesh, media and text inputs.
    pub fn create_effect_controls_from_state(&mut self) -> Result<(), RenderError> {
        let structure = self.state.get_structure();
        if structure.is_empty() {
            self.stop_all_media(false);
            self.text_inputs.clear();
            self.parsed_structure = Vec::new();
            return Ok(());
        }
        let has_plans = self
            .state
            .get_compiled()
            .is_some_and(|c| member(c, "plans").is_truthy());
        if !has_plans {
            return Ok(());
        }
        // Preserve media state (by occurrence) before rebuilding.
        let previous_media = self.preserve_media_state();
        self.stop_all_media(false);
        self.text_inputs.clear();
        self.state.clear_routing_overrides();
        self.parsed_structure = structure.clone();

        let mut occurrence_count: IndexMap<String, usize> = IndexMap::new();
        for info in &structure {
            let effect_name = if info.effect_key.is_empty() {
                info.name.clone()
            } else {
                info.effect_key.clone()
            };
            let occurrence = *occurrence_count.entry(effect_name.clone()).or_insert(0);
            if matches!(
                info.effect_key.as_str(),
                "_write" | "_read" | "_read3d" | "_write3d"
            ) {
                occurrence_count[&effect_name] += 1;
                continue;
            }
            let Some(entry) = self.effect_def(info) else {
                occurrence_count[&effect_name] += 1;
                continue;
            };
            let def = &entry.def;
            let globals = def.get("globals");
            if !globals.is_truthy() {
                occurrence_count[&effect_name] += 1;
                continue;
            }
            let effect_key = format!("step_{}", info.step_index);
            occurrence_count[&effect_name] += 1;
            let occurrence_key = format!("{effect_name}#{occurrence}");

            let grouped = group_globals_by_category(globals.as_object().unwrap_or(&Object::new()));
            for (_, items) in &grouped {
                for (key, spec) in items {
                    self.init_control(key, spec, info, &effect_key)?;
                }
            }

            // Mesh input section (effects with externalMesh).
            let external_mesh = def.get("externalMesh");
            if external_mesh.is_truthy() {
                self.create_mesh_input_section(
                    info.step_index,
                    &to_js_string(external_mesh),
                    def.get("builtinMeshes"),
                );
            }

            // Media input section (externalTexture other than textTex).
            let external_texture = def.get("externalTexture");
            if external_texture.is_truthy() && external_texture.as_str() != Some("textTex") {
                let texture_id = format!(
                    "{}_step_{}",
                    to_js_string(external_texture),
                    info.step_index
                );
                let preserved = previous_media.get(&occurrence_key).cloned();
                self.media_generation += 1;
                let generation = self.media_generation;
                self.media_inputs.insert(
                    info.step_index,
                    MediaInput {
                        texture_id,
                        source: None,
                        generation,
                    },
                );
                match preserved {
                    // A preserved file source is restored; a section that had
                    // none skips the default load and restores nothing.
                    Some(Some(source)) => self.restore_media(info.step_index, source)?,
                    Some(None) => {}
                    None => self.tasks.push_back(HostTask::DefaultMedia {
                        step_index: info.step_index,
                        generation,
                    }),
                }
            }

            // Text canvas (externalTexture 'textTex').
            if external_texture.as_str() == Some("textTex") {
                self.init_text_canvas(info.step_index, &effect_key, "textTex");
            }
        }
        Ok(())
    }

    /// The parameter-writing part of `_createControlGroup(key, spec,
    /// effectInfo, effectKey)` and of the control it creates.
    fn init_control(
        &mut self,
        key: &str,
        spec: &Value,
        info: &EffectInfo,
        effect_key: &str,
    ) -> Result<(), RenderError> {
        let ui = member(spec, "ui");
        if strict_false(&member(&ui, "control")) || member(&ui, "hidden") == Value::Bool(true) {
            return Ok(());
        }
        // Prefer the value already in the state, then the DSL argument, then
        // the default.
        let preserved = self.state.get_value(effect_key, key);
        let value = if !preserved.is_undefined() {
            preserved
        } else if !info.args.get_or_undefined(key).is_undefined() {
            info.args.get_or_undefined(key).clone()
        } else {
            member(spec, "default")
        };
        let raw_kwarg = member(&info.raw_kwargs, key);
        let automation_value =
            if value.is_truthy() && matches!(value, Value::Object(_) | Value::Array(_)) {
                value.clone()
            } else {
                info.args.get_or_undefined(key).clone()
            };
        let raw_is_automation = raw_kwarg.is_truthy()
            && matches!(raw_kwarg, Value::Object(_) | Value::Array(_))
            && matches!(
                member(&raw_kwarg, "type").as_str(),
                Some("Oscillator" | "Midi" | "Audio")
            );
        if is_automation_object(&automation_value) || raw_is_automation {
            if raw_kwarg.is_truthy() && member(&raw_kwarg, "type").as_str() == Some("Ident") {
                let mut binding = Object::new();
                binding.insert("_varRef", member(&raw_kwarg, "name"));
                self.state
                    .set_value(effect_key, key, Value::Object(binding))
                    .map_err(Self::js)?;
            }
            return Ok(());
        }
        // Initialize the value in program state.
        self.state
            .set_value(effect_key, key, value.clone())
            .map_err(Self::js)?;

        // The control types; only the member control writes at creation.
        let control = member(&ui, "control");
        let ty = member(spec, "type");
        let is = |v: &Value, s: &str| v.as_str() == Some(s);
        if is(&control, "button")
            || is(&control, "checkbox")
            || is(&ty, "boolean")
            || is(&control, "color")
            || is(&ty, "vec4")
            || is(&ty, "vec2")
            || is(&ty, "vec3")
            || member(spec, "choices").is_truthy()
            || (member(spec, "enum").is_truthy() && is(&ty, "int"))
        {
            return Ok(());
        }
        if is(&ty, "member") {
            self.init_member_control(key, spec, &value, effect_key)?;
        }
        Ok(())
    }

    /// `_createMemberControl`: the select stores the enum path of the value
    /// (or the first entry when nothing matches).
    fn init_member_control(
        &mut self,
        key: &str,
        spec: &Value,
        value: &Value,
        effect_key: &str,
    ) -> Result<(), RenderError> {
        let mut enum_path = member(spec, "enum");
        if !enum_path.is_truthy() {
            enum_path = member(spec, "enumPath");
        }
        if !enum_path.is_truthy()
            && let Value::String(default) = member(spec, "default")
        {
            let parts: Vec<&str> = default.split('.').collect();
            if parts.len() > 1 {
                enum_path = Value::from(parts[..parts.len() - 1].join("."));
            }
        }
        if !enum_path.is_truthy() {
            return Ok(());
        }
        let enum_path = to_js_string(&enum_path);
        let enums = self.renderer().enums().clone();
        let mut node: Value = (*enums).clone();
        for part in enum_path.split('.') {
            let next = member(&node, part);
            if node.is_truthy() && next.is_truthy() {
                node = next;
            } else {
                node = Value::Null;
                break;
            }
        }
        if !node.is_truthy() {
            return Ok(());
        }
        let mut entries: Vec<(String, Value)> = Vec::new();
        if let Value::Object(o) = &node {
            for (k, entry) in o.iter() {
                let numeric = if matches!(entry, Value::Object(_) | Value::Array(_))
                    && entry.is_truthy()
                    && noisemaker_dsl::unparser::jsv::has_property(entry, "value")
                {
                    member(entry, "value")
                } else {
                    entry.clone()
                };
                entries.push((format!("{enum_path}.{k}"), numeric));
            }
        }
        let mut initial = entries
            .first()
            .map(|(p, _)| Value::from(p.as_str()))
            .unwrap_or(Value::Undefined);
        for (path, numeric) in &entries {
            if crate::jsv::strict_equals(numeric, value) || value.as_str() == Some(path.as_str()) {
                initial = Value::from(path.as_str());
                break;
            }
        }
        self.state
            .set_value(effect_key, key, initial)
            .map_err(Self::js)
    }

    /// `_createMeshInputSection(stepIndex, meshId, builtinMeshes)`: auto-load
    /// the first built-in shape.
    fn create_mesh_input_section(&mut self, step_index: usize, mesh_id: &str, builtins: &Value) {
        self.mesh_inputs
            .entry(step_index)
            .or_insert_with(|| MeshInput {
                mesh_id: mesh_id.to_owned(),
                loaded: false,
                vertex_count: 0,
                status: "no mesh loaded".into(),
            });
        if !builtins.is_truthy() {
            return;
        }
        if let Some((shape, path)) = builtins.as_object().and_then(|b| b.iter().next()) {
            if let Some(input) = self.mesh_inputs.get_mut(&step_index) {
                input.status = "loading...".into();
            }
            self.tasks.push_back(HostTask::BuiltinMesh {
                step_index,
                shape: shape.clone(),
                path: to_js_string(path),
            });
        }
    }

    /// `_initTextCanvas(stepIndex, effectKey, effectDef)`: capture the step's
    /// text parameters and schedule the canvas render.
    fn init_text_canvas(&mut self, step_index: usize, effect_key: &str, external_texture: &str) {
        let values = self.state.get_step_values(effect_key);
        self.text_inputs.insert(
            step_index,
            TextInput {
                texture_id: format!("{external_texture}_step_{step_index}"),
                effect_key: effect_key.to_owned(),
                params: text_params_from_values(&values),
            },
        );
        self.tasks.push_back(HostTask::RenderText { step_index });
    }

    /// `stopAllMedia(preserveState)`.
    fn stop_all_media(&mut self, preserve_state: bool) {
        if !preserve_state {
            self.media_inputs.clear();
        }
    }

    /// `_preserveMediaState()`: the media source of each media step, keyed by
    /// effect occurrence.
    fn preserve_media_state(&self) -> IndexMap<String, Option<Rc<Rgba8Image>>> {
        let mut preserved = IndexMap::new();
        if self.media_inputs.is_empty() {
            return preserved;
        }
        let mut occurrence_count: IndexMap<String, usize> = IndexMap::new();
        for info in &self.parsed_structure {
            let name = if info.effect_key.is_empty() {
                info.name.clone()
            } else {
                info.effect_key.clone()
            };
            let occurrence = *occurrence_count.entry(name.clone()).or_insert(0);
            occurrence_count[&name] += 1;
            let Some(media) = self.media_inputs.get(&info.step_index) else {
                continue;
            };
            preserved.insert(format!("{name}#{occurrence}"), media.source.clone());
        }
        preserved
    }

    /// `_restoreMediaFromPreviousState` for a file image.
    fn restore_media(
        &mut self,
        step_index: usize,
        source: Rc<Rgba8Image>,
    ) -> Result<(), RenderError> {
        if let Some(media) = self.media_inputs.get_mut(&step_index) {
            media.source = Some(source);
        }
        self.update_media_texture(step_index)?;
        self.apply_step_params()
    }

    /// `_applyStepParams()`.
    fn apply_step_params(&mut self) -> Result<(), RenderError> {
        let values = self.state.get_all_step_values();
        self.renderer_mut().apply_step_parameter_values(&values)
    }

    /// `_updateMediaTexture(stepIndex)`: upload the step's image (`flipY:
    /// false`) and set its `imageSize`.
    fn update_media_texture(&mut self, step_index: usize) -> Result<(), RenderError> {
        let Some(media) = self.media_inputs.get(&step_index).cloned() else {
            return Ok(());
        };
        let Some(source) = media.source else {
            return Ok(());
        };
        if self.renderer_mut().pipeline().is_none() {
            return Ok(());
        }
        let (w, h) = self.renderer_mut().update_texture_from_source(
            &media.texture_id,
            source.width,
            source.height,
            &source.data,
            TextureUpdateOptions { flip_y: false },
        )?;
        if w > 0 && h > 0 {
            self.state
                .set_value(
                    &format!("step_{step_index}"),
                    "imageSize",
                    Value::Array(vec![Value::Number(w as f64), Value::Number(h as f64)]),
                )
                .map_err(Self::js)?;
        }
        Ok(())
    }

    /// A parameter control's change: `programState.setValue(stepKey,
    /// paramName, value)` (validated and coerced, then applied to the
    /// pipeline), then the page's `_onControlChange()`: the editor DSL follows
    /// the state (`_updateDslFromEffectParams`) and every text canvas re-reads
    /// its step's values and is drawn again (`_syncTextInputsFromParams`). A
    /// changed compile-time (`define`) parameter requests a recompile, which
    /// [`DemoHost::settle`] runs.
    pub fn set_control_value(
        &mut self,
        step_key: &str,
        param_name: &str,
        value: Value,
    ) -> Result<(), RenderError> {
        self.state
            .set_value(step_key, param_name, value)
            .map_err(Self::js)?;
        self.on_control_change()
    }

    /// `_onControlChange()` without its DOM duties.
    fn on_control_change(&mut self) -> Result<(), RenderError> {
        // _updateDslFromEffectParams
        let new_dsl = self.state.to_dsl();
        if !new_dsl.is_empty() && new_dsl != self.dsl() {
            self.set_dsl(&new_dsl);
            self.renderer_mut().set_current_dsl(new_dsl);
        }
        // _syncTextInputsFromParams
        let steps: Vec<usize> = self.text_inputs.keys().copied().collect();
        for step_index in steps {
            let Some(input) = self.text_inputs.get(&step_index) else {
                continue;
            };
            let values = self.state.get_step_values(&input.effect_key);
            let params = merge_text_params(&input.params, &values);
            if let Some(input) = self.text_inputs.get_mut(&step_index) {
                input.params = params;
            }
            self.render_text(step_index)?;
        }
        Ok(())
    }

    /// The media file input (`_handleMediaFileChange` for an image), or a
    /// playing video's next frame (`_updateAllMediaTextures`): show `image`
    /// in the step's media texture and apply the step values.
    pub fn set_media_image(
        &mut self,
        step_index: usize,
        image: Rc<Rgba8Image>,
    ) -> Result<(), RenderError> {
        let Some(media) = self.media_inputs.get_mut(&step_index) else {
            return Ok(());
        };
        media.source = Some(image);
        self.update_media_texture(step_index)?;
        self.apply_step_params()
    }

    /// `_renderTextToCanvas(stepIndex)` and `_updateTextTexture(stepIndex)`:
    /// draw the step's text at the renderer width, upload it as its
    /// `textTex_step_N` (`flipY: true` of the canvas) and set `textSize`.
    fn render_text(&mut self, step_index: usize) -> Result<(), RenderError> {
        let Some(input) = self.text_inputs.get(&step_index).cloned() else {
            return Ok(());
        };
        if self.renderer_mut().pipeline().is_none() {
            return Ok(());
        }
        let (w, h) = demo_canvas_size(self.renderer().width(), self.renderer().height());
        let image = noisemaker_host::text::render_text_canvas_with(
            &self.options.text_fonts,
            &self.options.text_style,
            &input.params,
            w,
            h,
        );
        let (w, h) = self.renderer_mut().update_texture_from_source(
            &input.texture_id,
            image.width,
            image.height,
            &image.data,
            TextureUpdateOptions { flip_y: false },
        )?;
        if w > 0 && h > 0 {
            self.state
                .set_value(
                    &format!("step_{step_index}"),
                    "textSize",
                    Value::Array(vec![Value::Number(w as f64), Value::Number(h as f64)]),
                )
                .map_err(Self::js)?;
        }
        Ok(())
    }

    /// `_loadBuiltinMesh(stepIndex, shapeName, relativePath)`.
    fn load_builtin_mesh(&mut self, step_index: usize, shape: &str, path: &str) {
        let Some(mesh_id) = self.mesh_inputs.get(&step_index).map(|m| m.mesh_id.clone()) else {
            return;
        };
        let result = self.renderer_mut().load_builtin_mesh(path, &mesh_id);
        if let Some(input) = self.mesh_inputs.get_mut(&step_index) {
            if result.success {
                input.loaded = true;
                input.vertex_count = result.vertex_count;
                input.status = format!("{shape}: {} vertices", result.vertex_count);
            } else {
                input.status = format!("error: {}", result.error.as_deref().unwrap_or("unknown"));
            }
        }
    }

    fn run_task(&mut self, task: HostTask) -> Result<(), RenderError> {
        match task {
            HostTask::DefaultMedia {
                step_index,
                generation,
            } => {
                let Some(image) = self.options.default_media.clone() else {
                    // img.onerror: "no media loaded".
                    return Ok(());
                };
                // The load belongs to the section that started it; a rebuilt
                // section at the same step keeps its own source.
                if let Some(media) = self.media_inputs.get_mut(&step_index)
                    && media.generation == generation
                {
                    media.source = Some(image);
                }
                self.update_media_texture(step_index)?;
                self.apply_step_params()
            }
            HostTask::RenderText { step_index } => self.render_text(step_index),
            HostTask::BuiltinMesh {
                step_index,
                shape,
                path,
            } => {
                self.load_builtin_mesh(step_index, &shape, &path);
                Ok(())
            }
        }
    }

    /// `_recompilePipeline()`: recompile the editor's DSL, then rebind the
    /// state and controls.
    fn recompile_pipeline(&mut self) -> Result<(), RenderError> {
        let dsl = self.dsl();
        if dsl.is_empty() {
            return Ok(());
        }
        let options = CompileOptions {
            shader_overrides: self.shader_overrides.clone(),
        };
        self.renderer_mut().compile(&dsl, &options)?;
        if !self.check_structure_and_apply_state(&dsl)? {
            self.load_dsl_and_create_controls(&dsl)?;
        }
        Ok(())
    }

    /// Whether host inputs are still loading.
    pub fn has_pending_inputs(&self) -> bool {
        !self.tasks.is_empty() || self.page.borrow().recompile_requested
    }

    /// Complete everything the page does asynchronously: queued host inputs
    /// (media, text, meshes, in the order they were started), requested
    /// recompiles, then the pipeline's debounced overlay regenerations and
    /// overlay traces.
    pub fn settle(&mut self) -> Result<(), RenderError> {
        loop {
            while let Some(task) = self.tasks.pop_front() {
                self.run_task(task)?;
            }
            let recompile = std::mem::take(&mut self.page.borrow_mut().recompile_requested);
            if recompile {
                self.recompile_pipeline()?;
                continue;
            }
            if self.tasks.is_empty() {
                break;
            }
        }
        if let Some(p) = self.renderer_mut().pipeline_mut() {
            p.settle_async_effects()?;
        }
        Ok(())
    }
}

fn regex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if "\\^$.|?*+()[]{}".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categories_put_general_first() {
        let globals = Value::from_json(
            r#"{"a":{"ui":{"category":"transform"}},"b":{},"c":{"ui":{"control":false}},
                "d":{"ui":{"category":"transform"}},"e":{"ui":{"category":"color"}}}"#,
        )
        .unwrap();
        let grouped = group_globals_by_category(globals.as_object().unwrap());
        let names: Vec<(String, Vec<String>)> = grouped
            .into_iter()
            .map(|(c, items)| (c, items.into_iter().map(|(k, _)| k).collect()))
            .collect();
        assert_eq!(
            names,
            vec![
                ("general".into(), vec!["b".into()]),
                ("transform".into(), vec!["a".into(), "d".into()]),
                ("color".into(), vec!["e".into()]),
            ]
        );
    }

    #[test]
    fn text_state_reads_like_the_demo() {
        let values = Value::from_json(
            r#"{"text":"Hi","font":"serif","size":0.2,"posX":0.25,"posY":0.75,
                "rotation":15,"color":[1,0.5,0,1],"justify":"left"}"#,
        )
        .unwrap();
        let p = text_params_from_values(values.as_object().unwrap());
        assert_eq!(p.text, "Hi");
        assert_eq!(p.font, "serif");
        assert_eq!(p.color, TextColor::Array(vec![1.0, 0.5, 0.0]));
        let empty = text_params_from_values(&Object::new());
        assert_eq!(empty.text, "");
        assert_eq!(empty.font, "undefined");
        assert!(empty.size.is_nan());
        assert_eq!(empty.color, TextColor::Hex("undefined".into()));
    }
}
