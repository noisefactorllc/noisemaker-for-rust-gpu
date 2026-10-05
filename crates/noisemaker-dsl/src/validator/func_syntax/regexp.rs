//! Regular-expression literal validation, as V8 performs it while parsing.
//!
//! A port of the Qt port's `js_regexp.cpp`, itself a port of the RegExp
//! validator in acorn 8.16 (`acorn/dist/acorn.js`, `pp$1.regexp_*`). The acorn
//! code carries this notice:
//!
//! > MIT License
//! >
//! > Copyright (C) 2012-2022 by various contributors (see AUTHORS)
//! >
//! > Permission is hereby granted, free of charge, to any person obtaining a copy
//! > of this software and associated documentation files (the "Software"), to deal
//! > in the Software without restriction, including without limitation the rights
//! > to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! > copies of the Software, and to permit persons to whom the Software is
//! > furnished to do so, subject to the following conditions:
//! >
//! > The above copyright notice and this permission notice shall be included in
//! > all copies or substantial portions of the Software.
//! >
//! > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! > IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! > FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! > AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! > LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! > OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! > SOFTWARE.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use super::unicode::{is_id_part, is_id_start};

/// The verdict on one regular-expression literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RegexCheck {
    Valid,
    /// Rejected; the text is a reason for logs (not a V8 message).
    Invalid(String),
    /// Groups nested deeper than the validator follows.
    Undecidable(String),
}

enum RegexFail {
    Error(String),
    TooDeep,
}

type RR<T> = Result<T, RegexFail>;

/// Groups and classes nested deeper than this are not decided: the validator
/// recurses once per level (about 0.5 KiB of stack each), while V8's regexp
/// parser nests groups without recursing (up to the capture limit) and
/// overflows only on nested `v`-flag classes, 5153 levels deep.
const MAX_REGEX_DEPTH: i32 = 400;

/// V8's limit on capturing groups (`RegExpMacroAssembler::kMaxCaptures`):
/// a pattern with more is a SyntaxError.
const MAX_CAPTURES: f64 = 32767.0;

// ECMA-262 tables of Unicode property names and values (acorn's
// unicode-property-data, Script values through Unicode 17). V8 rejects the Script
// value Katakana_Or_Hiragana (Hrkt), which acorn lists, so it is omitted; every
// name and value here was checked against the oracle's V8.
const BINARY_PROPERTIES: &str = "ASCII ASCII_Hex_Digit AHex Alphabetic Alpha Any Assigned Bidi_Control Bidi_C Bidi_Mirrored Bidi_M \
    Case_Ignorable CI Cased Changes_When_Casefolded CWCF Changes_When_Casemapped CWCM Changes_When_Lowercased CWL \
    Changes_When_NFKC_Casefolded CWKCF Changes_When_Titlecased CWT Changes_When_Uppercased CWU Dash \
    Default_Ignorable_Code_Point DI Deprecated Dep Diacritic Dia Emoji Emoji_Component Emoji_Modifier \
    Emoji_Modifier_Base Emoji_Presentation Extender Ext Grapheme_Base Gr_Base Grapheme_Extend Gr_Ext Hex_Digit Hex \
    IDS_Binary_Operator IDSB IDS_Trinary_Operator IDST ID_Continue IDC ID_Start IDS Ideographic Ideo Join_Control \
    Join_C Logical_Order_Exception LOE Lowercase Lower Math Noncharacter_Code_Point NChar Pattern_Syntax Pat_Syn \
    Pattern_White_Space Pat_WS Quotation_Mark QMark Radical Regional_Indicator RI Sentence_Terminal STerm \
    Soft_Dotted SD Terminal_Punctuation Term Unified_Ideograph UIdeo Uppercase Upper Variation_Selector VS \
    White_Space space XID_Continue XIDC XID_Start XIDS Extended_Pictographic EBase EComp EMod EPres ExtPict";
const BINARY_PROPERTIES_OF_STRINGS: &str = "Basic_Emoji Emoji_Keycap_Sequence RGI_Emoji_Modifier_Sequence RGI_Emoji_Flag_Sequence \
    RGI_Emoji_Tag_Sequence RGI_Emoji_ZWJ_Sequence RGI_Emoji";
const GENERAL_CATEGORY_VALUES: &str = "Cased_Letter LC Close_Punctuation Pe Connector_Punctuation Pc Control Cc cntrl Currency_Symbol Sc \
    Dash_Punctuation Pd Decimal_Number Nd digit Enclosing_Mark Me Final_Punctuation Pf Format Cf \
    Initial_Punctuation Pi Letter L Letter_Number Nl Line_Separator Zl Lowercase_Letter Ll Mark M Combining_Mark \
    Math_Symbol Sm Modifier_Letter Lm Modifier_Symbol Sk Nonspacing_Mark Mn Number N Open_Punctuation Ps Other C \
    Other_Letter Lo Other_Number No Other_Punctuation Po Other_Symbol So Paragraph_Separator Zp Private_Use Co \
    Punctuation P punct Separator Z Space_Separator Zs Spacing_Mark Mc Surrogate Cs Symbol S Titlecase_Letter Lt \
    Unassigned Cn Uppercase_Letter Lu";
const SCRIPT_VALUES: &str = "Adlam Adlm Ahom Anatolian_Hieroglyphs Hluw Arabic Arab Armenian Armn Avestan Avst Balinese Bali Bamum Bamu \
    Bassa_Vah Bass Batak Batk Bengali Beng Bhaiksuki Bhks Bopomofo Bopo Brahmi Brah Braille Brai Buginese Bugi \
    Buhid Buhd Canadian_Aboriginal Cans Carian Cari Caucasian_Albanian Aghb Chakma Cakm Cham Cham Cherokee Cher \
    Common Zyyy Coptic Copt Qaac Cuneiform Xsux Cypriot Cprt Cyrillic Cyrl Deseret Dsrt Devanagari Deva Duployan \
    Dupl Egyptian_Hieroglyphs Egyp Elbasan Elba Ethiopic Ethi Georgian Geor Glagolitic Glag Gothic Goth Grantha \
    Gran Greek Grek Gujarati Gujr Gurmukhi Guru Han Hani Hangul Hang Hanunoo Hano Hatran Hatr Hebrew Hebr Hiragana \
    Hira Imperial_Aramaic Armi Inherited Zinh Qaai Inscriptional_Pahlavi Phli Inscriptional_Parthian Prti Javanese \
    Java Kaithi Kthi Kannada Knda Katakana Kana Kayah_Li Kali Kharoshthi Khar Khmer Khmr Khojki Khoj Khudawadi Sind \
    Lao Laoo Latin Latn Lepcha Lepc Limbu Limb Linear_A Lina Linear_B Linb Lisu Lisu Lycian Lyci Lydian Lydi \
    Mahajani Mahj Malayalam Mlym Mandaic Mand Manichaean Mani Marchen Marc Masaram_Gondi Gonm Meetei_Mayek Mtei \
    Mende_Kikakui Mend Meroitic_Cursive Merc Meroitic_Hieroglyphs Mero Miao Plrd Modi Mongolian Mong Mro Mroo \
    Multani Mult Myanmar Mymr Nabataean Nbat New_Tai_Lue Talu Newa Newa Nko Nkoo Nushu Nshu Ogham Ogam Ol_Chiki \
    Olck Old_Hungarian Hung Old_Italic Ital Old_North_Arabian Narb Old_Permic Perm Old_Persian Xpeo \
    Old_South_Arabian Sarb Old_Turkic Orkh Oriya Orya Osage Osge Osmanya Osma Pahawh_Hmong Hmng Palmyrene Palm \
    Pau_Cin_Hau Pauc Phags_Pa Phag Phoenician Phnx Psalter_Pahlavi Phlp Rejang Rjng Runic Runr Samaritan Samr \
    Saurashtra Saur Sharada Shrd Shavian Shaw Siddham Sidd SignWriting Sgnw Sinhala Sinh Sora_Sompeng Sora \
    Soyombo Soyo Sundanese Sund Syloti_Nagri Sylo Syriac Syrc Tagalog Tglg Tagbanwa Tagb Tai_Le Tale Tai_Tham Lana \
    Tai_Viet Tavt Takri Takr Tamil Taml Tangut Tang Telugu Telu Thaana Thaa Thai Thai Tibetan Tibt Tifinagh Tfng \
    Tirhuta Tirh Ugaritic Ugar Vai Vaii Warang_Citi Wara Yi Yiii Zanabazar_Square Zanb \
    Dogra Dogr Gunjala_Gondi Gong Hanifi_Rohingya Rohg Makasar Maka Medefaidrin Medf Old_Sogdian Sogo Sogdian Sogd \
    Elymaic Elym Nandinagari Nand Nyiakeng_Puachue_Hmong Hmnp Wancho Wcho \
    Chorasmian Chrs Diak Dives_Akuru Khitan_Small_Script Kits Yezi Yezidi \
    Cypro_Minoan Cpmn Old_Uyghur Ougr Tangsa Tnsa Toto Vithkuqi Vith \
    Berf Beria_Erfe Gara Garay Gukh Gurung_Khema Kawi Kirat_Rai Krai Nag_Mundari Nagm \
    Ol_Onal Onao Sidetic Sidt Sunu Sunuwar Tai_Yo Tayo Todhri Todr Tolong_Siki Tols Tulu_Tigalari Tutg Unknown Zzzz";

struct PropertyData {
    /// Binary properties and lone General_Category values.
    binary: HashSet<&'static str>,
    /// Binary properties of strings (`v` flag only).
    binary_of_strings: HashSet<&'static str>,
    general_category: HashSet<&'static str>,
    script: HashSet<&'static str>,
}

fn property_data() -> &'static PropertyData {
    static DATA: OnceLock<PropertyData> = OnceLock::new();
    DATA.get_or_init(|| {
        let words = |list: &'static str| list.split_whitespace().collect::<HashSet<_>>();
        let mut binary = words(BINARY_PROPERTIES);
        binary.extend(words(GENERAL_CATEGORY_VALUES));
        PropertyData {
            binary,
            binary_of_strings: words(BINARY_PROPERTIES_OF_STRINGS),
            general_category: words(GENERAL_CATEGORY_VALUES),
            script: words(SCRIPT_VALUES),
        }
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CharSet {
    None,
    Ok,
    String,
}

/// Disjunction structure, to allow a duplicate group name in a separate
/// alternative (ES2025 duplicate named groups).
#[derive(Clone, Copy)]
struct Branch {
    parent: Option<usize>,
    base: usize,
}

fn is_syntax_character(ch: i64) -> bool {
    ch == 0x24
        || (0x28..=0x2B).contains(&ch)
        || ch == 0x2E
        || ch == 0x3F
        || (0x5B..=0x5E).contains(&ch)
        || (0x7B..=0x7D).contains(&ch)
}
fn is_decimal_digit(ch: i64) -> bool {
    (0x30..=0x39).contains(&ch)
}
fn is_hex_digit(ch: i64) -> bool {
    (0x30..=0x39).contains(&ch) || (0x41..=0x46).contains(&ch) || (0x61..=0x66).contains(&ch)
}
fn hex_to_int(ch: i64) -> f64 {
    if (0x41..=0x46).contains(&ch) {
        return (10 + ch - 0x41) as f64;
    }
    if (0x61..=0x66).contains(&ch) {
        return (10 + ch - 0x61) as f64;
    }
    (ch - 0x30) as f64
}
fn is_octal_digit(ch: i64) -> bool {
    (0x30..=0x37).contains(&ch)
}
fn is_control_letter(ch: i64) -> bool {
    (0x41..=0x5A).contains(&ch) || (0x61..=0x7A).contains(&ch)
}
fn is_character_class_escape(ch: i64) -> bool {
    matches!(ch, 0x64 | 0x44 | 0x73 | 0x53 | 0x77 | 0x57)
}
fn is_unicode_property_name_character(ch: i64) -> bool {
    is_control_letter(ch) || ch == 0x5F
}
fn is_unicode_property_value_character(ch: i64) -> bool {
    is_unicode_property_name_character(ch) || is_decimal_digit(ch)
}
fn is_regular_expression_modifier(ch: i64) -> bool {
    matches!(ch, 0x69 | 0x6D | 0x73)
}
fn is_regexp_identifier_start(ch: i64) -> bool {
    ch >= 0 && (is_id_start(ch as u32) || ch == 0x24 || ch == 0x5F)
}
fn is_regexp_identifier_part(ch: i64) -> bool {
    ch >= 0 && (is_id_part(ch as u32) || ch == 0x24 || ch == 0x5F || ch == 0x200C || ch == 0x200D)
}
fn is_class_set_reserved_double_punctuator_character(ch: i64) -> bool {
    ch == 0x21
        || (0x23..=0x26).contains(&ch)
        || (0x2A..=0x2C).contains(&ch)
        || ch == 0x2E
        || (0x3A..=0x40).contains(&ch)
        || ch == 0x5E
        || ch == 0x60
        || ch == 0x7E
}
fn is_class_set_syntax_character(ch: i64) -> bool {
    matches!(ch, 0x28 | 0x29 | 0x2D | 0x2F)
        || (0x5B..=0x5D).contains(&ch)
        || (0x7B..=0x7D).contains(&ch)
}
fn is_class_set_reserved_punctuator(ch: i64) -> bool {
    matches!(
        ch,
        0x21 | 0x23 | 0x25 | 0x26 | 0x2C | 0x2D | 0x40 | 0x60 | 0x7E
    ) || (0x3A..=0x3E).contains(&ch)
}

fn push_code_point(out: &mut String, cp: i64) {
    if let Some(c) = u32::try_from(cp).ok().and_then(char::from_u32) {
        out.push(c);
    } else if cp >= 0 {
        // A lone surrogate never names a group or a property (neither is an
        // identifier or property-name character), so a placeholder is exact.
        out.push('\u{FFFD}');
    }
}

struct RegexValidator<'a> {
    source: &'a [u16],
    flags: &'a [u16],
    switch_u: bool,
    switch_v: bool,
    switch_n: bool,
    pos: usize,
    last_int_value: f64,
    last_string_value: String,
    last_assertion_is_quantifiable: bool,
    num_capturing_parens: f64,
    max_back_reference: f64,
    group_names: HashMap<String, Vec<usize>>,
    back_reference_names: Vec<String>,
    branches: Vec<Branch>,
    branch_id: Option<usize>,
    depth: i32,
}

impl<'a> RegexValidator<'a> {
    fn new(source: &'a [u16], flags: &'a [u16]) -> Self {
        let has = |c: u8| flags.contains(&u16::from(c));
        let unicode_sets = has(b'v');
        let unicode = has(b'u');
        let (switch_u, switch_v, switch_n) = if unicode_sets {
            (true, true, true)
        } else {
            (unicode, false, unicode)
        };
        RegexValidator {
            source,
            flags,
            switch_u,
            switch_v,
            switch_n,
            pos: 0,
            last_int_value: 0.0,
            last_string_value: String::new(),
            last_assertion_is_quantifiable: false,
            num_capturing_parens: 0.0,
            max_back_reference: 0.0,
            group_names: HashMap::new(),
            back_reference_names: Vec::new(),
            branches: Vec::new(),
            branch_id: None,
            depth: 0,
        }
    }

    fn validate(&mut self) -> RR<()> {
        self.validate_flags()?;
        self.pattern()?;
        // A pattern with a GroupName is reparsed with the N parameter.
        if !self.switch_n && !self.group_names.is_empty() {
            self.switch_n = true;
            self.pattern()?;
        }
        Ok(())
    }

    fn raise<T>(&self, message: &str) -> RR<T> {
        Err(RegexFail::Error(format!(
            "Invalid regular expression: /{}/: {message}",
            String::from_utf16_lossy(self.source)
        )))
    }

    /// The code point at index `i` when the u (or forced) mode is on, else the
    /// code unit; -1 past the end.
    fn at(&self, i: usize, force_u: bool) -> i64 {
        let l = self.source.len();
        if i >= l {
            return -1;
        }
        let c = self.source[i];
        if !(force_u || self.switch_u) || !(0xD800..=0xDBFF).contains(&c) || i + 1 >= l {
            return i64::from(c);
        }
        let next = self.source[i + 1];
        if (0xDC00..=0xDFFF).contains(&next) {
            return 0x10000 + ((i64::from(c) - 0xD800) << 10) + (i64::from(next) - 0xDC00);
        }
        i64::from(c)
    }

    fn next_index(&self, i: usize, force_u: bool) -> usize {
        let l = self.source.len();
        if i >= l {
            return l;
        }
        let c = self.source[i];
        if !(force_u || self.switch_u) || !(0xD800..=0xDBFF).contains(&c) || i + 1 >= l {
            return i + 1;
        }
        let next = self.source[i + 1];
        if (0xDC00..=0xDFFF).contains(&next) {
            i + 2
        } else {
            i + 1
        }
    }

    fn current(&self) -> i64 {
        self.at(self.pos, false)
    }
    fn current_u(&self, force_u: bool) -> i64 {
        self.at(self.pos, force_u)
    }
    fn lookahead(&self) -> i64 {
        self.at(self.next_index(self.pos, false), false)
    }
    fn advance(&mut self) {
        self.pos = self.next_index(self.pos, false);
    }
    fn advance_u(&mut self, force_u: bool) {
        self.pos = self.next_index(self.pos, force_u);
    }
    fn eat(&mut self, ch: i64) -> bool {
        if self.current() == ch {
            self.advance();
            return true;
        }
        false
    }
    fn eat_chars(&mut self, chs: &[i64]) -> bool {
        let mut pos = self.pos;
        for &ch in chs {
            let cur = self.at(pos, false);
            if cur == -1 || cur != ch {
                return false;
            }
            pos = self.next_index(pos, false);
        }
        self.pos = pos;
        true
    }

    fn validate_flags(&self) -> RR<()> {
        let mut u = false;
        let mut v = false;
        for (i, &flag) in self.flags.iter().enumerate() {
            if !b"dgimsuyv".iter().any(|&f| u16::from(f) == flag) {
                return self.raise("Invalid regular expression flag");
            }
            if self.flags[i + 1..].contains(&flag) {
                return self.raise("Duplicate regular expression flag");
            }
            if flag == u16::from(b'u') {
                u = true;
            }
            if flag == u16::from(b'v') {
                v = true;
            }
        }
        if u && v {
            return self.raise("Invalid regular expression flag");
        }
        Ok(())
    }

    fn new_branch(&mut self, parent: Option<usize>, base: Option<usize>) -> usize {
        let id = self.branches.len();
        self.branches.push(Branch {
            parent,
            base: base.unwrap_or(id),
        });
        id
    }

    fn separated_from(&self, this: usize, alt: usize) -> bool {
        let mut s = Some(this);
        while let Some(self_id) = s {
            let mut o = Some(alt);
            while let Some(other) = o {
                if self.branches[self_id].base == self.branches[other].base && self_id != other {
                    return true;
                }
                o = self.branches[other].parent;
            }
            s = self.branches[self_id].parent;
        }
        false
    }

    fn pattern(&mut self) -> RR<()> {
        self.pos = 0;
        self.last_int_value = 0.0;
        self.last_string_value.clear();
        self.last_assertion_is_quantifiable = false;
        self.num_capturing_parens = 0.0;
        self.max_back_reference = 0.0;
        self.group_names.clear();
        self.back_reference_names.clear();
        self.branch_id = None;

        self.disjunction()?;

        if self.pos != self.source.len() {
            if self.eat(0x29) {
                return self.raise("Unmatched ')'");
            }
            if self.eat(0x5D) || self.eat(0x7D) {
                return self.raise("Lone quantifier brackets");
            }
            return self.raise("Unexpected character");
        }
        if self.num_capturing_parens > MAX_CAPTURES {
            return self.raise("Too many captures");
        }
        if self.max_back_reference > self.num_capturing_parens {
            return self.raise("Invalid escape");
        }
        for name in &self.back_reference_names {
            if !self.group_names.contains_key(name) {
                return self.raise("Invalid named capture referenced");
            }
        }
        Ok(())
    }

    fn disjunction(&mut self) -> RR<()> {
        self.depth += 1;
        if self.depth > MAX_REGEX_DEPTH {
            return Err(RegexFail::TooDeep);
        }
        let id = self.new_branch(self.branch_id, None);
        self.branch_id = Some(id);
        self.alternative()?;
        while self.eat(0x7C) {
            let cur = self.branches[self.branch_id.expect("inside a disjunction")];
            let id = self.new_branch(cur.parent, Some(cur.base));
            self.branch_id = Some(id);
            self.alternative()?;
        }
        self.branch_id = self.branches[self.branch_id.expect("inside a disjunction")].parent;
        if self.eat_quantifier(true)? {
            return self.raise("Nothing to repeat");
        }
        if self.eat(0x7B) {
            return self.raise("Lone quantifier brackets");
        }
        self.depth -= 1;
        Ok(())
    }

    fn alternative(&mut self) -> RR<()> {
        while self.pos < self.source.len() && self.eat_term()? {}
        Ok(())
    }

    fn eat_term(&mut self) -> RR<bool> {
        if self.eat_assertion()? {
            if self.last_assertion_is_quantifiable && self.eat_quantifier(false)? && self.switch_u {
                return self.raise("Invalid quantifier");
            }
            return Ok(true);
        }
        let atom = if self.switch_u {
            self.eat_atom()?
        } else {
            self.eat_extended_atom()?
        };
        if atom {
            self.eat_quantifier(false)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn eat_assertion(&mut self) -> RR<bool> {
        let start = self.pos;
        self.last_assertion_is_quantifiable = false;
        if self.eat(0x5E) || self.eat(0x24) {
            return Ok(true);
        }
        if self.eat(0x5C) {
            if self.eat(0x42) || self.eat(0x62) {
                return Ok(true);
            }
            self.pos = start;
        }
        if self.eat(0x28) && self.eat(0x3F) {
            let lookbehind = self.eat(0x3C);
            if self.eat(0x3D) || self.eat(0x21) {
                self.disjunction()?;
                if !self.eat(0x29) {
                    return self.raise("Unterminated group");
                }
                self.last_assertion_is_quantifiable = !lookbehind;
                return Ok(true);
            }
        }
        self.pos = start;
        Ok(false)
    }

    fn eat_quantifier(&mut self, no_error: bool) -> RR<bool> {
        if self.eat_quantifier_prefix(no_error)? {
            self.eat(0x3F);
            return Ok(true);
        }
        Ok(false)
    }

    fn eat_quantifier_prefix(&mut self, no_error: bool) -> RR<bool> {
        if self.eat(0x2A) || self.eat(0x2B) || self.eat(0x3F) {
            return Ok(true);
        }
        self.eat_braced_quantifier(no_error)
    }

    fn eat_braced_quantifier(&mut self, no_error: bool) -> RR<bool> {
        let start = self.pos;
        if self.eat(0x7B) {
            if self.eat_decimal_digits() {
                let min = self.last_int_value;
                let mut max = -1.0;
                if self.eat(0x2C) && self.eat_decimal_digits() {
                    max = self.last_int_value;
                }
                if self.eat(0x7D) {
                    if max != -1.0 && max < min && !no_error {
                        return self.raise("numbers out of order in {} quantifier");
                    }
                    return Ok(true);
                }
            }
            if self.switch_u && !no_error {
                return self.raise("Incomplete quantifier");
            }
            self.pos = start;
        }
        Ok(false)
    }

    fn eat_atom(&mut self) -> RR<bool> {
        Ok(self.eat_pattern_characters()
            || self.eat(0x2E)
            || self.eat_reverse_solidus_atom_escape()?
            || self.eat_character_class()?
            || self.eat_uncapturing_group()?
            || self.eat_capturing_group()?)
    }

    fn eat_reverse_solidus_atom_escape(&mut self) -> RR<bool> {
        let start = self.pos;
        if self.eat(0x5C) {
            if self.eat_atom_escape()? {
                return Ok(true);
            }
            self.pos = start;
        }
        Ok(false)
    }

    fn eat_modifiers(&mut self) -> Vec<i64> {
        let mut modifiers = Vec::new();
        loop {
            let ch = self.current();
            if ch == -1 || !is_regular_expression_modifier(ch) {
                break;
            }
            modifiers.push(ch);
            self.advance();
        }
        modifiers
    }

    fn eat_uncapturing_group(&mut self) -> RR<bool> {
        let start = self.pos;
        if self.eat(0x28) {
            if self.eat(0x3F) {
                let add_modifiers = self.eat_modifiers();
                let has_hyphen = self.eat(0x2D);
                if !add_modifiers.is_empty() || has_hyphen {
                    for (i, m) in add_modifiers.iter().enumerate() {
                        if add_modifiers[i + 1..].contains(m) {
                            return self.raise("Duplicate regular expression modifiers");
                        }
                    }
                    if has_hyphen {
                        let remove_modifiers = self.eat_modifiers();
                        if add_modifiers.is_empty()
                            && remove_modifiers.is_empty()
                            && self.current() == 0x3A
                        {
                            return self.raise("Invalid regular expression modifiers");
                        }
                        for (i, m) in remove_modifiers.iter().enumerate() {
                            if remove_modifiers[i + 1..].contains(m) || add_modifiers.contains(m) {
                                return self.raise("Duplicate regular expression modifiers");
                            }
                        }
                    }
                }
                if self.eat(0x3A) {
                    self.disjunction()?;
                    if self.eat(0x29) {
                        return Ok(true);
                    }
                    return self.raise("Unterminated group");
                }
            }
            self.pos = start;
        }
        Ok(false)
    }

    fn eat_capturing_group(&mut self) -> RR<bool> {
        if self.eat(0x28) {
            self.group_specifier()?;
            self.disjunction()?;
            if self.eat(0x29) {
                self.num_capturing_parens += 1.0;
                return Ok(true);
            }
            return self.raise("Unterminated group");
        }
        Ok(false)
    }

    fn eat_extended_atom(&mut self) -> RR<bool> {
        Ok(self.eat(0x2E)
            || self.eat_reverse_solidus_atom_escape()?
            || self.eat_character_class()?
            || self.eat_uncapturing_group()?
            || self.eat_capturing_group()?
            || self.eat_invalid_braced_quantifier()?
            || self.eat_extended_pattern_character())
    }

    fn eat_invalid_braced_quantifier(&mut self) -> RR<bool> {
        if self.eat_braced_quantifier(true)? {
            return self.raise("Nothing to repeat");
        }
        Ok(false)
    }

    fn eat_syntax_character(&mut self) -> bool {
        let ch = self.current();
        if is_syntax_character(ch) {
            self.last_int_value = ch as f64;
            self.advance();
            return true;
        }
        false
    }

    fn eat_pattern_characters(&mut self) -> bool {
        let start = self.pos;
        loop {
            let ch = self.current();
            if ch == -1 || is_syntax_character(ch) {
                break;
            }
            self.advance();
        }
        self.pos != start
    }

    fn eat_extended_pattern_character(&mut self) -> bool {
        let ch = self.current();
        if ch != -1
            && ch != 0x24
            && !(0x28..=0x2B).contains(&ch)
            && ch != 0x2E
            && ch != 0x3F
            && ch != 0x5B
            && ch != 0x5E
            && ch != 0x7C
        {
            self.advance();
            return true;
        }
        false
    }

    fn group_specifier(&mut self) -> RR<()> {
        if self.eat(0x3F) {
            if !self.eat_group_name()? {
                return self.raise("Invalid group");
            }
            let name = self.last_string_value.clone();
            let branch = self.branch_id.expect("inside a disjunction");
            if let Some(known) = self.group_names.get(&name) {
                for &alt in known {
                    if !self.separated_from(alt, branch) {
                        return self.raise("Duplicate capture group name");
                    }
                }
                self.group_names
                    .get_mut(&name)
                    .expect("known name")
                    .push(branch);
            } else {
                self.group_names.insert(name, vec![branch]);
            }
        }
        Ok(())
    }

    fn eat_group_name(&mut self) -> RR<bool> {
        self.last_string_value.clear();
        if self.eat(0x3C) {
            if self.eat_regexp_identifier_name()? && self.eat(0x3E) {
                return Ok(true);
            }
            return self.raise("Invalid capture group name");
        }
        Ok(false)
    }

    fn eat_regexp_identifier_name(&mut self) -> RR<bool> {
        self.last_string_value.clear();
        if self.eat_regexp_identifier_start()? {
            let mut name = String::new();
            push_code_point(&mut name, self.last_int_value as i64);
            while self.eat_regexp_identifier_part()? {
                push_code_point(&mut name, self.last_int_value as i64);
            }
            self.last_string_value = name;
            return Ok(true);
        }
        Ok(false)
    }

    fn eat_regexp_identifier_start(&mut self) -> RR<bool> {
        let start = self.pos;
        let mut ch = self.current_u(true);
        self.advance_u(true);
        if ch == 0x5C && self.eat_regexp_unicode_escape_sequence(true)? {
            ch = self.last_int_value as i64;
        }
        if is_regexp_identifier_start(ch) {
            self.last_int_value = ch as f64;
            return Ok(true);
        }
        self.pos = start;
        Ok(false)
    }

    fn eat_regexp_identifier_part(&mut self) -> RR<bool> {
        let start = self.pos;
        let mut ch = self.current_u(true);
        self.advance_u(true);
        if ch == 0x5C && self.eat_regexp_unicode_escape_sequence(true)? {
            ch = self.last_int_value as i64;
        }
        if is_regexp_identifier_part(ch) {
            self.last_int_value = ch as f64;
            return Ok(true);
        }
        self.pos = start;
        Ok(false)
    }

    fn eat_atom_escape(&mut self) -> RR<bool> {
        if self.eat_back_reference()
            || self.eat_character_class_escape()? != CharSet::None
            || self.eat_character_escape()?
            || (self.switch_n && self.eat_k_group_name()?)
        {
            return Ok(true);
        }
        if self.switch_u {
            if self.current() == 0x63 {
                return self.raise("Invalid unicode escape");
            }
            return self.raise("Invalid escape");
        }
        Ok(false)
    }

    fn eat_back_reference(&mut self) -> bool {
        let start = self.pos;
        if self.eat_decimal_escape() {
            let n = self.last_int_value;
            if self.switch_u {
                if n > self.max_back_reference {
                    self.max_back_reference = n;
                }
                return true;
            }
            if n <= self.num_capturing_parens {
                return true;
            }
            self.pos = start;
        }
        false
    }

    fn eat_k_group_name(&mut self) -> RR<bool> {
        if self.eat(0x6B) {
            if self.eat_group_name()? {
                self.back_reference_names
                    .push(self.last_string_value.clone());
                return Ok(true);
            }
            return self.raise("Invalid named reference");
        }
        Ok(false)
    }

    fn eat_character_escape(&mut self) -> RR<bool> {
        Ok(self.eat_control_escape()
            || self.eat_c_control_letter()
            || self.eat_zero()
            || self.eat_hex_escape_sequence()?
            || self.eat_regexp_unicode_escape_sequence(false)?
            || (!self.switch_u && self.eat_legacy_octal_escape_sequence())
            || self.eat_identity_escape())
    }

    fn eat_c_control_letter(&mut self) -> bool {
        let start = self.pos;
        if self.eat(0x63) {
            if self.eat_control_letter() {
                return true;
            }
            self.pos = start;
        }
        false
    }

    fn eat_zero(&mut self) -> bool {
        if self.current() == 0x30 && !is_decimal_digit(self.lookahead()) {
            self.last_int_value = 0.0;
            self.advance();
            return true;
        }
        false
    }

    fn eat_control_escape(&mut self) -> bool {
        let value = match self.current() {
            0x74 => 0x09,
            0x6E => 0x0A,
            0x76 => 0x0B,
            0x66 => 0x0C,
            0x72 => 0x0D,
            _ => return false,
        };
        self.last_int_value = f64::from(value);
        self.advance();
        true
    }

    fn eat_control_letter(&mut self) -> bool {
        let ch = self.current();
        if is_control_letter(ch) {
            self.last_int_value = (ch % 0x20) as f64;
            self.advance();
            return true;
        }
        false
    }

    fn eat_regexp_unicode_escape_sequence(&mut self, force_u: bool) -> RR<bool> {
        let start = self.pos;
        let switch_u = force_u || self.switch_u;
        if self.eat(0x75) {
            if self.eat_fixed_hex_digits(4) {
                let lead = self.last_int_value;
                if switch_u && (0xD800 as f64..=0xDBFF as f64).contains(&lead) {
                    let lead_surrogate_end = self.pos;
                    if self.eat(0x5C) && self.eat(0x75) && self.eat_fixed_hex_digits(4) {
                        let trail = self.last_int_value;
                        if (0xDC00 as f64..=0xDFFF as f64).contains(&trail) {
                            self.last_int_value =
                                (lead - 0xD800 as f64) * 1024.0 + (trail - 0xDC00 as f64) + 65536.0;
                            return Ok(true);
                        }
                    }
                    self.pos = lead_surrogate_end;
                    self.last_int_value = lead;
                }
                return Ok(true);
            }
            if switch_u
                && self.eat(0x7B)
                && self.eat_hex_digits()
                && self.eat(0x7D)
                && self.last_int_value >= 0.0
                && self.last_int_value <= 1_114_111.0
            {
                return Ok(true);
            }
            if switch_u {
                return self.raise("Invalid unicode escape");
            }
            self.pos = start;
        }
        Ok(false)
    }

    fn eat_identity_escape(&mut self) -> bool {
        if self.switch_u {
            if self.eat_syntax_character() {
                return true;
            }
            if self.eat(0x2F) {
                self.last_int_value = 47.0;
                return true;
            }
            return false;
        }
        let ch = self.current();
        if ch != 0x63 && (!self.switch_n || ch != 0x6B) {
            self.last_int_value = ch as f64;
            self.advance();
            return true;
        }
        false
    }

    fn eat_decimal_escape(&mut self) -> bool {
        self.last_int_value = 0.0;
        let mut ch = self.current();
        if (0x31..=0x39).contains(&ch) {
            loop {
                self.last_int_value = 10.0 * self.last_int_value + (ch - 0x30) as f64;
                self.advance();
                ch = self.current();
                if !(0x30..=0x39).contains(&ch) {
                    break;
                }
            }
            return true;
        }
        false
    }

    fn eat_character_class_escape(&mut self) -> RR<CharSet> {
        let ch = self.current();
        if is_character_class_escape(ch) {
            self.last_int_value = -1.0;
            self.advance();
            return Ok(CharSet::Ok);
        }
        let negate = ch == 0x50;
        if self.switch_u && (negate || ch == 0x70) {
            self.last_int_value = -1.0;
            self.advance();
            if self.eat(0x7B) {
                let result = self.eat_unicode_property_value_expression()?;
                if result != CharSet::None && self.eat(0x7D) {
                    if negate && result == CharSet::String {
                        return self.raise("Invalid property name");
                    }
                    return Ok(result);
                }
            }
            return self.raise("Invalid property name");
        }
        Ok(CharSet::None)
    }

    fn eat_unicode_property_value_expression(&mut self) -> RR<CharSet> {
        let start = self.pos;
        if self.eat_unicode_property_name() && self.eat(0x3D) {
            let name = self.last_string_value.clone();
            if self.eat_unicode_property_value() {
                let value = self.last_string_value.clone();
                self.validate_unicode_property_name_and_value(&name, &value)?;
                return Ok(CharSet::Ok);
            }
        }
        self.pos = start;
        if self.eat_unicode_property_value() {
            let name_or_value = self.last_string_value.clone();
            return self.validate_unicode_property_name_or_value(&name_or_value);
        }
        Ok(CharSet::None)
    }

    fn validate_unicode_property_name_and_value(&self, name: &str, value: &str) -> RR<()> {
        let d = property_data();
        let values = match name {
            "General_Category" | "gc" => &d.general_category,
            "Script" | "sc" | "Script_Extensions" | "scx" => &d.script,
            _ => return self.raise("Invalid property name"),
        };
        if !values.contains(value) {
            return self.raise("Invalid property value");
        }
        Ok(())
    }

    fn validate_unicode_property_name_or_value(&self, name_or_value: &str) -> RR<CharSet> {
        let d = property_data();
        if d.binary.contains(name_or_value) {
            return Ok(CharSet::Ok);
        }
        if self.switch_v && d.binary_of_strings.contains(name_or_value) {
            return Ok(CharSet::String);
        }
        self.raise("Invalid property name")
    }

    fn eat_unicode_property_name(&mut self) -> bool {
        self.last_string_value.clear();
        loop {
            let ch = self.current();
            if !is_unicode_property_name_character(ch) {
                break;
            }
            push_code_point(&mut self.last_string_value, ch);
            self.advance();
        }
        !self.last_string_value.is_empty()
    }

    fn eat_unicode_property_value(&mut self) -> bool {
        self.last_string_value.clear();
        loop {
            let ch = self.current();
            if !is_unicode_property_value_character(ch) {
                break;
            }
            push_code_point(&mut self.last_string_value, ch);
            self.advance();
        }
        !self.last_string_value.is_empty()
    }

    fn eat_character_class(&mut self) -> RR<bool> {
        if self.eat(0x5B) {
            let negate = self.eat(0x5E);
            let result = self.class_contents()?;
            if !self.eat(0x5D) {
                return self.raise("Unterminated character class");
            }
            if negate && result == CharSet::String {
                return self.raise("Negated character class may contain strings");
            }
            return Ok(true);
        }
        Ok(false)
    }

    fn class_contents(&mut self) -> RR<CharSet> {
        if self.current() == 0x5D {
            return Ok(CharSet::Ok);
        }
        if self.switch_v {
            return self.class_set_expression();
        }
        self.non_empty_class_ranges()?;
        Ok(CharSet::Ok)
    }

    fn non_empty_class_ranges(&mut self) -> RR<()> {
        while self.eat_class_atom()? {
            let left = self.last_int_value;
            if self.eat(0x2D) && self.eat_class_atom()? {
                let right = self.last_int_value;
                if self.switch_u && (left == -1.0 || right == -1.0) {
                    return self.raise("Invalid character class");
                }
                if left != -1.0 && right != -1.0 && left > right {
                    return self.raise("Range out of order in character class");
                }
            }
        }
        Ok(())
    }

    fn eat_class_atom(&mut self) -> RR<bool> {
        let start = self.pos;
        if self.eat(0x5C) {
            if self.eat_class_escape()? {
                return Ok(true);
            }
            if self.switch_u {
                let ch = self.current();
                if ch == 0x63 || is_octal_digit(ch) {
                    return self.raise("Invalid class escape");
                }
                return self.raise("Invalid escape");
            }
            self.pos = start;
        }
        let ch = self.current();
        if ch != 0x5D {
            self.last_int_value = ch as f64;
            self.advance();
            return Ok(true);
        }
        Ok(false)
    }

    fn eat_class_escape(&mut self) -> RR<bool> {
        let start = self.pos;
        if self.eat(0x62) {
            self.last_int_value = 8.0;
            return Ok(true);
        }
        if self.switch_u && self.eat(0x2D) {
            self.last_int_value = 45.0;
            return Ok(true);
        }
        if !self.switch_u && self.eat(0x63) {
            if self.eat_class_control_letter() {
                return Ok(true);
            }
            self.pos = start;
        }
        Ok(self.eat_character_class_escape()? != CharSet::None || self.eat_character_escape()?)
    }

    fn class_set_expression(&mut self) -> RR<CharSet> {
        let mut result = CharSet::Ok;
        if self.eat_class_set_range()? {
        } else {
            let sub_result = self.eat_class_set_operand()?;
            if sub_result != CharSet::None {
                if sub_result == CharSet::String {
                    result = CharSet::String;
                }
                let start = self.pos;
                while self.eat_chars(&[0x26, 0x26]) {
                    if self.current() != 0x26 {
                        let sub = self.eat_class_set_operand()?;
                        if sub != CharSet::None {
                            if sub != CharSet::String {
                                result = CharSet::Ok;
                            }
                            continue;
                        }
                    }
                    return self.raise("Invalid character in character class");
                }
                if start != self.pos {
                    return Ok(result);
                }
                while self.eat_chars(&[0x2D, 0x2D]) {
                    if self.eat_class_set_operand()? != CharSet::None {
                        continue;
                    }
                    return self.raise("Invalid character in character class");
                }
                if start != self.pos {
                    return Ok(result);
                }
            } else {
                return self.raise("Invalid character in character class");
            }
        }
        loop {
            if self.eat_class_set_range()? {
                continue;
            }
            let sub_result = self.eat_class_set_operand()?;
            if sub_result == CharSet::None {
                return Ok(result);
            }
            if sub_result == CharSet::String {
                result = CharSet::String;
            }
        }
    }

    fn eat_class_set_range(&mut self) -> RR<bool> {
        let start = self.pos;
        if self.eat_class_set_character()? {
            let left = self.last_int_value;
            if self.eat(0x2D) && self.eat_class_set_character()? {
                let right = self.last_int_value;
                if left != -1.0 && right != -1.0 && left > right {
                    return self.raise("Range out of order in character class");
                }
                return Ok(true);
            }
            self.pos = start;
        }
        Ok(false)
    }

    fn eat_class_set_operand(&mut self) -> RR<CharSet> {
        if self.eat_class_set_character()? {
            return Ok(CharSet::Ok);
        }
        let disjunction_result = self.eat_class_string_disjunction()?;
        if disjunction_result != CharSet::None {
            return Ok(disjunction_result);
        }
        self.eat_nested_class()
    }

    fn eat_nested_class(&mut self) -> RR<CharSet> {
        let start = self.pos;
        if self.eat(0x5B) {
            let negate = self.eat(0x5E);
            self.depth += 1;
            if self.depth > MAX_REGEX_DEPTH {
                return Err(RegexFail::TooDeep);
            }
            let result = self.class_contents();
            self.depth -= 1;
            let result = result?;
            if self.eat(0x5D) {
                if negate && result == CharSet::String {
                    return self.raise("Negated character class may contain strings");
                }
                return Ok(result);
            }
            self.pos = start;
        }
        if self.eat(0x5C) {
            let result = self.eat_character_class_escape()?;
            if result != CharSet::None {
                return Ok(result);
            }
            self.pos = start;
        }
        Ok(CharSet::None)
    }

    fn eat_class_string_disjunction(&mut self) -> RR<CharSet> {
        let start = self.pos;
        if self.eat_chars(&[0x5C, 0x71]) {
            if self.eat(0x7B) {
                let result = self.class_string_disjunction_contents()?;
                if self.eat(0x7D) {
                    return Ok(result);
                }
            } else {
                return self.raise("Invalid escape");
            }
            self.pos = start;
        }
        Ok(CharSet::None)
    }

    fn class_string_disjunction_contents(&mut self) -> RR<CharSet> {
        let mut result = self.class_string()?;
        while self.eat(0x7C) {
            if self.class_string()? == CharSet::String {
                result = CharSet::String;
            }
        }
        Ok(result)
    }

    fn class_string(&mut self) -> RR<CharSet> {
        let mut count = 0;
        while self.eat_class_set_character()? {
            count += 1;
        }
        Ok(if count == 1 {
            CharSet::Ok
        } else {
            CharSet::String
        })
    }

    fn eat_class_set_character(&mut self) -> RR<bool> {
        let start = self.pos;
        if self.eat(0x5C) {
            if self.eat_character_escape()? || self.eat_class_set_reserved_punctuator() {
                return Ok(true);
            }
            if self.eat(0x62) {
                self.last_int_value = 8.0;
                return Ok(true);
            }
            self.pos = start;
            return Ok(false);
        }
        let ch = self.current();
        if ch < 0
            || (ch == self.lookahead() && is_class_set_reserved_double_punctuator_character(ch))
        {
            return Ok(false);
        }
        if is_class_set_syntax_character(ch) {
            return Ok(false);
        }
        self.advance();
        self.last_int_value = ch as f64;
        Ok(true)
    }

    fn eat_class_set_reserved_punctuator(&mut self) -> bool {
        let ch = self.current();
        if is_class_set_reserved_punctuator(ch) {
            self.last_int_value = ch as f64;
            self.advance();
            return true;
        }
        false
    }

    fn eat_class_control_letter(&mut self) -> bool {
        let ch = self.current();
        if is_decimal_digit(ch) || ch == 0x5F {
            self.last_int_value = (ch % 0x20) as f64;
            self.advance();
            return true;
        }
        false
    }

    fn eat_hex_escape_sequence(&mut self) -> RR<bool> {
        let start = self.pos;
        if self.eat(0x78) {
            if self.eat_fixed_hex_digits(2) {
                return Ok(true);
            }
            if self.switch_u {
                return self.raise("Invalid escape");
            }
            self.pos = start;
        }
        Ok(false)
    }

    fn eat_decimal_digits(&mut self) -> bool {
        let start = self.pos;
        self.last_int_value = 0.0;
        loop {
            let ch = self.current();
            if !is_decimal_digit(ch) {
                break;
            }
            self.last_int_value = 10.0 * self.last_int_value + (ch - 0x30) as f64;
            self.advance();
        }
        self.pos != start
    }

    fn eat_hex_digits(&mut self) -> bool {
        let start = self.pos;
        self.last_int_value = 0.0;
        loop {
            let ch = self.current();
            if !is_hex_digit(ch) {
                break;
            }
            self.last_int_value = 16.0 * self.last_int_value + hex_to_int(ch);
            self.advance();
        }
        self.pos != start
    }

    fn eat_legacy_octal_escape_sequence(&mut self) -> bool {
        if self.eat_octal_digit() {
            let n1 = self.last_int_value;
            if self.eat_octal_digit() {
                let n2 = self.last_int_value;
                if n1 <= 3.0 && self.eat_octal_digit() {
                    self.last_int_value += n1 * 64.0 + n2 * 8.0;
                } else {
                    self.last_int_value = n1 * 8.0 + n2;
                }
            } else {
                self.last_int_value = n1;
            }
            return true;
        }
        false
    }

    fn eat_octal_digit(&mut self) -> bool {
        let ch = self.current();
        if is_octal_digit(ch) {
            self.last_int_value = (ch - 0x30) as f64;
            self.advance();
            return true;
        }
        self.last_int_value = 0.0;
        false
    }

    fn eat_fixed_hex_digits(&mut self, length: usize) -> bool {
        let start = self.pos;
        self.last_int_value = 0.0;
        for _ in 0..length {
            let ch = self.current();
            if !is_hex_digit(ch) {
                self.pos = start;
                return false;
            }
            self.last_int_value = 16.0 * self.last_int_value + hex_to_int(ch);
            self.advance();
        }
        true
    }
}

/// Validates the flags and pattern of the literal `/pattern/flags` (UTF-16 code
/// units) as V8 does.
pub(crate) fn validate_regexp_literal(pattern: &[u16], flags: &[u16]) -> RegexCheck {
    let mut validator = RegexValidator::new(pattern, flags);
    match validator.validate() {
        Ok(()) => RegexCheck::Valid,
        Err(RegexFail::Error(message)) => RegexCheck::Invalid(message),
        Err(RegexFail::TooDeep) => RegexCheck::Undecidable(format!(
            "regular expression groups nested deeper than {MAX_REGEX_DEPTH}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(pattern: &str, flags: &str) -> bool {
        let p: Vec<u16> = pattern.encode_utf16().collect();
        let f: Vec<u16> = flags.encode_utf16().collect();
        validate_regexp_literal(&p, &f) == RegexCheck::Valid
    }

    #[test]
    fn literals() {
        assert!(check("a", "g"));
        assert!(!check("a", "gg"));
        assert!(!check("a**", ""));
        assert!(!check(r"\p{Foo}", "u"));
        assert!(check(r"\p{L}", "u"));
        assert!(!check("(?<a>x)(?<a>y)", ""));
        assert!(check("(?<a>x)|(?<a>y)", ""));
        assert!(!check("(?ii:a)", ""));
        assert!(check("(?i:a)", ""));
        assert!(!check("{1}", ""));
        assert!(!check("a{2,1}", ""));
        assert!(!check(r"[\d-z]", "u"));
        assert!(check("[a--b]", "v"));
        assert!(check(r"\1(a)", ""));
        assert!(check(r"\c", ""));
    }

    #[test]
    fn capture_limit() {
        assert!(check(&"()".repeat(32767), ""));
        assert!(!check(&"()".repeat(32768), ""));
        assert!(!check(&"(?:())".repeat(32768), "u"));
    }
}
