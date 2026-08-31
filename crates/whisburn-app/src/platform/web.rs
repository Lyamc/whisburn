use std::cell::RefCell;

use gloo_storage::{LocalStorage, Storage};
use js_sys::Uint8Array;
use whisburn_core::Settings;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{FileReader, HtmlInputElement, ProgressEvent};

use super::AudioFile;

const SETTINGS_KEY: &str = "whisburn_settings";

thread_local! {
    static PICK_TX: RefCell<Option<futures_channel::oneshot::Sender<Option<AudioFile>>>> =
        RefCell::new(None);
    static INPUT_READY: RefCell<bool> = RefCell::new(false);
}

pub async fn pick_audio() -> Option<AudioFile> {
    ensure_file_input()?;

    let (tx, rx) = futures_channel::oneshot::channel();
    PICK_TX.with(|slot| *slot.borrow_mut() = Some(tx));

    let input = file_input()?;
    input.set_value("");
    input.click();

    rx.await.ok().flatten()
}

fn ensure_file_input() -> Option<()> {
    let ready = INPUT_READY.with(|r| *r.borrow());
    if ready {
        return Some(());
    }

    let input = file_input()?;
    let onchange = Closure::wrap(Box::new(move |_event: web_sys::Event| {
        wasm_bindgen_futures::spawn_local(async {
            let picked = read_selected_file().await;
            PICK_TX.with(|slot| {
                if let Some(tx) = slot.borrow_mut().take() {
                    let _ = tx.send(picked);
                }
            });
        });
    }) as Box<dyn FnMut(web_sys::Event)>);

    input.set_onchange(Some(onchange.as_ref().unchecked_ref()));
    onchange.forget();
    INPUT_READY.with(|r| *r.borrow_mut() = true);
    Some(())
}

async fn read_selected_file() -> Option<AudioFile> {
    let input = file_input()?;
    let files = input.files()?;
    if files.length() == 0 {
        return None;
    }
    let file = files.get(0)?;
    let name = file.name();
    let reader = FileReader::new().ok()?;

    let (tx, rx) = futures_channel::oneshot::channel::<()>();
    let reader_for_cb = reader.clone();
    let tx_slot = std::rc::Rc::new(std::cell::RefCell::new(Some(tx)));
    let tx_for_cb = tx_slot.clone();
    let onloadend = Closure::wrap(Box::new(move |_event: ProgressEvent| {
        if let Some(sender) = tx_for_cb.borrow_mut().take() {
            let _ = sender.send(());
        }
    }) as Box<dyn FnMut(ProgressEvent)>);
    reader.set_onloadend(Some(onloadend.as_ref().unchecked_ref()));
    onloadend.forget();

    reader.read_as_array_buffer(&file).ok()?;
    rx.await.ok()?;

    let buffer = reader_for_cb.result().ok()?;
    let array = Uint8Array::new(&buffer);
    Some(AudioFile {
        name,
        bytes: array.to_vec(),
    })
}

fn file_input() -> Option<HtmlInputElement> {
    web_sys::window()?
        .document()?
        .get_element_by_id("audio-file")?
        .dyn_into::<HtmlInputElement>()
        .ok()
}

pub fn load_app_settings() -> Settings {
    LocalStorage::get(SETTINGS_KEY).unwrap_or_else(|_| default_web_settings())
}

pub fn save_app_settings(settings: &Settings) -> Result<std::path::PathBuf, String> {
    LocalStorage::set(SETTINGS_KEY, settings).map_err(|e| format!("localStorage error: {e:?}"))?;
    Ok(std::path::PathBuf::from("browser localStorage"))
}

pub fn settings_hint() -> &'static str {
    "Saved in browser localStorage. With Trunk, leave Server URL as this page origin; API calls proxy to port 8787."
}

fn default_web_settings() -> Settings {
    Settings {
        server_url: default_api_base(),
        ..Settings::default()
    }
}

fn default_api_base() -> String {
    if let Some(window) = web_sys::window() {
        let origin = window.location().origin().unwrap_or_default();
        if !origin.is_empty() && origin != "null" {
            return origin;
        }
    }
    "http://127.0.0.1:8787".to_string()
}