use iced::time;
use iced::widget::{
    button, column, container, pick_list, row, scrollable, text, text_editor, text_input, Space,
};
use iced::{border::Radius, Alignment, Border, Color, Element, Length, Padding, Subscription, Task};

use whisburn_core::{PreloadModels, Settings};

use crate::api::{self, ModelInfo};
use crate::platform::{self, AudioFile};

const FORMATS: &[&str] = &["json", "txt", "srt", "vtt"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Screen {
    Transcribe,
    Settings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ServerStatus {
    Error,
    Warning,
    Processing,
    Ready,
    Connected,
}

impl ServerStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Error => "Server error",
            Self::Warning => "Warning",
            Self::Processing => "Processing",
            Self::Connected => "Connected",
            Self::Ready => "Models ready",
        }
    }

    fn color(self, blink_on: bool) -> Color {
        match self {
            Self::Error => Color::from_rgb(0.92, 0.22, 0.22),
            Self::Warning => Color::from_rgb(0.95, 0.78, 0.15),
            Self::Processing if blink_on => Color::from_rgb(0.18, 0.82, 0.32),
            Self::Processing => Color::from_rgb(0.18, 0.82, 0.32).scale_alpha(0.25),
            Self::Ready => Color::from_rgb(0.18, 0.82, 0.32),
            Self::Connected => Color::from_rgb(0.28, 0.52, 0.95),
        }
    }
}

pub struct App {
    screen: Screen,
    settings: Settings,
    settings_draft: SettingsDraft,
    models: Vec<ModelInfo>,
    selected_model: Option<String>,
    selected_format: String,
    language: String,
    audio: Option<AudioFile>,
    status_line: String,
    transcript: String,
    output: text_editor::Content,
    server_status: ServerStatus,
    server_connected: bool,
    server_warning: bool,
    processing: bool,
    blink_on: bool,
}

#[derive(Debug, Clone)]
struct SettingsDraft {
    server_url: String,
    default_model: String,
    preload_models: String,
}

impl From<&Settings> for SettingsDraft {
    fn from(s: &Settings) -> Self {
        Self {
            server_url: s.server_url.clone(),
            default_model: s.default_model.clone(),
            preload_models: preload_to_text(&s.preload_models),
        }
    }
}

#[derive(Debug, Clone)]
struct AppInit {
    settings: Settings,
    models: Vec<ModelInfo>,
    default_model: String,
    server_warning: bool,
    models_loaded: bool,
    status_line: String,
}

impl App {
    pub fn title(&self) -> String {
        "whisburn".to_string()
    }

    pub fn new() -> (Self, Task<Message>) {
        let settings = platform::load_app_settings();
        let server_url = settings.server_url.clone();
        (
            Self {
                screen: Screen::Transcribe,
                settings_draft: SettingsDraft::from(&settings),
                selected_format: "txt".to_string(),
                language: "en".to_string(),
                models: Vec::new(),
                selected_model: Some(settings.default_model.clone()),
                audio: None,
                status_line: "Connecting to server…".to_string(),
                transcript: String::new(),
                output: text_editor::Content::new(),
                server_status: ServerStatus::Connected,
                server_connected: false,
                server_warning: false,
                processing: false,
                blink_on: true,
                settings,
            },
            Task::perform(refresh_catalog(server_url), Message::CatalogLoaded),
        )
    }

    pub fn subscription(&self) -> Subscription<Message> {
        if self.server_status == ServerStatus::Processing {
            time::every(time::Duration::from_secs(1)).map(|_| Message::BlinkTick)
        } else {
            Subscription::none()
        }
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::BlinkTick => {
                self.blink_on = !self.blink_on;
                Task::none()
            }
            Message::OpenSettings => {
                self.screen = Screen::Settings;
                self.settings_draft = SettingsDraft::from(&self.settings);
                Task::none()
            }
            Message::CloseSettings => {
                self.screen = Screen::Transcribe;
                Task::none()
            }
            Message::ServerUrlChanged(v) => {
                self.settings_draft.server_url = v;
                Task::none()
            }
            Message::DefaultModelChanged(v) => {
                self.settings_draft.default_model = v;
                Task::none()
            }
            Message::PreloadChanged(v) => {
                self.settings_draft.preload_models = v;
                Task::none()
            }
            Message::SaveSettings => {
                let mut next = self.settings.clone();
                next.server_url = self.settings_draft.server_url.trim().to_string();
                next.default_model = self.settings_draft.default_model.trim().to_string();
                next.preload_models = parse_preload_text(&self.settings_draft.preload_models);
                match platform::save_app_settings(&next) {
                    Ok(path) => {
                        self.settings = next.clone();
                        self.selected_model = Some(next.default_model.clone());
                        self.set_status(format!("Settings saved to {}", path.display()));
                        self.screen = Screen::Transcribe;
                        Task::perform(
                            refresh_catalog(self.settings.server_url.clone()),
                            Message::CatalogLoaded,
                        )
                    }
                    Err(e) => {
                        self.set_status(format!("Failed to save settings: {e}"));
                        Task::none()
                    }
                }
            }
            Message::PickAudio => {
                Task::perform(platform::pick_audio(), Message::AudioPicked)
            }
            Message::AudioPicked(file) => {
                if let Some(audio) = file {
                    self.set_status(format!("Selected {}", audio.name));
                    self.audio = Some(audio);
                }
                Task::none()
            }
            Message::ModelSelected(name) => {
                self.selected_model = Some(name);
                Task::none()
            }
            Message::FormatSelected(fmt) => {
                self.selected_format = fmt;
                Task::none()
            }
            Message::LanguageChanged(v) => {
                self.language = v;
                Task::none()
            }
            Message::OutputAction(action) => {
                if let text_editor::Action::Edit(_) = action {
                    Task::none()
                } else {
                    self.output.perform(action);
                    Task::none()
                }
            }
            Message::Refresh => Task::perform(
                refresh_catalog(self.settings.server_url.clone()),
                Message::CatalogLoaded,
            ),
            Message::CatalogLoaded(result) => match result {
                Ok(init) => {
                    self.settings = init.settings;
                    self.models = init.models;
                    self.server_connected = true;
                    self.server_warning = init.server_warning;
                    self.processing = false;
                    if self.selected_model.is_none()
                        || !self
                            .models
                            .iter()
                            .any(|m| Some(&m.name) == self.selected_model.as_ref())
                    {
                        self.selected_model = Some(init.default_model);
                    }
                    self.recompute_server_status(init.models_loaded);
                    self.set_status(init.status_line);
                    Task::none()
                }
                Err(e) => {
                    self.server_connected = false;
                    self.server_warning = false;
                    self.processing = false;
                    self.recompute_server_status(false);
                    self.set_status(format!("Server error: {e}"));
                    Task::none()
                }
            },
            Message::Transcribe => {
                let Some(audio) = self.audio.clone() else {
                    self.set_status("Choose an audio file first.".to_string());
                    return Task::none();
                };
                let Some(model) = self.selected_model.clone() else {
                    self.set_status("Choose a model.".to_string());
                    return Task::none();
                };
                self.processing = true;
                self.blink_on = true;
                self.recompute_server_status(self.models.iter().any(|m| m.loaded));
                self.set_status(format!("Transcribing with {model}…"));
                self.transcript.clear();
                let base = self.settings.server_url.clone();
                let format = self.selected_format.clone();
                let language = self.language.clone();
                Task::perform(
                    async move {
                        api::transcribe_bytes(
                            &base,
                            &audio.name,
                            audio.bytes,
                            &model,
                            &format,
                            &language,
                        )
                        .await
                        .map_err(|e| e.to_string())
                    },
                    Message::Transcribed,
                )
            }
            Message::Transcribed(result) => {
                self.processing = false;
                match result {
                    Ok(bytes) => {
                        self.transcript = String::from_utf8_lossy(&bytes).into_owned();
                        self.set_status("Done.".to_string());
                        Task::perform(
                            refresh_catalog(self.settings.server_url.clone()),
                            Message::CatalogLoaded,
                        )
                    }
                    Err(e) => {
                        self.recompute_server_status(self.models.iter().any(|m| m.loaded));
                        self.set_status(format!("Transcription failed: {e}"));
                        Task::none()
                    }
                }
            }
        }
    }

    pub fn view(&self) -> Element<'_, Message> {
        match self.screen {
            Screen::Transcribe => self.view_transcribe(),
            Screen::Settings => self.view_settings(),
        }
    }

    fn recompute_server_status(&mut self, models_loaded: bool) {
        let client_error = self.status_line.starts_with("Transcription failed")
            || self.status_line.starts_with("Failed to save");
        self.server_status = if !self.server_connected || self.status_line.starts_with("Server error") {
            ServerStatus::Error
        } else if self.processing {
            ServerStatus::Processing
        } else if client_error {
            ServerStatus::Error
        } else if self.server_warning {
            ServerStatus::Warning
        } else if models_loaded {
            ServerStatus::Ready
        } else {
            ServerStatus::Connected
        };
    }

    fn set_status(&mut self, line: String) {
        self.status_line = line;
        self.sync_output();
        let models_loaded = self.models.iter().any(|m| m.loaded);
        self.recompute_server_status(models_loaded);
    }

    fn sync_output(&mut self) {
        let combined = if self.transcript.is_empty() {
            self.status_line.clone()
        } else {
            format!("{}\n\n{}", self.status_line, self.transcript)
        };
        self.output = text_editor::Content::with_text(&combined);
    }

    fn status_indicator(&self) -> Element<'_, Message> {
        let color = self.server_status.color(self.blink_on);
        let dot = container(Space::new(Length::Fixed(14.0), Length::Fixed(14.0)))
            .style(move |_theme| container::Style {
                background: Some(color.into()),
                border: Border {
                    radius: Radius::from(7.0),
                    ..Default::default()
                },
                ..Default::default()
            });
        row![
            dot,
            text(self.server_status.label()).size(12),
        ]
        .spacing(8)
        .align_y(Alignment::Center)
        .into()
    }

    fn view_transcribe(&self) -> Element<'_, Message> {
        let ready_models: Vec<String> = self
            .models
            .iter()
            .filter(|m| m.burn_ready)
            .map(|m| m.name.clone())
            .collect();

        let model_pick: Element<'_, Message> = if ready_models.is_empty() {
            text_input("tiny_en", self.selected_model.as_deref().unwrap_or("tiny_en"))
                .on_input(Message::ModelSelected)
                .width(Length::Fill)
                .into()
        } else {
            pick_list(
                ready_models,
                self.selected_model.clone(),
                Message::ModelSelected,
            )
            .placeholder("Select model")
            .width(Length::Fill)
            .into()
        };

        let audio_label = self
            .audio
            .as_ref()
            .map(|a| a.name.as_str())
            .unwrap_or("No file selected");

        column![
            row![
                text("whisburn").size(26),
                Space::with_width(Length::Fill),
                self.status_indicator(),
                button("Settings").on_press(Message::OpenSettings),
                button("Refresh").on_press(Message::Refresh),
            ]
            .align_y(Alignment::Center),
            text("Transcribe audio via the API server. Models load on first request unless preloaded.")
                .size(13),
            container(
                column![
                    text("Audio file").size(14),
                    row![
                        button("Choose file…").on_press(Message::PickAudio),
                        text(audio_label).size(14),
                    ]
                    .spacing(12)
                    .align_y(Alignment::Center),
                    text("Model").size(14),
                    model_pick,
                    row![
                        column![
                            text("Output format").size(14),
                            pick_list(
                                FORMATS.iter().copied().map(str::to_string).collect::<Vec<_>>(),
                                Some(self.selected_format.clone()),
                                Message::FormatSelected,
                            )
                            .width(Length::FillPortion(1)),
                        ]
                        .width(Length::FillPortion(1)),
                        column![
                            text("Language").size(14),
                            text_input("en", &self.language)
                                .on_input(Message::LanguageChanged)
                                .width(Length::FillPortion(1)),
                        ]
                        .width(Length::FillPortion(1)),
                    ]
                    .spacing(16),
                    button("Transcribe")
                        .width(Length::Fill)
                        .padding(12)
                        .on_press(Message::Transcribe),
                ]
                .spacing(10)
                .padding(16)
            )
            .style(container::rounded_box)
            .width(Length::Fill),
            text("Result (select text to copy)").size(14),
            scrollable(
                text_editor(&self.output)
                    .on_action(Message::OutputAction)
                    .height(Length::Fill)
            )
            .height(Length::Fill),
        ]
        .spacing(12)
        .padding(Padding::new(4.0))
        .into()
    }

    fn view_settings(&self) -> Element<'_, Message> {
        column![
            text("Settings").size(24),
            text(platform::settings_hint()).size(13),
            text("Server URL").size(14),
            text_input("http://127.0.0.1:8787", &self.settings_draft.server_url)
                .on_input(Message::ServerUrlChanged)
                .width(Length::Fill),
            text("Default model (UI + server)").size(14),
            text_input("tiny_en", &self.settings_draft.default_model)
                .on_input(Message::DefaultModelChanged)
                .width(Length::Fill),
            text("Preload models at server startup").size(14),
            text_input("none | all | tiny_en,base", &self.settings_draft.preload_models)
                .on_input(Message::PreloadChanged)
                .width(Length::Fill),
            text("Use none for lazy load when audio arrives; all or an empty --preload-models flag preloads every burn_ready model.")
                .size(12),
            row![
                button("Save").on_press(Message::SaveSettings).padding(10),
                button("Cancel").on_press(Message::CloseSettings).padding(10),
            ]
            .spacing(12),
            scrollable(
                text_editor(&self.output)
                    .on_action(Message::OutputAction)
                    .height(Length::Fill)
            )
            .height(Length::FillPortion(1)),
        ]
        .spacing(10)
        .padding(Padding::new(4.0))
        .into()
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    BlinkTick,
    OpenSettings,
    CloseSettings,
    ServerUrlChanged(String),
    DefaultModelChanged(String),
    PreloadChanged(String),
    SaveSettings,
    PickAudio,
    AudioPicked(Option<AudioFile>),
    ModelSelected(String),
    FormatSelected(String),
    LanguageChanged(String),
    OutputAction(text_editor::Action),
    Refresh,
    CatalogLoaded(Result<AppInit, String>),
    Transcribe,
    Transcribed(Result<Vec<u8>, String>),
}

async fn refresh_catalog(server_url: String) -> Result<AppInit, String> {
    let settings = platform::load_app_settings();
    api::fetch_health(&server_url)
        .await
        .map_err(|e| e.to_string())?;
    let models = api::fetch_models(&server_url)
        .await
        .map_err(|e| e.to_string())?;
    let config_result = api::fetch_config(&server_url).await;
    let server_warning = config_result.is_err();
    let default_model = config_result
        .ok()
        .map(|c| c.default_model)
        .unwrap_or_else(|| settings.default_model.clone());
    let models_loaded = models.iter().any(|m| m.loaded);
    let loaded = models.iter().filter(|m| m.loaded).count();
    let status_line = if server_warning {
        format!(
            "Warning: config endpoint unavailable. Loaded {loaded} model(s) in GPU memory. Default: {default_model}"
        )
    } else {
        format!("Loaded {loaded} model(s) in GPU memory. Default: {default_model}")
    };
    Ok(AppInit {
        settings,
        models,
        default_model,
        server_warning,
        models_loaded,
        status_line,
    })
}

fn preload_to_text(value: &PreloadModels) -> String {
    match value {
        PreloadModels::None => "none".to_string(),
        PreloadModels::All => "all".to_string(),
        PreloadModels::List(names) => names.join(","),
    }
}

fn parse_preload_text(raw: &str) -> PreloadModels {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") {
        PreloadModels::None
    } else if trimmed.eq_ignore_ascii_case("all") {
        PreloadModels::All
    } else {
        PreloadModels::List(
            trimmed
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect(),
        )
    }
}