#![deny(clippy::unwrap_used)]
#![warn(clippy::all, clippy::perf, clippy::missing_const_for_fn)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use clap::Parser;
use gladius_shared::loader::{Loader, STLLoader, ThreeMFLoader};
use gladius_shared::types::*;
use input::load_settings;

use crate::plotter::convert_objects_into_moves;
use crate::tower::{create_towers, TriangleTower, TriangleTowerIterator};
use geo::{
    coordinate_position, Closest, ClosestPoint, Contains, Coord, CoordinatePosition, GeoFloat,
    Line, MultiPolygon, Point,
};
use gladius_shared::settings::{PartialSettingsFile, Settings, SettingsValidationResult};
use std::fs::File;

use std::ffi::OsStr;
use std::path::Path;

use crate::bounds_checking::{check_model_bounds, check_moves_bounds};
use crate::calculation::calculate_values;
use crate::command_pass::{CommandPass, OptimizePass, SlowDownLayerPass};
use crate::converter::convert;
use crate::plotter::polygon_operations::PolygonOperations;
use crate::slice_pass::*;
use crate::slicing::slice;
use crate::utils::{
    display_state_update, send_error_message, send_warning_message, show_error_message,
    show_warning_message,
};
use gladius_shared::error::SlicerErrors;
use gladius_shared::messages::Message;
use itertools::Itertools;
use log::{debug, info, LevelFilter};
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use simple_logger::SimpleLogger;
use std::collections::HashMap;
use std::io::BufWriter;

mod bounds_checking;
mod calculation;
mod command_pass;
mod converter;
mod input;
mod optimizer;
mod plotter;
mod slice_pass;
mod slicing;
mod tower;
mod utils;
mod test;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[clap(group(
    clap::ArgGroup::new("settings-group")
        .required(true)
        .args(&["settings_file_path", "settings_json"]),
))]
struct Args {
    #[arg(required = true)]
    input: Vec<String>,
    #[arg(short = 'o', help = "Sets the output dir")]
    output: Option<String>,
    #[arg(short = 'v', action = clap::ArgAction::Count, conflicts_with = "message", help = "Sets the level of verbosity")]
    verbose: u8,
    #[arg(short = 's', help = "Sets the settings file to use")]
    settings_file_path: Option<String>,
    #[arg(short = 'S', help = "The contents of a json settings file.")]
    settings_json: Option<String>,
    #[arg(
        short = 'm',
        help = "Use the Message System (useful for interprocess communication)"
    )]
    message: bool,
    #[arg( long="print_settings",help = "Print the final combined settings out to Stdout and Terminate. Verbose level 4 will print but continue.")]
    print_settings: bool,
    #[arg(
        short = 'j',
        help = "Sets the number of threads to use in the thread pool (defaults to number of CPUs)"
    )]
    thread_count: Option<usize>,
}

fn main() {
    // The YAML file is found relative to the current file, similar to how modules are found
    let args: Args = Args::parse();

    // set number of cores for rayon
    if let Some(number_of_threads) = args.thread_count {
        rayon::ThreadPoolBuilder::new()
            .num_threads(number_of_threads)
            .build_global()
            .expect("Only call to build global");
    }

    let send_messages = args.message;

    if !send_messages {
        // Vary the output based on how many times the user used the "verbose" flag
        // (i.e. 'myprog -v -v -v' or 'myprog -vvv' vs 'myprog -v'

        SimpleLogger::new()
            .with_level(match args.verbose {
                0 => LevelFilter::Error,
                1 => LevelFilter::Warn,
                2 => LevelFilter::Info,
                3 => LevelFilter::Debug,
                _ => LevelFilter::Trace,
            })
            .init()
            .expect("Only Logger Setup");
    }

    display_state_update("Loading Inputs", send_messages);

    let settings_json = args.settings_json.unwrap_or(handle_err_or_return(
        input::load_settings_json(
            args.settings_file_path
                .as_deref()
                .expect("CLAP should handle requring a settings option to be Some"),
        ),
        send_messages,
    ));

    let settings =  handle_err_or_return(
        load_settings(
            args.settings_file_path.as_deref(),
            &settings_json
        ),
        send_messages,
    );


    let models = handle_err_or_return(
        crate::input::load_models(
            Some(args.input),
            &settings
        ),
        send_messages,
    );
    if args.print_settings{
        for line in gladius_shared::settings::SettingsPrint::to_strings(&settings){
            println!("{}",line);
        }
        std::process::exit(0);        
    }
    else if log::log_enabled!(log::Level::Trace){
        for line in gladius_shared::settings::SettingsPrint::to_strings(&settings){
            log::trace!("{}",line);
        }
    }

    handle_err_or_return(check_model_bounds(&models, &settings), send_messages);

    handle_setting_validation(settings.validate_settings(), send_messages);

    display_state_update("Creating Towers", send_messages);

    let towers: Vec<TriangleTower> = handle_err_or_return(create_towers(&models), send_messages);

    display_state_update("Slicing", send_messages);

    let objects = handle_err_or_return(slice(&towers, &settings), send_messages);

    display_state_update("Generating Moves", send_messages);

    let mut moves = handle_err_or_return(
        generate_moves(objects, &settings, send_messages),
        send_messages,
    );

    handle_err_or_return(check_moves_bounds(&moves, &settings), send_messages);

    display_state_update("Optimizing", send_messages);
    debug!("Optimizing {} Moves", moves.len());

    OptimizePass::pass(&mut moves, &settings);
    display_state_update("Slowing Layer Down", send_messages);

    SlowDownLayerPass::pass(&mut moves, &settings);

    if send_messages {
        let message = Message::Commands(moves.clone());
        bincode::serialize_into(BufWriter::new(std::io::stdout()), &message)
            .expect("Write Limit should not be hit");
    }
    display_state_update("Calculate Values", send_messages);

    let cv = calculate_values(&moves, &settings);

    if send_messages {
        let message = Message::CalculatedValues(cv);
        bincode::serialize_into(BufWriter::new(std::io::stdout()), &message)
            .expect("Write Limit should not be hit");
    } else {
        let (hour, min, sec, _) = cv.get_hours_minutes_seconds_fract_time();

        info!(
            "Total Time: {} hours {} minutes {:.3} seconds",
            hour, min, sec
        );
        info!(
            "Total Filament Volume: {:.3} cm^3",
            cv.plastic_volume / 1000.0
        );
        info!("Total Filament Mass: {:.3} grams", cv.plastic_weight);
        info!(
            "Total Filament Length: {:.3} meters",
            cv.plastic_length / 1000.0
        );
        info!(
            "Total Filament Cost: ${:.2}",
            (((cv.plastic_volume / 1000.0) * settings.filament.density) / 1000.0)
                * settings.filament.cost
        );
    }

    display_state_update("Outputting G-code", send_messages);

    // Output the GCode
    if let Some(file_path) = &args.output {
        // Output to file
        debug!("Converting {} Moves", moves.len());
        handle_err_or_return(
            convert(
                &moves,
                &settings,
                &mut handle_err_or_return(
                    File::create(file_path).map_err(|_| SlicerErrors::FileCreateError {
                        filepath: file_path.to_string(),
                    }),
                    send_messages,
                ),
            ),
            send_messages,
        );
    } else if send_messages {
        // Output as message
        let mut gcode: Vec<u8> = Vec::new();
        handle_err_or_return(convert(&moves, &settings, &mut gcode), send_messages);
        let message = Message::GCode(
            String::from_utf8(gcode).expect("All write occur from write macro so should be utf8"),
        );
        bincode::serialize_into(BufWriter::new(std::io::stdout()), &message)
            .expect("Write Limit should not be hit");
    } else {
        // Output to stdout
        let stdout = std::io::stdout();
        let mut stdio_lock = stdout.lock();
        debug!("Converting {} Moves", moves.len());
        handle_err_or_return(convert(&moves, &settings, &mut stdio_lock), send_messages);
    };
}

fn generate_moves(
    mut objects: Vec<Object>,
    settings: &Settings,
    send_messages: bool,
) -> Result<Vec<Command>, SlicerErrors> {
    // Creates Support Towers
    SupportTowerPass::pass(&mut objects, settings, send_messages);

    // Adds a skirt
    SkirtPass::pass(&mut objects, settings, send_messages);

    // Adds a brim
    BrimPass::pass(&mut objects, settings, send_messages);

    let v: Result<Vec<()>, SlicerErrors> = objects
        .par_iter_mut()
        .map(|object| {
            let slices = &mut object.layers;

            // Shrink layer
            ShrinkPass::pass(slices, settings, send_messages)?;

            // Handle Perimeters
            PerimeterPass::pass(slices, settings, send_messages)?;

            // Handle Bridging
            BridgingPass::pass(slices, settings, send_messages)?;

            // Handle Top Layer
            TopLayerPass::pass(slices, settings, send_messages)?;

            // Handle Top And Bottom Layers
            TopAndBottomLayersPass::pass(slices, settings, send_messages)?;

            // Handle Support
            SupportPass::pass(slices, settings, send_messages)?;

            // Lightning Infill
            LightningFillPass::pass(slices, settings, send_messages)?;

            // Fill Remaining areas
            FillAreaPass::pass(slices, settings, send_messages)?;

            // Order the move chains
            OrderPass::pass(slices, settings, send_messages)
        })
        .collect();

    v?;

    Ok(convert_objects_into_moves(objects, settings))
}

fn handle_err_or_return<T>(res: Result<T, SlicerErrors>, send_message: bool) -> T {
    match res {
        Ok(data) => data,
        Err(slicer_error) => {
            if send_message {
                send_error_message(slicer_error);
            } else {
                show_error_message(slicer_error);
            }
            std::process::exit(-1);
        }
    }
}

/// Sends an apropreate error/warning message for a `SettingsValidationResult`
fn handle_setting_validation(res: SettingsValidationResult, send_message: bool) {
    match res {
        SettingsValidationResult::NoIssue => {}
        SettingsValidationResult::Warning(slicer_warning) => {
            if send_message {
                send_warning_message(slicer_warning);
            } else {
                show_warning_message(slicer_warning);
            }
        }
        SettingsValidationResult::Error(slicer_error) => {
            if send_message {
                send_error_message(slicer_error);
            } else {
                show_error_message(slicer_error);
            }
            std::process::exit(-1);
        }
    }
}
