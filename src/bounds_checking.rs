use geo::{Contains, MultiPolygon, Point};
use gladius_shared::error::SlicerErrors;
use gladius_shared::settings::{BedDimensions, Settings};
use gladius_shared::types::{Command, IndexedTriangle, Vertex};
use itertools::Itertools;

/// Check if the point is in an excluded
fn check_excluded(
    v_point: Point,
    bed_exclude_areas: &Option<MultiPolygon>,
) -> Result<(), SlicerErrors> {
    for polygon in bed_exclude_areas.as_ref() {
        if polygon.contains(&v_point) {
            return Err(SlicerErrors::InExcludeArea(polygon.to_owned()));
        }
    }

    Ok(())
}

pub fn check_model_bounds(
    models: &[(Vec<Vertex>, Vec<IndexedTriangle>)],
    settings: &Settings,
) -> Result<(), SlicerErrors> {
    let brim_width = settings.brim_width.unwrap_or(0.0);
    let shrink_distance = settings.layer_shrink_amount.unwrap_or(0.0);

    let total_offset = brim_width + shrink_distance;

    models
        .iter()
        .flat_map(|model| model.0.iter())
        .map(|v| {
            // Check if the point is in an excluded
            check_excluded(Point::new(v.x, v.y), &settings.bed_exclude_areas)?;

            // Checks if vertices are within bounds of the printer
            match settings.print_dimensions {
                BedDimensions::RectangularBed { print_x, print_y } => {
                    if v.x < total_offset
                        || v.y < total_offset
                        || v.z < -0.00001
                        || v.x > print_x - total_offset
                        || v.y > print_y - total_offset
                    {
                        return Err(SlicerErrors::ModelOutsideBuildArea)
                    };
                },
                BedDimensions::CircularBed { print_radius } => {
                    if v.x < total_offset
                        || v.y < total_offset
                        || v.z < -0.00001
                        || (v.x).powi(2) + (v.y).powi(2) > print_radius.powi(2) // Maybe minus 200 in the case the center of the circle doesn't work
                    {
                        return Err(SlicerErrors::ModelOutsideBuildArea)
                    };
                },
            }

            if v.z > settings.print_z {
                Err(SlicerErrors::ModelOutsideBuildArea)
            } else {
                Ok(())
            }
        })
        .try_collect()
}

pub fn check_moves_bounds(moves: &[Command], settings: &Settings) -> Result<(), SlicerErrors> {
    moves
        .iter()
        .map(|command| match command {
            Command::MoveTo { end, .. } | Command::MoveAndExtrude { end, .. } => {
                match settings.print_dimensions {
                    BedDimensions::RectangularBed { print_x, print_y } => {
                        if end.x < 0.0
                            || end.x > print_x
                            || end.y < 0.0
                            || end.y > print_y
                        {
                            Err(SlicerErrors::MovesOutsideBuildArea)
                        } else {
                            Ok(())
                        }
                    }
                    BedDimensions::CircularBed { print_radius } => {
                        if end.x > -print_radius
                            || end.x < print_radius
                            || end.y > -print_radius
                            || end.y < print_radius
                        {
                            Err(SlicerErrors::MovesOutsideBuildArea)
                        } else {
                            Ok(())
                        }
                    },
                }
            }
            Command::LayerChange { z, .. } => {
                if *z > settings.print_z || *z < 0.0 {
                    Err(SlicerErrors::MovesOutsideBuildArea)
                } else {
                    Ok(())
                }
            }
            Command::Arc { .. } => {
                unimplemented!()
            }
            Command::SetState { .. }
            | Command::Delay { .. }
            | Command::NoAction
            | Command::ChangeObject { .. } => Ok(()),
        })
        .try_collect()
}

#[cfg(test)]
mod bounds_check_tests {
    use super::*;
    use geo::{LineString, Polygon};

    #[test]
    fn test_slice_with_model_in_excluded_area() {
        check_excluded(
            Point::new(30.1, 58.6),
            &Some(MultiPolygon::new(vec![Polygon::new(
                LineString::from(vec![(0.0, 0.0), (256.0, 0.0), (256.0, 256.0), (0.0, 256.0)]),
                Vec::new(),
            )])),
        )
        .unwrap_err();

        check_excluded(
            Point::new(5.7, 8.4),
            &Some(MultiPolygon::new(vec![Polygon::new(
                LineString::from(vec![(0.0, 0.0), (2.0, 0.0), (6.0, 5.0), (0.0, 2.0)]),
                Vec::new(),
            )])),
        )
        .unwrap();
    }
}