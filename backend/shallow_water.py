"""A small, reflective-boundary finite-volume shallow-water predictor."""

from __future__ import annotations

import math

import torch


GRAVITY_M_S2 = 9.81
DRY_DEPTH_M = 1e-4


def _as_field(value: torch.Tensor, state: torch.Tensor, name: str) -> torch.Tensor:
    field = torch.as_tensor(value, dtype=state.dtype, device=state.device)
    if field.ndim == 0:
        field = field.reshape(1, 1, 1, 1)
    elif field.ndim == 2:
        field = field.unsqueeze(0).unsqueeze(0)
    elif field.ndim == 3:
        field = field.unsqueeze(1)
    elif field.ndim != 4:
        raise ValueError(f"{name} must be a scalar or a 2D grid")

    if field.shape[1] != 1 or field.shape[-2:] != state.shape[-2:]:
        raise ValueError(f"{name} must match the state grid")
    if field.shape[0] not in (1, state.shape[0]):
        raise ValueError(f"{name} batch dimension must be 1 or match the state")
    if not torch.isfinite(field).all():
        raise ValueError(f"{name} contains non-finite values")
    return field.expand(state.shape[0], -1, -1, -1)


def enforce_physical_state(
    state: torch.Tensor,
    dry_depth_m: float = DRY_DEPTH_M,
) -> torch.Tensor:
    """Keep depth nonnegative and remove momentum from dry cells."""
    if state.ndim != 4 or state.shape[1] != 3:
        raise ValueError("state must have shape (batch, 3, height, width)")
    depth = state[:, 0:1].clamp_min(0.0)
    wet = depth > dry_depth_m
    momentum = torch.where(wet, state[:, 1:3], torch.zeros_like(state[:, 1:3]))
    return torch.cat((depth, momentum), dim=1)


def conserve_depth_volume(
    predicted_depth: torch.Tensor,
    reference_depth: torch.Tensor,
) -> torch.Tensor:
    """Project nonnegative depth predictions to the reference total volume."""
    if (
        predicted_depth.ndim != 4
        or predicted_depth.shape[1] != 1
        or predicted_depth.shape != reference_depth.shape
    ):
        raise ValueError(
            "depth fields must have matching (batch, 1, height, width) shapes"
        )
    reference = reference_depth.clamp_min(0.0)
    predicted = predicted_depth.clamp_min(0.0)
    reference_volume = reference.sum(dim=(-2, -1), keepdim=True)
    predicted_volume = predicted.sum(dim=(-2, -1), keepdim=True)
    minimum_volume = (reference_volume * 1e-6).clamp_min(1e-8)
    has_prediction = predicted_volume > minimum_volume
    scale = reference_volume / predicted_volume.clamp_min(1e-8)
    projected = predicted * scale
    return torch.where(has_prediction, projected, reference)


def needs_physics_fallback(
    predicted_state: torch.Tensor,
    current_state: torch.Tensor,
    rainfall_effective_m_s: torch.Tensor,
    threshold_pct: float,
    dt_seconds: float = 60.0,
) -> bool:
    if not math.isfinite(threshold_pct) or threshold_pct <= 0:
        raise ValueError("fallback volume threshold must be finite and positive")
    if not math.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError("dt_seconds must be finite and positive")
    if (
        predicted_state.ndim != 4
        or predicted_state.shape[1] != 3
        or predicted_state.shape != current_state.shape
    ):
        raise ValueError("predicted and current states must have matching shapes")
    if not torch.isfinite(predicted_state).all():
        return True

    rainfall = _as_field(
        rainfall_effective_m_s,
        current_state,
        "rainfall",
    )
    if (rainfall < 0).any():
        raise ValueError("effective rainfall cannot be negative")

    predicted_volume_depth = predicted_state[:, 0:1].clamp_min(0.0).sum()
    current_volume_depth = current_state[:, 0:1].clamp_min(0.0).sum()
    expected_volume_depth = current_volume_depth + rainfall.sum() * dt_seconds
    if expected_volume_depth <= 1e-8:
        return bool(predicted_volume_depth > 1e-8)
    relative_error = (
        torch.abs(predicted_volume_depth - expected_volume_depth)
        / expected_volume_depth
        * 100.0
    )
    return bool(relative_error > threshold_pct)


def is_dry_without_rain(
    state: torch.Tensor,
    rainfall_effective_m_s: torch.Tensor,
) -> bool:
    if state.ndim != 4 or state.shape[1] != 3:
        raise ValueError("state must have shape (batch, 3, height, width)")
    rainfall = _as_field(
        rainfall_effective_m_s,
        state,
        "rainfall",
    )
    if (rainfall < 0).any():
        raise ValueError("effective rainfall cannot be negative")
    no_water = state[:, 0:1].clamp_min(0.0).sum() <= 1e-8
    no_rain = not bool((rainfall > 0).any())
    return bool(no_water) and no_rain


def _axis_fluxes(
    state: torch.Tensor,
    elevation: torch.Tensor,
    axis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    size = state.shape[axis]
    normal_channel = 1 if axis == -1 else 2
    transverse_channel = 2 if axis == -1 else 1

    ghost_left = state.narrow(axis, 0, 1).clone()
    ghost_right = state.narrow(axis, size - 1, 1).clone()
    ghost_left[:, normal_channel : normal_channel + 1] *= -1.0
    ghost_right[:, normal_channel : normal_channel + 1] *= -1.0
    left = torch.cat((ghost_left, state), dim=axis)
    right = torch.cat((state, ghost_right), dim=axis)
    z_left = torch.cat((elevation.narrow(axis, 0, 1), elevation), dim=axis)
    z_right = torch.cat((elevation, elevation.narrow(axis, size - 1, 1)), dim=axis)

    left_depth = left[:, 0:1].clamp_min(0.0)
    right_depth = right[:, 0:1].clamp_min(0.0)
    face_bed = torch.maximum(z_left, z_right)
    left_face_depth = (z_left + left_depth - face_bed).clamp_min(0.0)
    right_face_depth = (z_right + right_depth - face_bed).clamp_min(0.0)
    left_scale = left_face_depth / left_depth.clamp_min(DRY_DEPTH_M)
    right_scale = right_face_depth / right_depth.clamp_min(DRY_DEPTH_M)
    left_reconstructed = torch.cat(
        (left_face_depth, left[:, 1:3] * left_scale),
        dim=1,
    )
    right_reconstructed = torch.cat(
        (right_face_depth, right[:, 1:3] * right_scale),
        dim=1,
    )

    def physical_flux(
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        depth = values[:, 0:1]
        normal_momentum = values[:, normal_channel : normal_channel + 1]
        transverse_momentum = values[:, transverse_channel : transverse_channel + 1]
        safe_depth = depth.clamp_min(DRY_DEPTH_M)
        normal_velocity = normal_momentum / safe_depth
        flux = torch.zeros_like(values)
        flux[:, 0:1] = normal_momentum
        flux[:, normal_channel : normal_channel + 1] = (
            normal_momentum * normal_velocity
            + 0.5 * GRAVITY_M_S2 * depth.square()
        )
        flux[:, transverse_channel : transverse_channel + 1] = (
            transverse_momentum * normal_velocity
        )
        wave_speed = normal_velocity.abs() + torch.sqrt(
            GRAVITY_M_S2 * depth
        )
        return flux, wave_speed

    left_flux, left_speed = physical_flux(left_reconstructed)
    right_flux, right_speed = physical_flux(right_reconstructed)
    wave_speed = torch.maximum(left_speed, right_speed)
    numerical_flux = (
        0.5 * (left_flux + right_flux)
        - 0.5 * wave_speed * (right_reconstructed - left_reconstructed)
    )

    left_correction = torch.zeros_like(numerical_flux)
    right_correction = torch.zeros_like(numerical_flux)
    left_correction[:, normal_channel : normal_channel + 1] = (
        0.5
        * GRAVITY_M_S2
        * (left_depth.square() - left_face_depth.square())
    )
    right_correction[:, normal_channel : normal_channel + 1] = (
        0.5
        * GRAVITY_M_S2
        * (right_depth.square() - right_face_depth.square())
    )
    return numerical_flux + left_correction, numerical_flux + right_correction


def shallow_water_step(
    state: torch.Tensor,
    rainfall_effective_m_s: torch.Tensor,
    elevation_m: torch.Tensor,
    manning_n: torch.Tensor,
    dt_seconds: float = 60.0,
    cell_size_m: float = 30.0,
    cfl: float = 0.45,
    max_step_seconds: float = 5.0,
    max_substeps: int = 1024,
) -> torch.Tensor:
    """Advance (h, hu, hv) with a CFL-subcycled, first-order SWE solver.

    Raster rows increase southward, so the y-momentum is sign-flipped while
    applying the symmetric finite-volume fluxes and restored on return.
    Rainfall input is net of infiltration, in metres per second.
    """
    if state.ndim != 4 or state.shape[1] != 3:
        raise ValueError("state must have shape (batch, 3, height, width)")
    if not math.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError("dt_seconds must be finite and positive")
    if not math.isfinite(cell_size_m) or cell_size_m <= 0:
        raise ValueError("cell_size_m must be finite and positive")
    if not 0 < cfl <= 1:
        raise ValueError("cfl must be in (0, 1]")
    if max_step_seconds <= 0 or max_substeps < 1:
        raise ValueError("substep limits must be positive")

    rainfall = _as_field(rainfall_effective_m_s, state, "rainfall")
    elevation = _as_field(elevation_m, state, "elevation")
    manning = _as_field(manning_n, state, "Manning roughness")
    if (rainfall < 0).any():
        raise ValueError("effective rainfall cannot be negative")
    if (manning < 0).any():
        raise ValueError("Manning roughness cannot be negative")

    state = enforce_physical_state(state)
    state = torch.cat((state[:, 0:2], -state[:, 2:3]), dim=1)
    remaining = dt_seconds
    substeps = 0

    while remaining > dt_seconds * 1e-10:
        depth = state[:, 0:1]
        safe_depth = depth.clamp_min(DRY_DEPTH_M)
        wave_speed = torch.sqrt(GRAVITY_M_S2 * depth)
        speed_x = state[:, 1:2].abs() / safe_depth + wave_speed
        speed_y = state[:, 2:3].abs() / safe_depth + wave_speed
        rate = (speed_x + speed_y).amax().item() / cell_size_m
        cfl_step = cfl / rate if rate > 0 else max_step_seconds
        step_seconds = min(remaining, max_step_seconds, cfl_step)
        if step_seconds <= 0 or not math.isfinite(step_seconds):
            raise RuntimeError("Could not determine a finite shallow-water step")
        substeps += 1
        if substeps > max_substeps:
            raise RuntimeError(
                f"Shallow-water step exceeded {max_substeps} CFL substeps"
            )

        flux_x_left, flux_x_right = _axis_fluxes(state, elevation, axis=-1)
        flux_y_left, flux_y_right = _axis_fluxes(state, elevation, axis=-2)
        divergence_x = flux_x_left.narrow(-1, 1, state.shape[-1]) - (
            flux_x_right.narrow(-1, 0, state.shape[-1])
        )
        divergence_y = flux_y_left.narrow(-2, 1, state.shape[-2]) - (
            flux_y_right.narrow(-2, 0, state.shape[-2])
        )
        state = state - (step_seconds / cell_size_m) * (
            divergence_x + divergence_y
        )
        depth = (state[:, 0:1] + step_seconds * rainfall).clamp_min(0.0)
        momentum = state[:, 1:3]
        momentum_magnitude = torch.linalg.vector_norm(
            momentum,
            dim=1,
            keepdim=True,
        )
        friction_denominator = 1.0 + (
            step_seconds
            * GRAVITY_M_S2
            * manning.square()
            * momentum_magnitude
            / depth.clamp_min(DRY_DEPTH_M).pow(7.0 / 3.0)
        )
        momentum = momentum / friction_denominator
        state = enforce_physical_state(
            torch.cat((depth, momentum), dim=1)
        )
        remaining = max(0.0, remaining - step_seconds)

    state = torch.cat((state[:, 0:2], -state[:, 2:3]), dim=1)
    if not torch.isfinite(state).all():
        raise FloatingPointError("Shallow-water step produced non-finite values")
    return state
