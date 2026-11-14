# Physics-Informed Neural Surrogate for Real-Time Urban Flood Forecasting

> **Project status:** Research prototype / proposed system architecture\
> **Initial study area:** Bellandur Lake--Devarabeesanahalli corridor,
> Bengaluru, India\
> **Primary physics engine:** ANUGA\
> **Secondary comparison model:** LISFLOOD-FP\
> **Proposed neural surrogate:** CNN + ConvLSTM with physics-informed
> losses\
> **Proposed frontend:** Next.js + TypeScript + Tailwind CSS + MapLibre
> GL JS\
> **Proposed backend:** Python + FastAPI

------------------------------------------------------------------------

## 1. Project overview

Urban flash flooding can develop faster than conventional,
high-resolution hydrodynamic simulations can be completed. Physics-based
models solve equations describing the conservation of water mass and
momentum over a terrain representation. They can provide detailed
flood-depth and flow information, but repeated simulations across many
rainfall scenarios can be computationally expensive.

This project aims to build a **physics-informed neural surrogate** that
learns from physics-based flood simulations and predicts the evolution
of urban flooding much faster.

The system will:

1.  Assemble terrain, land-cover, soil, building, drainage and rainfall
    data for a selected urban catchment.
2.  Configure a physics-based hydrodynamic model and simulate multiple
    rainfall scenarios.
3.  Save time-varying water depth and momentum/velocity fields as a
    training dataset.
4.  Train a spatiotemporal neural surrogate to predict future flood
    states.
5.  Add physical consistency penalties, especially water-mass
    conservation, to the training objective.
6.  Display rainfall inputs and predicted flood conditions on an
    interactive map.
7.  Evaluate surrogate accuracy, physical consistency, multi-step
    stability and inference speed against the reference solver.

**Important distinction:** The neural model initially learns to
reproduce the selected numerical solver. Agreement with a simulator is
not, by itself, proof of real-world flood accuracy. Independent
observations such as gauge data, surveyed water levels, satellite flood
extents or documented flood locations are needed for real-world
validation.

------------------------------------------------------------------------

## 2. Study area and scope

### 2.1 Candidate study area

The initial candidate is the **Bellandur Lake--Devarabeesanahalli
corridor in Bengaluru, Karnataka, India**. The wider Bellandur / eastern
Bengaluru region has experienced urban waterlogging and flood impacts
and is a relevant candidate for a pilot study.

The exact computational catchment boundary must be finalized after
inspecting the DEM, drainage connectivity and available local data. A
suggested first experiment is a manageable pilot domain of approximately
**1 km × 1 km**. This is a planning choice, not a surveyed catchment
boundary.

### 2.2 Initial modelling resolution

Start with a prototype resolution that the available terrain data can
support. A 30 m DEM can support an initial regional prototype, but it
cannot resolve individual street curbs, narrow drains, building
thresholds or many urban micro-topographic features.

For street-scale modelling, obtain higher-resolution bare-earth
elevation data (for example, LiDAR or surveyed terrain), detailed
stormwater-drain geometry and reliable rainfall observations. Do not
interpret resampling a 30 m DEM to 5 m as creating genuine 5 m terrain
information.

### 2.3 Modelling scope

The first milestone should model pluvial / rainfall-driven surface
flooding. River inflows, lake operations, sewer surcharge, pump
operation and detailed pipe-network hydraulics should be included only
when reliable input data and suitable coupling methods are available.

------------------------------------------------------------------------

## 3. System architecture

``` text
                    GEOSPATIAL + WEATHER DATA
       DEM | Land cover | Soil | Buildings | Drains | Rainfall
                              |
                              v
                    DATA PREPROCESSING
       CRS alignment | clipping | rasterization | quality checks
                              |
                              v
                    PHYSICS SIMULATION LAYER
                    ANUGA (primary solver)
                 LISFLOOD-FP (comparison solver)
                              |
                              v
                   SIMULATION DATASET STORE
       time | rainfall | depth | x-momentum | y-momentum | masks
                              |
                              v
                 PHYSICS-INFORMED SURROGATE
                   CNN encoder + ConvLSTM
             future depth / momentum / flood probability
                              |
                              v
                     BACKEND / INFERENCE API
             scenario management | inference | metrics
                              |
                              v
                       WEB FRONTEND
          interactive map | rainfall controls | timeline
          depth / velocity layers | charts | warnings
```

### 3.1 Main components

  -----------------------------------------------------------------------
  Component                           Responsibility
  ----------------------------------- -----------------------------------
  Geospatial data pipeline            Obtain, clean, align and prepare
                                      geographic layers

  Physics engine                      Generate physically based flood
                                      simulations

  Scenario generator                  Define rainfall patterns, durations
                                      and initial states

  Dataset builder                     Convert simulation outputs into
                                      training sequences

  Neural surrogate                    Predict future flood states

  Physics-informed loss               Penalize mass/momentum
                                      inconsistencies and invalid depths

  Backend API                         Run scenarios, serve predictions
                                      and expose metadata

  Frontend                            Configure rainfall and visualize
                                      results

  Evaluation module                   Compare predictions with solver
                                      output and observations
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 4. Geospatial and rainfall datasets

No single public dataset is expected to contain every required variable.
The project combines several sources. Coverage, licensing, resolution
and completeness must be checked for the selected boundary before
modelling.

  ------------------------------------------------------------------------------------------------------------------------------------
  Data              Candidate source                                                              Model use          Important
                                                                                                                     limitation
  ----------------- ----------------------------------------------------------------------------- ------------------ -----------------
  Elevation / DEM   [Copernicus Data Space](https://dataspace.copernicus.eu/) -- Copernicus DEM   Ground/surface     GLO-30 is a
                    GLO-30                                                                        elevation and      digital surface
                                                                                                  terrain gradients  model; buildings
                                                                                                                     and vegetation
                                                                                                                     can affect it

  Land cover        [ESA WorldCover](https://esa-worldcover.org/en/data-access)                   Land-cover classes Land-cover class
                                                                                                  for roughness /    is not a direct
                                                                                                  infiltration       measurement of
                                                                                                  parameterization   Manning's n

  Soil properties   [ISRIC SoilGrids](https://soilgrids.org/)                                     Texture and other  SoilGrids does
                                                                                                  soil attributes    not directly
                                                                                                  used to estimate   provide every
                                                                                                  infiltration       hydraulic
                                                                                                  parameters         parameter
                                                                                                                     required by
                                                                                                                     Green-Ampt

  Buildings, roads, [OpenStreetMap](https://www.openstreetmap.org/) / [Geofabrik                  Building           Completeness and
  waterways         India](https://download.geofabrik.de/asia/india.html)                         footprints, roads  positional
                                                                                                  and mapped         accuracy vary;
                                                                                                  waterways          mapped drains may
                                                                                                                     be incomplete

  Stormwater drains BBMP / [OpenCity Bengaluru datasets](https://data.opencity.in/)               Drain locations,   Verify source
                                                                                                  connectivity and   date, geometry,
                                                                                                  attributes         connectivity and
                                                                                                                     completeness

  Historical        [IMD gridded                                                                  Historical event   0.25° daily
  rainfall          rainfall](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html)   context and        rainfall is too
                                                                                                  scenario design    coarse for
                                                                                                                     minute-scale
                                                                                                                     street-level
                                                                                                                     forcing

  Rain gauges /     Local IMD, municipal or research observations, where accessible               High-frequency     Availability and
  radar                                                                                           rainfall forcing   access must be
                                                                                                  and validation     confirmed

  Flood             BBMP records, field surveys, satellite flood products, reports and gauge      Independent        Records may be
  observations      observations                                                                  validation         incomplete,
                                                                                                                     irregular or lack
                                                                                                                     precise
                                                                                                                     depth/time
  ------------------------------------------------------------------------------------------------------------------------------------

### 4.1 Preprocessing requirements

All spatial layers must be prepared consistently:

1.  Define and document the study-area polygon.
2.  Select an appropriate projected coordinate reference system in
    metres.
3.  Clip input layers to the study area plus a suitable buffer.
4.  Check DEM vertical datum, nodata cells, sinks and artifacts.
5.  Align raster layers to a common transform, extent and grid or map
    them to the ANUGA mesh.
6.  Rasterize or vectorize building and drainage features as required by
    the chosen solver.
7.  Derive land-cover-based roughness classes and document the assigned
    values.
8.  Estimate soil hydraulic parameters using defensible pedotransfer
    functions or local measurements.
9.  Preserve source metadata, acquisition dates, units, CRS and
    processing history.

Suggested tools: QGIS, GDAL, Rasterio, GeoPandas, Shapely, Xarray and
NumPy.

### 4.2 Important parameter distinction

-   **Manning roughness, n:** represents resistance to surface flow. It
    can be assigned by land-cover class and calibrated.
-   **Saturated hydraulic conductivity, Ks:** controls infiltration
    capacity. It is not the same as surface roughness.
-   **Initial soil moisture:** affects infiltration and must be
    measured, sourced or set as a documented scenario assumption.
-   **Building representation:** buildings may be treated as blocked
    areas, elevated cells, sub-grid obstructions or through a
    solver-specific urban representation. The chosen method must be
    documented.
-   **Drainage:** a line on a map is not a hydraulic drain model.
    Capacity, invert elevation, connectivity, inlets and outlet
    conditions matter.

------------------------------------------------------------------------

## 5. Initial conditions, forcing and boundary conditions

### 5.1 Static spatial inputs

Examples include:

-   Bed / terrain elevation: `z_b(x, y)` in metres.
-   Manning roughness: `n(x, y)` (commonly represented as a
    dimensionless empirical coefficient).
-   Soil hydraulic conductivity: `K_s(x, y)` in m/s.
-   Initial soil moisture: `theta_i(x, y)`.
-   Buildings / impermeable masks.
-   Drainage network and outlet locations.
-   Computational mesh or raster-cell dimensions.

### 5.2 Dynamic state variables

The 2D shallow-water state is commonly represented by:

\[ `\mathbf{U}`{=tex} =
```{=tex}
\begin{bmatrix}
h\\hu\\hv
\end{bmatrix}
```
\]

where:

-   `h` = water depth (m).
-   `u` = depth-averaged X velocity (m/s).
-   `v` = depth-averaged Y velocity (m/s).
-   `hu`, `hv` = unit-width discharges / conserved momentum variables
    (m²/s).

The water-surface elevation is:

\[ `\eta `{=tex}= z_b + h \]

### 5.3 Initial conditions

For a dry-start prototype:

\[ h(x,y,0)=0,`\qquad `{=tex}hu(x,y,0)=0,`\qquad `{=tex}hv(x,y,0)=0 \]

and:

\[ `\eta`{=tex}(x,y,0)=z_b(x,y) \]

For a wet-start scenario, specify a measured or explicitly assumed
initial depth `h0(x,y)` and initial velocities. Do not present an
arbitrary wet-start value as an observation.

### 5.4 Rainfall and other forcing

Rainfall intensity `R(x,y,t)` is converted to a consistent
depth-per-time unit, usually m/s, before being passed to the solver.

A rainfall scenario should define:

-   spatial rainfall pattern;
-   time series / hyetograph;
-   event duration;
-   temporal resolution;
-   units and source;
-   whether the rainfall is observed or synthetic.

Other source/sink terms may include infiltration `I`, drainage loss `D`,
river or upstream inflow, and other model-specific terms. These terms
must be implemented in a manner consistent with the selected solver.

### 5.5 Boundary conditions

Specify each boundary as appropriate:

-   closed / reflective boundary;
-   prescribed stage;
-   prescribed inflow or discharge;
-   open / transmissive boundary;
-   outlet or drainage boundary.

Boundary conditions can strongly affect results. They must represent the
hydrological connectivity of the selected domain rather than being
chosen only for convenience.

------------------------------------------------------------------------

## 6. Physics model 1: ANUGA

### 6.1 Role

ANUGA is the primary candidate physics engine for this project. It is an
open-source Python hydrodynamic model that solves the 2D shallow-water
equations using a finite-volume approach on an unstructured triangular
mesh.

Official documentation: [ANUGA
documentation](https://anuga.readthedocs.io/en/stable/).

### 6.2 Computational representation

The study area is represented by triangular cells. Each cell carries
quantities such as elevation, stage, water depth and momentum. Fluxes
are computed across cell edges, and the conserved state is updated over
time.

### 6.3 Governing equations

A general conservative 2D shallow-water formulation is:

\[ `\frac{\partial \mathbf U}{\partial t}`{=tex}
+`\frac{\partial \mathbf F(\mathbf U)}{\partial x}`{=tex}
+`\frac{\partial \mathbf G(\mathbf U)}{\partial y}`{=tex}
=`\mathbf `{=tex}S \]

with:

\[ `\mathbf `{=tex}U =
```{=tex}
\begin{bmatrix}
h\\hu\\hv
\end{bmatrix}
```
\]

\[ `\mathbf `{=tex}F =
```{=tex}
\begin{bmatrix}
hu\\
hu^2+\frac{1}{2}gh^2\\
huv
\end{bmatrix}
```
, `\qquad`{=tex} `\mathbf `{=tex}G =
```{=tex}
\begin{bmatrix}
hv\\
huv\\
hv^2+\frac{1}{2}gh^2
\end{bmatrix}
```
\]

A representative source vector is:

\[ `\mathbf `{=tex}S =
```{=tex}
\begin{bmatrix}
R-I-D\\
-gh\frac{\partial z_b}{\partial x}-\tau_{fx}\\
-gh\frac{\partial z_b}{\partial y}-\tau_{fy}
\end{bmatrix}
```
\]

where `g` is gravitational acceleration, `R` rainfall input, `I`
infiltration, `D` a depth-equivalent drainage sink, and `tau_f`
represents friction source terms. Actual source/operator implementation
depends on the ANUGA setup.

For Manning friction, a commonly used depth-averaged friction term in
the X momentum equation is:

\[ `\tau`{=tex}\_{fx} = `\frac{g n^2 u\sqrt{u^2+v^2}}{h^{1/3}}`{=tex} \]

and similarly:

\[ `\tau`{=tex}\_{fy} = `\frac{g n^2 v\sqrt{u^2+v^2}}{h^{1/3}}`{=tex} \]

The solver requires suitable handling near dry cells to avoid division
by very small water depths.

### 6.4 Finite-volume update

For cell `i` with area `A_i`, a conceptual conservative update is:

\[ `\mathbf `{=tex}U_i\^{n+1} = `\mathbf `{=tex}U_i\^n
-`\frac{\Delta t}{A_i}`{=tex} `\sum`{=tex}\_{e
`\in `{=tex}`\partial `{=tex}i} `\widehat{\mathbf F}`{=tex}\_e L_e
+`\Delta `{=tex}t,`\mathbf `{=tex}S_i \]

where `L_e` is the edge length and `Fhat_e` is the numerical flux across
the edge. The precise flux reconstruction, limiting, source treatment
and wetting/drying treatment are solver implementation details.

### 6.5 ANUGA outputs

Depending on configured output quantities and export steps:

-   stage / water-surface elevation;
-   water depth;
-   X momentum and Y momentum;
-   derived X and Y velocity;
-   flood extent;
-   time and scenario metadata.

Velocity is derived from momentum when depth is safely above a wet/dry
threshold:

\[ u=`\frac{hu}{h}`{=tex},`\qquad `{=tex}v=`\frac{hv}{h}`{=tex} \]

Speed and direction are:

\[ V=`\sqrt{u^2+v^2}`{=tex} \]

\[ `\theta`{=tex}=`\operatorname{atan2}`{=tex}(v,u) \]

The angle convention must be documented (for example, mathematical angle
from positive X, or a converted compass bearing). Direction is undefined
or not meaningful in dry cells.

------------------------------------------------------------------------

## 7. Physics model 2: LISFLOOD-FP

### 7.1 Role

LISFLOOD-FP is a raster-based flood inundation model with multiple
solver formulations. It is proposed as a secondary model for comparison
and independent numerical experiments.

Official project information: [University of Bristol
LISFLOOD-FP](https://www.bristol.ac.uk/geography/research/hydrology/models/lisflood/).

### 7.2 Raster representation

The study domain is represented by a regular grid. Cells store terrain
elevation and evolving water state. Water is routed between neighbouring
cell faces.

### 7.3 Solver formulations

LISFLOOD-FP has multiple formulations, and the exact governing equations
depend on the selected solver/version. One computationally efficient
option is the **local inertial formulation**, which retains local
acceleration, water-surface gradients and friction while neglecting
advective acceleration terms.

For this formulation, define unit-width discharges:

\[ q_x=hu,`\qquad `{=tex}q_y=hv \]

The continuity equation is:

\[ `\frac{\partial h}{\partial t}`{=tex}
+`\frac{\partial q_x}{\partial x}`{=tex}
+`\frac{\partial q_y}{\partial y}`{=tex} =R-I-D \]

A representative local-inertial momentum equation is:

\[ `\frac{\partial q_x}{\partial t}`{=tex}
+gh`\frac{\partial \eta}{\partial x}`{=tex}
+`\frac{g n^2 q_x\sqrt{q_x^2+q_y^2}}{h^{7/3}}`{=tex}=0 \]

\[ `\frac{\partial q_y}{\partial t}`{=tex}
+gh`\frac{\partial \eta}{\partial y}`{=tex}
+`\frac{g n^2 q_y\sqrt{q_x^2+q_y^2}}{h^{7/3}}`{=tex}=0 \]

where `eta = z_b + h`. These equations describe the local-inertial
approximation, not every LISFLOOD-FP solver.

### 7.4 Why compare it with ANUGA?

The models use different spatial representations and may use different
numerical approximations. A comparison can help assess sensitivity to
solver choice, but it is not automatically an independent real-world
validation. For a fair comparison:

-   use consistent terrain and vertical datum;
-   align spatial domains and compare on a common grid;
-   use equivalent rainfall, initial and boundary conditions;
-   document differences in roughness, infiltration and drainage
    treatment;
-   compare at matching output times;
-   report both model configuration and numerical limitations.

------------------------------------------------------------------------

## 8. Physics simulation workflow

The order of operations for each simulation is:

1.  Load and validate terrain and spatial parameter layers.
2.  Create the ANUGA mesh and assign elevation / roughness.
3.  Set initial stage, water depth and momentum.
4.  Configure boundary conditions.
5.  Configure rainfall and other source/sink processes.
6.  Select solver settings and a stable time-step strategy.
7.  Advance the hydrodynamic state.
8.  Save outputs at chosen intervals.
9.  Derive depth, velocity, speed, direction and flood extent.
10. Check water balance, numerical stability and output validity.
11. Repeat for additional rainfall and parameter scenarios.

A CFL-based stability restriction is commonly relevant for explicit
shallow-water solvers. The allowable time step depends on cell size and
local wave/flow speeds. Use the solver's own time-step control and
document the settings; do not assume one fixed time step works for every
mesh and event.

------------------------------------------------------------------------

## 9. Simulation dataset generation

### 9.1 Scenario design

Create multiple rainfall events that vary in:

-   peak intensity;
-   duration;
-   temporal pattern (constant, front-loaded, back-loaded, multi-peak);
-   spatial distribution;
-   antecedent soil moisture;
-   initial water state;
-   drainage assumptions and parameter uncertainty.

Use historical rainfall events where suitable observations exist. Use
synthetic hyetographs for controlled experiments, and label them clearly
as synthetic.

### 9.2 Saved variables

At each simulation time step, save at minimum:

  -----------------------------------------------------------------------
  Variable                            Description
  ----------------------------------- -----------------------------------
  `time`                              Simulation timestamp / elapsed
                                      seconds

  `rainfall`                          Rainfall forcing for the time step

  `water_depth`                       `h` in metres

  `xmomentum`                         `hu`

  `ymomentum`                         `hv`

  `x_velocity`                        Derived `u`, with dry-cell mask

  `y_velocity`                        Derived `v`, with dry-cell mask

  `speed`                             `sqrt(u^2 + v^2)`

  `direction`                         `atan2(v, u)` with documented
                                      convention

  `water_surface`                     `z_b + h`

  `flood_mask`                        Boolean inundation mask using a
                                      documented depth threshold
  -----------------------------------------------------------------------

Also store static channels such as elevation, roughness, soil parameter
estimates, building mask and drainage representation.

### 9.3 Suggested storage format

Use NetCDF or Zarr for gridded time series, with GeoTIFF for static
raster layers and GeoJSON / GeoPackage for vector data. Store metadata
including CRS, grid transform, units, no-data values, solver version,
mesh/grid configuration, scenario parameters and source-data provenance.

Example logical tensor layout:

``` text
scenario/
  metadata.json
  static/
    elevation
    manning_n
    soil_parameters
    building_mask
    drainage_features
  forcing/
    rainfall(time, y, x)
  state/
    water_depth(time, y, x)
    xmomentum(time, y, x)
    ymomentum(time, y, x)
```

For ANUGA's triangular mesh, the raw data may initially be indexed by
triangle rather than `(y, x)`. If the neural network uses raster
tensors, interpolate or rasterize the mesh outputs onto a documented
common grid. Preserve masks and avoid treating interpolated values as
direct solver cell values.

### 9.4 Train / validation / test split

Split by complete rainfall scenarios or events, not by randomly shuffled
frames from the same simulation. This reduces temporal leakage.

A possible split is 70% training scenarios, 15% validation scenarios and
15% held-out test scenarios. The final split should be grouped by event
and documented.

------------------------------------------------------------------------

## 10. Neural surrogate model

### 10.1 Proposed architecture: CNN + ConvLSTM

The proposed surrogate is a spatiotemporal model:

-   **CNN encoder:** learns local spatial features from terrain, land
    cover, roughness, buildings, drainage, rainfall and current state.
-   **ConvLSTM:** learns how spatial flood patterns evolve over time
    while retaining spatial structure.
-   **Prediction head:** outputs future water depth and preferably X/Y
    momentum (or velocity with appropriate wet/dry handling).

The exact architecture, channel counts, number of layers, sequence
length and forecast horizon are experimental choices to be tuned on
validation scenarios.

### 10.2 Inputs

A possible input tensor contains:

-   static channels: elevation, roughness, soil parameters, building
    mask and drainage representation;
-   dynamic channels: recent water depth, X momentum and Y momentum;
-   rainfall sequence: recent and forecast rainfall fields / time
    series;
-   optional boundary-condition and scenario channels.

### 10.3 Outputs

Prefer predicting conserved quantities:

\[ `\widehat{\mathbf U}`{=tex}\_{t+1:t+K} =
\[`\hat `{=tex}h,`\widehat{hu}`{=tex},`\widehat{hv}`{=tex}\] \]

Then derive velocity, speed and direction:

\[ `\hat `{=tex}u=`\frac{\widehat{hu}}{\max(\hat h,\epsilon)}`{=tex},
`\quad`{=tex}
`\hat `{=tex}v=`\frac{\widehat{hv}}{\max(\hat h,\epsilon)}`{=tex} \]

\[ `\hat `{=tex}V=`\sqrt{\hat u^2+\hat v^2}`{=tex}, `\quad`{=tex}
`\hat`{=tex}`\theta`{=tex}=`\operatorname{atan2}`{=tex}(`\hat `{=tex}v,`\hat `{=tex}u)
\]

Use a wet/dry mask and a documented threshold. A non-negative depth
output can be enforced with an appropriate output activation or penalty,
while avoiding numerical artifacts.

### 10.4 CNN operation

A convolution extracts local spatial patterns:

\[ Y(i,j)=`\sum`{=tex}\_m`\sum`{=tex}\_n X(i+m,j+n)K(m,n) \]

where `X` is an input feature map and `K` is a learned filter.

### 10.5 ConvLSTM operation

ConvLSTM uses convolutional gates to maintain spatial memory over time.
A representative form is:

\[ i_t=`\sigma`{=tex}(W\_{xi}*X_t+W\_{hi}*H\_{t-1}+b_i) \]

\[ f_t=`\sigma`{=tex}(W\_{xf}*X_t+W\_{hf}*H\_{t-1}+b_f) \]

\[ C_t=f_t`\odot `{=tex}C\_{t-1}+i_t`\odot`{=tex}`\widetilde `{=tex}C_t
\]

\[ H_t=o_t`\odot`{=tex}`\tanh`{=tex}(C_t) \]

Here `*` denotes convolution, `sigma` is the sigmoid function, `C` is
cell memory and `H` is hidden state. The output gate equation is omitted
above for brevity; a full implementation includes its learned gate
calculation.

------------------------------------------------------------------------

## 11. Physics-informed training objective

A data-only model may reproduce training examples while producing
physically implausible states during rollout. The proposed training
objective combines data fit with physical constraints.

\[ L\_{`\text{total}`{=tex}}= `\lambda`{=tex}*d L*{`\text{data}`{=tex}}
+`\lambda`{=tex}*m L*{`\text{mass}`{=tex}} +`\lambda`{=tex}*p
L*{`\text{momentum}`{=tex}} +`\lambda`{=tex}*+
L*{`\text{positive}`{=tex}} \]

### 11.1 Data loss

For example, mean squared error over wet and valid cells:

\[ L\_{`\text{data}`{=tex}}=
`\operatorname{MSE}`{=tex}(`\hat `{=tex}h,h)
+`\alpha`{=tex}`\operatorname{MSE}`{=tex}(`\widehat{hu}`{=tex},hu)
+`\beta`{=tex}`\operatorname{MSE}`{=tex}(`\widehat{hv}`{=tex},hv) \]

Use masks and variable scaling so that dry cells and variables with
different magnitudes do not dominate incorrectly.

### 11.2 Mass conservation loss

The mass residual is:

\[ r_m= `\frac{\partial\hat h}{\partial t}`{=tex}
+`\frac{\partial(\widehat{hu})}{\partial x}`{=tex}
+`\frac{\partial(\widehat{hv})}{\partial y}`{=tex} -R+I+D \]

\[ L\_{`\text{mass}`{=tex}}=`\operatorname{mean}`{=tex}(r_m\^2) \]

The derivatives should be computed consistently with the model grid and
time spacing. Boundary fluxes must be included when calculating
domain-wide water balance.

### 11.3 Momentum residual loss

Momentum residuals can be formed from the selected shallow-water
equations, including pressure, bed-slope, friction and any implemented
source terms:

\[ L\_{`\text{momentum}`{=tex}}=
`\operatorname{mean}`{=tex}(r_x^2+r_y^2) \]

The residual formulation must match the physics solver assumptions and
the variables predicted by the network.

### 11.4 Non-negative depth loss

\[ L\_{`\text{positive}`{=tex}}=
`\operatorname{mean}`{=tex}`\left`{=tex}(`\max`{=tex}(0,-`\hat `{=tex}h)\^2`\right`{=tex})
\]

Alternatively, constrain the output parameterization to be non-negative.

### 11.5 Long-horizon stability

Train and evaluate autoregressive rollouts, not only one-step
predictions. Useful methods to investigate include:

-   multi-step rollout loss;
-   scheduled sampling / curriculum rollout;
-   mass-balance penalties at each predicted step;
-   positivity constraints;
-   gradient clipping;
-   CFL-aware training time increments;
-   uncertainty or out-of-distribution warnings.

Do not claim stability until it has been measured on long, held-out
forecast rollouts.

------------------------------------------------------------------------

## 12. Frontend application

### 12.1 Proposed stack

  -----------------------------------------------------------------------
  Technology                          Purpose
  ----------------------------------- -----------------------------------
  Next.js + React + TypeScript        Web application and component
                                      architecture

  Tailwind CSS                        Responsive styling and design
                                      system

  MapLibre GL JS                      Interactive map, raster overlays
                                      and vector layers

  Recharts                            Rainfall, depth and forecast
                                      time-series charts

  Zustand or React state              UI and scenario state management

  WebSocket or polling                Receive simulation / inference
                                      status and updates
  -----------------------------------------------------------------------

MapLibre is proposed for an open map-rendering workflow. Map tiles and
basemaps must come from a permitted provider and follow that provider's
attribution and usage terms.

### 12.2 Dashboard layout

The interface should function as an urban flood-monitoring command
centre.

**A. Header / top status bar** - Project name and selected study area. -
Current scenario name. - Simulation status: idle, running, completed or
failed. - Selected forecast time and data freshness.

**B. Main interactive map** - Bellandur study-area boundary. - Terrain /
basemap. - Water-depth heatmap. - Flood extent overlay. - Flow-direction
arrows or vector field. - Optional buildings, roads, lakes and drainage
layers. - Legend with units and thresholds.

**C. Rainfall and scenario controls** - Rainfall intensity or
time-series editor. - Rainfall duration. - Spatial rainfall pattern
selection. - Initial condition selection (dry / configured wet start). -
Forecast horizon. - Run, pause, reset and compare actions.

**D. KPI cards** - Maximum predicted water depth. - Inundated area (with
stated depth threshold). - Peak flow speed. - Selected forecast time. -
Scenario rainfall total.

**E. Time-series panel** - Rainfall hyetograph. - Maximum / mean water
depth over time. - Inundated area over time. - Optional water-balance
diagnostic.

**F. Layer controls** - Water depth. - Velocity magnitude. - Flow
direction. - Flood extent. - Elevation. - Roads / buildings / drainage.

**G. Scenario comparison** - Compare surrogate output with
physics-solver output. - Compare two rainfall scenarios. - Display error
maps only when reference outputs exist.

### 12.3 Map colour and interpretation

Use a documented depth legend, for example, increasing shades for
increasing depth. The thresholds must be configurable and should not be
described as official safety categories unless sourced from an
applicable standard. Show units (m) and the forecast timestamp on every
map layer.

Flow arrows should be derived from predicted velocity components. Do not
infer flow direction from terrain slope alone.

### 12.4 Responsive behaviour and accessibility

-   Desktop: map-centred multi-panel dashboard.
-   Tablet: collapsible side panels.
-   Mobile: map-first layout with bottom sheets for controls and charts.
-   Provide keyboard-accessible controls, visible focus states,
    colour-safe legends and text alternatives for map summaries.

------------------------------------------------------------------------

## 13. Backend and API design

### 13.1 Proposed backend stack

-   Python 3.
-   FastAPI for REST endpoints.
-   Pydantic for request / response validation.
-   NumPy, Xarray and Rasterio for data handling.
-   GeoPandas / Shapely for vector geometry.
-   PyTorch for neural inference.
-   ANUGA and LISFLOOD-FP invoked through a managed simulation workflow.
-   Background worker / job queue for long-running physics simulations.
-   Object storage or filesystem for scenario outputs and model
    artefacts.

Physics simulations should not run synchronously inside a normal web
request. Create a job, return a job ID, execute it in a worker and
report status.

### 13.2 Suggested API endpoints

  --------------------------------------------------------------------------------------
  Method                  Endpoint                               Purpose
  ----------------------- -------------------------------------- -----------------------
  GET                     `/api/health`                          Service health

  GET                     `/api/study-area`                      Study-area boundary and
                                                                 metadata

  GET                     `/api/layers`                          Available geographic
                                                                 layers

  POST                    `/api/scenarios`                       Create rainfall /
                                                                 initial-condition
                                                                 scenario

  POST                    `/api/simulations`                     Submit physics
                                                                 simulation job

  GET                     `/api/jobs/{job_id}`                   Check simulation or
                                                                 inference status

  POST                    `/api/predictions`                     Submit surrogate
                                                                 prediction request

  GET                     `/api/scenarios/{id}/times`            Available output
                                                                 timestamps

  GET                     `/api/scenarios/{id}/layers/{layer}`   Retrieve depth /
                                                                 velocity / extent layer

  GET                     `/api/scenarios/{id}/metrics`          Return summary metrics

  GET                     `/api/scenarios/compare`               Compare scenario or
                                                                 reference outputs
  --------------------------------------------------------------------------------------

These are proposed routes, not existing deployed endpoints.

### 13.3 Example scenario request

``` json
{
  "study_area_id": "bellandur_pilot",
  "rainfall": {
    "type": "hyetograph",
    "units": "mm/hr",
    "duration_minutes": 60,
    "values": [
      {"minute": 0, "intensity": 0},
      {"minute": 10, "intensity": 30},
      {"minute": 20, "intensity": 60},
      {"minute": 40, "intensity": 100},
      {"minute": 60, "intensity": 0}
    ]
  },
  "initial_condition": "dry",
  "forecast_minutes": 60
}
```

The values above are illustrative synthetic inputs, not an observed
rainfall event.

### 13.4 Live update flow

1.  User changes rainfall inputs.
2.  Frontend validates and submits a scenario.
3.  Backend creates a scenario/job ID.
4.  The inference worker loads static geographic channels and rainfall
    inputs.
5.  Surrogate predicts future states.
6.  Backend stores outputs and metrics.
7.  Frontend receives job status and fetches the selected time layer.
8.  Map, KPIs and charts update together.

For a fast surrogate forecast, the frontend should request a new
inference when inputs change, with debouncing to avoid launching a
request for every keystroke. A physics simulation should be clearly
distinguished from a surrogate inference in the UI.

------------------------------------------------------------------------

## 14. Evaluation and research metrics

The project must evaluate more than visual similarity.

### 14.1 Numerical accuracy against the reference solver

For depth:

\[
MAE=`\frac{1}{N}`{=tex}`\sum`{=tex}\_{i=1}\^{N}\|`\hat `{=tex}h_i-h_i\|
\]

\[ RMSE=`\sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat h_i-h_i)^2}`{=tex} \]

Also report: - depth bias; - maximum-depth error; - flood-extent
Intersection over Union (IoU); - precision / recall for inundated
cells; - errors at multiple forecast lead times; - velocity-component
and speed errors, with wet-cell masks.

### 14.2 Physical consistency

Report: - domain-wide water-volume balance; - mass residual over time; -
negative-depth frequency / magnitude; - momentum residuals where
implemented; - stability during multi-step rollout.

A useful volume-balance diagnostic is:

\[ E\_{`\text{mass}`{=tex}}=
V\_{`\text{final}`{=tex}}-V\_{`\text{initial}`{=tex}}
-V\_{`\text{rain}`{=tex}} +V\_{`\text{infiltration}`{=tex}}
+V\_{`\text{drainage}`{=tex}} -V\_{`\text{net boundary inflow}`{=tex}}
\]

Use consistent sign conventions and include boundary fluxes.

### 14.3 Speed comparison

Measure: - physics solver wall-clock time; - surrogate inference time; -
preprocessing time separately; - hardware, grid/mesh size, batch size
and forecast horizon.

Speedup is:

\[ `\text{Speedup}`{=tex}= `\frac{T_{\text{physics solver}}}`{=tex}
{T\_{`\text{surrogate inference}`{=tex}}} \]

Compare equivalent domains, outputs and forecast horizons. Do not report
a speedup until measured under documented conditions.

### 14.4 Real-world validation

Where possible, compare against independent observations: - measured
water depths / gauge levels; - time-stamped flood extent from satellite
imagery; - field-surveyed inundation; - documented road closure /
waterlogging locations; - observed rainfall and lake / drain levels.

Separate simulator-to-surrogate agreement from real-world validation in
all reports.

------------------------------------------------------------------------

## 15. Suggested repository structure

``` text
urban-flood-surrogate/
│
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
│
├── data/
│   ├── raw/
│   │   ├── dem/
│   │   ├── landcover/
│   │   ├── soil/
│   │   ├── buildings/
│   │   ├── roads/
│   │   ├── drainage/
│   │   ├── rainfall/
│   │   └── flood_observations/
│   ├── processed/
│   └── scenarios/
│
├── physics/
│   ├── anuga/
│   │   ├── build_mesh.py
│   │   ├── configure_domain.py
│   │   ├── rainfall.py
│   │   ├── boundaries.py
│   │   └── run_simulation.py
│   ├── lisflood/
│   │   ├── prepare_inputs.py
│   │   ├── run_model.py
│   │   └── compare_outputs.py
│   └── validation/
│       ├── mass_balance.py
│       └── solver_checks.py
│
├── dataset/
│   ├── export_outputs.py
│   ├── rasterize_mesh.py
│   ├── make_sequences.py
│   └── split_scenarios.py
│
├── ml/
│   ├── models/
│   │   ├── cnn_encoder.py
│   │   ├── convlstm.py
│   │   └── surrogate.py
│   ├── losses/
│   │   ├── data_loss.py
│   │   ├── mass_loss.py
│   │   ├── momentum_loss.py
│   │   └── positivity_loss.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── workers/
│   └── requirements.txt
│
├── frontend/
│   ├── app/
│   ├── components/
│   │   ├── map/
│   │   ├── charts/
│   │   ├── controls/
│   │   └── metrics/
│   ├── hooks/
│   ├── lib/
│   └── package.json
│
├── configs/
│   ├── study_area.yaml
│   ├── physics.yaml
│   ├── scenarios.yaml
│   └── model.yaml
│
├── tests/
│   ├── test_data_alignment.py
│   ├── test_mass_balance.py
│   ├── test_api.py
│   └── test_model_shapes.py
│
└── docs/
    ├── data_sources.md
    ├── equations.md
    ├── methodology.md
    └── experiment_log.md
```

This is a recommended structure. Directories and files should be created
as implementation proceeds.

------------------------------------------------------------------------

## 16. Technology stack summary

  -----------------------------------------------------------------------
  Layer                               Recommended technology
  ----------------------------------- -----------------------------------
  GIS inspection                      QGIS

  Raster processing                   GDAL, Rasterio, Xarray

  Vector processing                   GeoPandas, Shapely

  Physics simulation                  ANUGA

  Secondary physics comparison        LISFLOOD-FP

  Scientific computing                Python, NumPy, SciPy

  Deep learning                       PyTorch

  Spatial feature model               CNN

  Temporal model                      ConvLSTM

  Physics-informed objective          Data + mass + momentum + positivity
                                      losses

  Backend                             FastAPI

  Background jobs                     Worker / task queue

  Frontend                            Next.js, React, TypeScript

  Styling                             Tailwind CSS

  Interactive map                     MapLibre GL JS

  Charts                              Recharts

  Data storage                        NetCDF / Zarr, GeoTIFF, GeoJSON /
                                      GeoPackage

  Version control                     Git and GitHub
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 17. Implementation roadmap

### Phase 1 --- Study-area and data preparation

-   [ ] Finalize Bellandur pilot boundary using terrain and drainage
    connectivity.
-   [ ] Obtain DEM and verify vertical datum / quality.
-   [ ] Obtain land cover, soil, buildings, roads and drainage layers.
-   [ ] Obtain rainfall observations or define labelled synthetic
    scenarios.
-   [ ] Align and document all layers.

### Phase 2 --- Physics baseline

-   [ ] Install ANUGA and run a documented example.
-   [ ] Build the pilot computational mesh.
-   [ ] Set initial and boundary conditions.
-   [ ] Configure rainfall, roughness, infiltration and drainage
    assumptions.
-   [ ] Run a simple dry-start rainfall event.
-   [ ] Check water balance and wet/dry behaviour.
-   [ ] Export depth and momentum at fixed intervals.

### Phase 3 --- Dataset generation

-   [ ] Define a scenario matrix.
-   [ ] Run multiple rainfall events.
-   [ ] Export state and forcing variables.
-   [ ] Rasterize / align ANUGA outputs if required by the neural
    architecture.
-   [ ] Create scenario-level train, validation and test splits.
-   [ ] Record solver settings and provenance.

### Phase 4 --- Surrogate model

-   [ ] Build CNN spatial encoder.
-   [ ] Add ConvLSTM temporal module.
-   [ ] Predict depth and momentum.
-   [ ] Implement data loss and physical residual losses.
-   [ ] Train on simulation sequences.
-   [ ] Evaluate one-step and multi-step forecasts.
-   [ ] Test unseen rainfall patterns and parameter ranges.

### Phase 5 --- Backend and frontend

-   [ ] Build FastAPI health, scenario, job and prediction routes.
-   [ ] Implement background inference jobs.
-   [ ] Build the Next.js dashboard.
-   [ ] Add MapLibre depth and flood-extent layers.
-   [ ] Add rainfall controls and forecast timeline.
-   [ ] Add velocity arrows, charts and scenario comparison.
-   [ ] Connect the UI to the backend.

### Phase 6 --- Evaluation and reporting

-   [ ] Report MAE, RMSE and flood-extent IoU.
-   [ ] Report water-mass balance and physical residuals.
-   [ ] Report long-horizon stability.
-   [ ] Benchmark solver and surrogate runtime on documented hardware.
-   [ ] Validate against independent observations where available.
-   [ ] Document limitations and uncertainty.

------------------------------------------------------------------------

## 18. Risks, assumptions and limitations

1.  **DEM quality:** A coarse or surface-model DEM may not represent
    road crowns, underpasses, curbs and drainage channels.
2.  **Drainage data:** Public drain maps may not include capacity,
    invert levels, inlets, blockages or complete connectivity.
3.  **Rainfall resolution:** Daily gridded rainfall is not sufficient
    for minute-scale urban flood forecasting.
4.  **Soil parameter uncertainty:** Soil texture is not a direct
    substitute for measured hydraulic conductivity or infiltration
    curves.
5.  **Urban representation:** Buildings and roads require deliberate
    representation in the mesh / raster and may require sub-grid
    methods.
6.  **Model discrepancy:** ANUGA and LISFLOOD-FP can produce different
    results due to solver assumptions and discretization.
7.  **Surrogate extrapolation:** Neural predictions can fail outside the
    rainfall, terrain and parameter ranges represented in training.
8.  **Long-horizon drift:** Small one-step errors may accumulate during
    autoregressive forecasting.
9.  **Operational use:** The prototype is a research tool, not an
    emergency warning system. Forecasts need uncertainty communication
    and independent validation before operational use.

------------------------------------------------------------------------

## 19. Research contribution

The intended contribution is not merely a flood map or a conventional
dashboard. It is the combination of:

-   a reproducible urban hydrodynamic simulation dataset;
-   a spatiotemporal neural surrogate trained on physics-generated
    sequences;
-   explicit physical consistency constraints;
-   multi-step stability evaluation;
-   comparison against a numerical reference solver;
-   an interactive interface that lets users change rainfall scenarios
    and inspect forecast evolution.

The research question is:

> Can a physics-informed spatiotemporal neural surrogate reproduce the
> flood dynamics of a high-resolution hydrodynamic model with useful
> accuracy and physical consistency, while reducing forecast inference
> time enough to support interactive urban flood scenario exploration?

------------------------------------------------------------------------

## 20. References and official resources

### Physics models

-   [ANUGA documentation](https://anuga.readthedocs.io/en/stable/)
-   [ANUGA source
    repository](https://github.com/GeoscienceAustralia/anuga_core)
-   [LISFLOOD-FP -- University of
    Bristol](https://www.bristol.ac.uk/geography/research/hydrology/models/lisflood/)
-   [HEC-RAS
    documentation](https://www.hec.usace.army.mil/software/hec-ras/) ---
    alternative hydraulic model for future comparison.

### Geospatial and rainfall data

-   [Copernicus Data Space](https://dataspace.copernicus.eu/)
-   [ESA WorldCover](https://esa-worldcover.org/en/data-access)
-   [ISRIC SoilGrids](https://soilgrids.org/)
-   [OpenStreetMap](https://www.openstreetmap.org/)
-   [Geofabrik India
    extracts](https://download.geofabrik.de/asia/india.html)
-   [IMD gridded
    rainfall](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html)
-   [OpenCity Bengaluru data portal](https://data.opencity.in/)

### Frontend and backend

-   [Next.js documentation](https://nextjs.org/docs)
-   [Tailwind CSS documentation](https://tailwindcss.com/docs)
-   [MapLibre GL JS
    documentation](https://maplibre.org/maplibre-gl-js/docs/)
-   [Recharts documentation](https://recharts.org/)
-   [FastAPI documentation](https://fastapi.tiangolo.com/)
-   [PyTorch documentation](https://pytorch.org/docs/stable/)

------------------------------------------------------------------------

## 21. Reproducibility checklist

Before publishing results, record:

-   study-area polygon and CRS;
-   input dataset names, versions, dates and licenses;
-   DEM source, resolution and vertical datum;
-   mesh / raster resolution and cell count;
-   roughness and infiltration parameter sources;
-   rainfall hyetographs and units;
-   initial and boundary conditions;
-   physics solver version and configuration;
-   simulation output interval and time-step settings;
-   neural architecture, hyperparameters, random seeds and checkpoint;
-   train / validation / test scenario IDs;
-   evaluation hardware and runtime measurement procedure;
-   all accuracy, mass-balance and stability metrics.

**Project principle:** Use the physics solver to establish the reference
dynamics, the neural surrogate to accelerate prediction, and independent
observations to determine whether either is representative of real
flooding.
