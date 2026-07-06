# Diagrama de modulos y flujo de pipelines

Este documento resume la estructura funcional de `tpeanuts` a partir del codigo
actual del proyecto. El objetivo es separar los niveles de responsabilidad:
configuracion y orquestacion, modelos fisicos por medio, nucleo numerico, datos
externos y salidas.

## Vista general de modulos

```mermaid
flowchart TB
    Runner["Scripts y notebooks<br/>notebooks/runs/*<br/>run_*.py"]
    Pipeline["pipeline/<br/>orquestacion de workflows"]
    Config["pipeline/config.py<br/>PropagationConfig"]
    Common["pipeline/pipeline_common.py<br/>preparacion comun"]
    IO["pipeline/io.py<br/>persistencia y agregacion"]

    SolarPipe["pipeline_coherent.py<br/>pipeline_incoherent.py<br/>pipeline_legacypeanuts.py"]
    AtmosPipe["pipeline_atmosphere.py<br/>atmosphere_flux.py"]

    MediumSolar["medium/solar/<br/>perfil, IO, mezcla en materia,<br/>probabilidades solares"]
    MediumVacuum["medium/vacuum/<br/>evolucion en vacio"]
    MediumEarth["medium/earth/<br/>perfil terrestre, geometria,<br/>exposicion nadir, evolucion"]
    MediumAtmos["medium/atmosphere/<br/>densidad, geometria,<br/>evolucion atmosferica, IO"]

    Core["core/common + core/numerical + core/perturbative<br/>PMNS, Hamiltonianos, potenciales,<br/>evolutores, probabilidades"]
    Coherent["coherent/<br/>evolucion coherente solar"]
    External["external/honda + external/mceq + external/pymsis<br/>flujos atmosfericos y backends externos"]
    Legacy["peanuts/<br/>implementacion legacy NumPy/Numba"]
    Data["data/<br/>perfiles solares, densidad Tierra,<br/>tablas SNO, espectros, fluxes"]
    Util["util/<br/>contexto runtime, tipos,<br/>constantes, IO, paralelismo"]
    Output["Salidas .pt<br/>metadata + tensores"]

    Runner --> Pipeline
    Pipeline --> Config
    Pipeline --> Common
    Pipeline --> IO
    Pipeline --> SolarPipe
    Pipeline --> AtmosPipe

    SolarPipe --> MediumSolar
    SolarPipe --> MediumVacuum
    SolarPipe --> MediumEarth
    SolarPipe --> Coherent
    SolarPipe --> Legacy

    AtmosPipe --> MediumAtmos
    AtmosPipe --> MediumEarth
    AtmosPipe --> External

    MediumSolar --> Core
    MediumVacuum --> Core
    MediumEarth --> Core
    MediumAtmos --> Core
    Coherent --> Core
    Legacy --> Data

    Common --> MediumSolar
    Common --> MediumEarth
    Common --> Data
    Config --> Core
    Config --> MediumSolar
    Config --> MediumEarth
    Config --> MediumAtmos

    MediumSolar --> Data
    MediumEarth --> Data
    MediumAtmos --> Data
    External --> Data

    Pipeline --> Util
    MediumSolar --> Util
    MediumEarth --> Util
    MediumAtmos --> Util
    Core --> Util

    IO --> Output
    SolarPipe --> Output
    AtmosPipe --> Output
```

## Pipeline solar: produccion a detector

El pipeline solar tiene tres rutas de ejecucion: coherente torch-native,
incoherente torch-native y legacy `peanuts`. Las tres comparten configuracion,
preparacion de perfiles y formato de salida.

```mermaid
flowchart LR
    Inputs["Entradas<br/>E_MeV, fuente solar, parametros PMNS,<br/>perfil solar, densidad Tierra,<br/>distancia Sol-Tierra, exposicion nadir"]
    Config["PropagationConfig<br/>runtime, oscillation, exposure,<br/>earth, solar, production_mode"]
    Common["pipeline_common<br/>prepare_earth_profile<br/>prepare_earth_distance<br/>prepare_initial_state"]

    subgraph CoherentPath["Ruta coherente: pipeline_coherent.py"]
        ProdGrid["Grid de produccion rho<br/>point/coherent/incoherent"]
        SolarState["coherent.evolution.solar_surface_state<br/>estado en superficie solar"]
        Vacuum["medium.vacuum.evolutor<br/>propagacion superficie solar -> Tierra"]
        EarthOp["medium.earth.evolutor<br/>operador Tierra por eta"]
        CoherentProb["core.common.probability<br/>|psi|^2 en detector"]
        CoherentAvg["Promedio/integracion<br/>rho y exposicion nadir"]
    end

    subgraph IncoherentPath["Ruta incoherente: pipeline_incoherent.py"]
        SolarMass["medium.solar.probability.solar_probability_mass<br/>pesos de autoestados de masa"]
        Psolar["medium.solar.probability.psolar<br/>probabilidades solares"]
        Pearth["medium.earth.probability.pearth<br/>massbasis=True"]
        IncohAvg["Integracion sobre exposicion nadir"]
    end

    subgraph LegacyPath["Ruta legacy: pipeline_legacypeanuts.py"]
        LegacySolar["peanuts.solar.solar_flux_mass / Psolar"]
        LegacyEarth["peanuts.earth.Pearth<br/>massbasis=True"]
        LegacyAvg["Integracion legacy de exposicion"]
    end

    Output["Resultado .pt<br/>metadata, grids, probabilidades,<br/>estados u operadores segun modo"]

    Inputs --> Config --> Common
    Common --> ProdGrid --> SolarState --> Vacuum --> EarthOp --> CoherentProb --> CoherentAvg --> Output
    Common --> SolarMass --> Psolar --> Pearth --> IncohAvg --> Output
    Common --> LegacySolar --> LegacyEarth --> LegacyAvg --> Output
```

## Pipeline atmosferico: flujo de produccion a detector

El flujo atmosferico parte de tablas externas o generadas por MCEq/Honda,
selecciona una particula y angulo, propaga estados coherentes por atmosfera y
Tierra, y finalmente integra sobre la altura de produccion.

```mermaid
flowchart LR
    FluxSource["external/mceq o external/honda<br/>tablas Phi_beta(E, theta, h)"]
    FluxIO["medium.atmosphere.io<br/>load_directory"]
    Select["pipeline_atmosphere.select_particle_angle_flux<br/>particula + angulo"]
    Traj["build_atmosphere_trajectories<br/>L_atm, h_path, s_path"]
    AtmEvol["medium.atmosphere.evolutor<br/>S_atm(E,h,theta)"]
    Surface["surface_states<br/>probabilidades en superficie"]
    EarthProfile["pipeline_common.prepare_earth_profile<br/>EarthProfile"]
    EarthEvol["medium.earth.evolutor_from_zenith<br/>S_earth(E,theta)"]
    Detector["detector_states<br/>P(beta -> i)"]
    Integrate["integrate_initial_and_surface_fluxes<br/>integrate_height_and_sum_flavours"]
    Save["pipeline.io.save_detector_flux_result<br/>detector_flux_*.pt"]

    FluxSource --> FluxIO --> Select
    Select --> Traj
    Select --> AtmEvol --> Surface
    Surface --> EarthEvol --> Detector
    EarthProfile --> EarthEvol
    Select --> Integrate
    Surface --> Integrate
    Detector --> Integrate --> Save
```

## Flujo alternativo atmosferico por matriz de probabilidad

`pipeline/atmosphere_flux.py` ofrece una ruta mas compacta para propagar
directamente un vector o una malla de flujos usando
`P = |S_earth S_atm|^2`.

```mermaid
flowchart LR
    PhiIn["Phi(E,h)<br/>[nue, numu, nutau]"]
    BuildP["build_probability_matrix<br/>S_atm + S_earth"]
    P["P = |S_earth S_atm|^2"]
    Apply["probability_incoherent<br/>Phi_det = P Phi_prod"]
    Height["integrate_detector_flux_over_height"]
    PhiOut["Flujo detector<br/>Phi_det(E, flavour)"]

    PhiIn --> BuildP --> P --> Apply --> Height --> PhiOut
```

## Responsabilidades clave

- `pipeline/`: define workflows de alto nivel, decide que perfiles construir,
  que grids usar, como combinar etapas y como guardar resultados.
- `pipeline/config.py`: concentra parametros de runtime, oscilacion, medio
  solar, Tierra, atmosfera, exposicion y modo de produccion.
- `pipeline/pipeline_common.py`: prepara objetos compartidos como
  `EarthProfile`, distancia Sol-Tierra y estado inicial.
- `medium/*`: contiene la fisica especifica de cada medio: solar, vacio,
  Tierra y atmosfera.
- `core/*`: contiene los bloques de bajo nivel comunes: PMNS, Hamiltonianos,
  potenciales, operadores de evolucion y conversion a probabilidades.
- `external/*`: genera o adapta integraciones externas, especialmente
  MCEq/Honda para flujos atmosfericos y PyMSIS para densidad atmosferica.
- `peanuts/`: implementacion legacy usada para compatibilidad y validacion.
- `data/`: datos de entrada versionados o de referencia.
- `notebooks/runs/*` y `run_*.py`: scripts consumidores que parametrizan y
  lanzan los pipelines.

## Salidas principales

- Solar:
  - `run_and_save_solar_to_detector_coherent`
  - `run_and_save_solar_to_detector_incoherent`
  - `run_and_save_solar_to_detector_legacypeanuts`
  - salida: `.pt` con `metadata`, grids, probabilidades y, segun modo,
    estados coherentes u operadores.
- Atmosfera:
  - `save_detector_flux_result`
  - salida: `.pt` por particula y angulo, con `detector_flux_Ei`,
    `surface_flux_Ei`, `initial_flux_Ei`, probabilidades y metadata.
