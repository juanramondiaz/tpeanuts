#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on My 11 2022

@author Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""

import numpy as np
import copy

from peanuts.pmns import PMNS

class Param:

  def __init__(self, label, value):

    self.nparams = 1
    setattr(self,label, value)

  def add(self, label, value):
    self.nparams += 1
    setattr(self,label, value)

  def __repr__(self):
    ret = "["
    for attr in dir(self):
      if not attr.startswith("__") and not attr == "add" and not attr == "nparams":
        ret  += attr + " : " + str(getattr(self, attr)) + ", "
    ret = ret[:-2]+"]"
    return ret

class Scan:

  def __init__(self):

    self.params = list()
    self.labels = list()
    self._index = 0

  def __iter__(self):
    return self

  def __next__(self):
    if self._index < len(self.params):
      self._index += 1
      return self.params[self._index-1]
    else:
      raise StopIteration

  def __len__(self):
    return len(self.params)

  def enumerate(self):
    return enumerate(self.params)

  def add(self, label, param):

    if isinstance(param, list):
      # Assume it is given as [min, max], [min, max, step] or [min, max, step, mode]
      if len(param) < 2 or len(param) > 4:
        print("Error: Parameter", label, "should be given as single number or as range [min, max, (step), (mode)].")
        exit()

      self.labels.append(label)

      parammin = float(param[0])
      parammax = float(param[1])

      if len(param) > 2:
        step = float(param[2])
        N = int( (parammax-parammin)/step)+1
      else:
        # If step is not given, assume 10 iterations
        N = 10
        step = (parammax-parammin)/(N-1)

      # Set scan mode
      mode = "linear"
      if len(param) == 4:
        if param[3] == "log":
          mode = "log"
        elif not param[3] == "linear":
          print("Error: Unknown scan mode `"+param[3]+"`. It should be \"linear\" or \"log\".")

      # Set values, depdending on whether we are in linear or log mode
      if mode == "linear":
        values = [parammin + i*step for i in range(0,N)]
      elif mode == "log":
        values = [10**(parammin + i*step) for i in range(0,N)]

      if len(self.params):
        newparams = list()
        for par in self.params:
          for val in values:
            newparam = copy.copy(par)
            newparam.add(label,val)
            newparams.append(newparam)
        self.params = newparams
      else:
        for val in values:
          self.params.append(Param(label,val))

    else:
      if len(self.params):
        for par in self.params:
          par.add(label, param)
      else:
        self.params.append(Param(label,param))


class Settings:

  def __init__(self, *args):

    self.vacuum = False
    self.solar = False
    self.earth = False
    self.scan = Scan()

    # If there is only one argument, it is a settings dictionary
    if len(args) == 1:
      settings = args[0]

      # Select mode first
      if "vacuum" in settings:
        self.vacuum = True
      if "solar" in settings:
        self.solar = True
      if "earth" in settings:
        self.earth = True
      if not self.vacuum and not self.solar and not self.earth:
        print("Error: unkown mode, please provide a running mode, \"vacuum\", \"solar\", \"earth\" or a combination of them")
        exit()

      # Extract neutrino parameters
      if "Neutrinos" not in settings:
        print("Error: missing neutrino information, please provide dm21, dm3l, theta12, theta13, theta23 and delta")
        exit()
      elif isinstance(settings["Neutrinos"], dict):
        if "dm21" not in settings["Neutrinos"] or\
           "dm3l" not in settings["Neutrinos"] or\
           "theta12" not in settings["Neutrinos"] or\
           "theta13" not in settings["Neutrinos"] or\
           "theta23" not in settings["Neutrinos"] or\
           "delta" not in settings["Neutrinos"] :
          print("Error: missing neutrino information, please provide dm21, dm3l, theta12, theta13, theta23 and delta")
          exit()
        else:
          self.dm21 = settings["Neutrinos"]["dm21"]
          self.dm3l = settings["Neutrinos"]["dm3l"]
          self.theta12 = settings["Neutrinos"]["theta12"]
          self.theta13 = settings["Neutrinos"]["theta13"]
          self.theta23 = settings["Neutrinos"]["theta23"]
          self.delta = settings["Neutrinos"]["delta"]

          self.scan.add("dm21", self.dm21)
          self.scan.add("dm3l", self.dm3l)
          self.scan.add("theta12", self.theta12)
          self.scan.add("theta13", self.theta13)
          self.scan.add("theta23", self.theta23)
          self.scan.add("delta", self.delta)
      else:

        import peanuts.files as f

        slha_file = settings["Neutrinos"]
        try:
          nu_params = f.read_slha(slha_file)
          th12 = nu_params['theta12']
          th13 = nu_params['theta13']
          th23 = nu_params['theta23']
          d = nu_params['delta']
          self.pmns = PMNS(th12, th13, th23, d)
          self.dm21 = nu_params['dm21']
          self.dm3l = nu_params['dm3l']
        except FileNotFoundError:
          print("Error: slha file " + slha_file + " not found.")
          exit()

      # Extract vacuum parameters
      if "vacuum" in settings:
        if "solar" in settings or "earth" in settings:
          print("Error: vacuum mode can only be used on its own")
          exit()
        elif "state" not in settings["vacuum"] or "basis" not in settings["vacuum"] or "baseline" not in settings["vacuum"]:
          print("Error: vacuum oscillations require an input state, its basis and the baseline")
          exit()
        else:
          self.nustate = np.array(settings["vacuum"]["state"],dtype=complex)
          self.antinu = settings["vacuum"]["antinu"] if "antinu" in settings["vacuum"] else False
          self.basis = settings["vacuum"]["basis"]
          self.baseline = settings["vacuum"]["baseline"]
          self.scan.add("baseline", self.baseline)
          self.probabilities = settings["vacuum"]["probabilities"] if "probabilities" in settings["vacuum"] else True
          self.evolved_state = settings["vacuum"]["evolved_state"] if "evolved_state" in settings["vacuum"] else False

      # Extract solar parameters
      if "solar" in settings:

        if "fraction" not in settings["solar"]:
          print("Error: missing solar neutrino fraction.")
          exit()
        else:
          self.fraction = settings["solar"]["fraction"]

        self.antinu = False
        self.solar_file = settings["solar"]["solar_model"] if "solar_model" in settings["solar"] else None
        self.flux_file = settings["solar"]["flux_file"] if "flux_file" in settings["solar"] else None
        self.fluxrows = settings["solar"]["fluxrows"] if "fluxrows" in settings["solar"] else None
        self.fluxcols = settings["solar"]["fluxcols"] if "fluxcols" in settings["solar"] else None
        self.fluxscale = settings["solar"]["fluxscale"] if "fluxscale" in settings["solar"] else None
        self.distrow = settings["solar"]["distrow"] if "distrow" in settings["solar"] else None
        self.radiuscol = settings["solar"]["radiuscol"] if "radiuscol" in settings["solar"] else None
        self.densitycol = settings["solar"]["densitycol"] if "densitycol" in settings["solar"] else None
        self.fractioncols = settings["solar"]["fractioncols"] if "fractioncols" in settings["solar"] else None

        self.spectra = settings["solar"]["spectra"] if "spectra" in settings["solar"] else None

        self.probabilities = settings["solar"]["probabilities"] if "probabilities" in settings["solar"] else True

        self.flux = settings["solar"]["flux"] if "flux" in settings["solar"] else False

        self.undistorted_spectrum = False
        self.distorted_spectrum =  False
        if "spectrum" in settings["solar"]:
          if settings["solar"]["spectrum"] == "undistorted":
            self.undistorted_spectrum = True
          elif settings["solar"]["spectrum"] == "distorted":
            self.distorted_spectrum = True
          else:
            print("Error: unknown option for spectrum, select undistorted or distorted")
            exit()

      # Extract earth parameters
      if "earth" in settings:

        self.antinu = False
        if "solar" not in settings and\
           ("state" not in settings["earth"] or "basis" not in settings["earth"]):
          print("Error: missing input neutrino state or basis, please provide both.")
          exit()
        elif "solar" not in settings:
          self.nustate = np.array(settings["earth"]["state"],dtype=complex)
          self.antinu = settings["earth"]["antinu"] if "antinu" in settings["earth"] else False
          self.basis = settings["earth"]["basis"]
          self.probabilities = True

        if "depth" not in settings["earth"]:
          print("Error: missing depth of experiment, please provide it.")
          exit()
        else:
          self.depth = settings["earth"]["depth"]

        # Either a specific nadir angle, eta, or a latitude or exposure file must be provided
        if "eta" not in settings["earth"] and "latitude" not in settings["earth"] and not "exposure_file" in settings["earth"]:
          print("Error: please provide a nadir angle (eta), a latitude or exposure file path.")
          exit()
        elif "eta" in settings["earth"]:
          self.eta = settings["earth"]["eta"]
          self.exposure = False
          self.scan.add("eta", self.eta)
        elif "latitude" in settings["earth"] or "exposure_file" in settings["earth"]:
          if("latitude" in settings["earth"] and "exposure_file" in settings["earth"]):
            print("Warning: both latitude and exposure file provided, latitude value will be ignored")
          self.exposure = True
          self.latitude = settings["earth"]["latitude"] if "latitude" in settings["earth"] and "exposure_file" not in settings["earth"] else -1
          self.exposure_normalized = settings["earth"]["exposure_normalized"] if "exposure_normalized" in settings["earth"] else False
          self.exposure_time = settings["earth"]["exposure_time"] if "exposure_time" in settings["earth"] else [0,365]
          self.exposure_samples = settings["earth"]["exposure_samples"] if "exposure_samples" in settings["earth"] else 1000
          self.exposure_file = settings["earth"]["exposure_file"] if "exposure_file" in settings["earth"] else None
          self.exposure_angle = settings["earth"]["exposure_angle"] if "exposure_angle" in settings["earth"] else "Nadir"
        else:
          print("Error: a nadir angle (eta) and exposure option (latitude or file) were found, please provide only one of them.")
          exit()

        self.density_file = settings["earth"]["density"] if "density" in settings["earth"] else None
        self.custom_density = settings["earth"]["custom_density"] if "custom_density" in settings["earth"] else False
        self.tabulated_density = settings["earth"]["tabulated_density"] if "tabulated_density" in settings["earth"] else False
        self.evolution = settings["earth"]["evolution"] if "evolution" in settings["earth"] else "analytical"
        self.evolved_state = settings["earth"]["evolved_state"] if "evolved_state" in settings["earth"] else False

      # Extract energy
      if "Energy" not in settings:
        print("Error: missing energy, please provide value or range.")
        exit()
      else:
        self.energy = settings["Energy"]
        self.scan.add("energy", self.energy)

      # Select printing mode, default is stdout
      if "Output" in settings:
        self.output = settings["Output"]
      else:
        self.output = "stdout"

    # If there are exactly 6 arguments, we are in solar mode
    # args = (pmns, dm21, dm3l, E, fraction, options)
    elif len(args) == 6:

      self.solar = True
      self.pmns = args[0]
      self.theta12 = self.pmns.theta12
      self.theta13 = self.pmns.theta13
      self.theta23 = self.pmns.theta23
      self.delta = self.pmns.delta
      self.dm21 = args[1]
      self.dm3l = args[2]
      self.energy = args[3]
      self.fraction = args[4]
      self.solar_file = args[5].solar if args[5].solar != "" else None
      self.probabilities = True
      self.undistorted_spectrum = False
      self.distorted_spectrum =  False

    # If there are exactly 7 arguments, we are in earth mode
    # args = (pmns, dm21, dm3l, E, eta, depth, options)
    elif len(args) == 7:

      self.earth = True
      self.antinu = args[6].antinu
      self.pmns = args[0]
      self.theta12 = self.pmns.theta12
      self.theta13 = self.pmns.theta13
      self.theta23 = self.pmns.theta23
      self.delta = self.pmns.delta
      self.dm21 = args[1]
      self.dm3l = args[2]
      self.energy = args[3]
      self.eta = args[4]
      self.depth = args[5]

      if args[6].flavour is not None:
        self.basis = "flavour"
        self.nustate = args[6].flavour
      elif args[6].mass is not None:
        self.basis = "mass"
        self.nustate = args[6].mass
      else:
        print("Error: unknown basis, please choose either flavour or mass basis.")
        exit()

      self.density_file = args[6].density if args[6].density != "" else None
      self.evolution = "analytical" if args[6].analytical else "numerical"

      self.exposure = False

